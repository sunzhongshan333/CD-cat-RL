"""
utils/health_monitor.py

全流程健康监控与预警系统
========================================
在训练（NCDM、RL）和评估全程实时检测数值异常与性能退化，
通过 logging.WARNING / ERROR 级别高亮显示预警，并给出可操作的诊断建议。

检测范围
--------
NCDM 训练 (NCDMTrainingMonitor)
  · NaN/Inf 训练损失（立即 ERROR）
  · 验证 AUC 持续不提升（连续 ≥3 epoch WARNING）
  · 最终 AUC 低于可用阈值（训练结束 WARNING）

RL 训练 (RLTrainingMonitor)
  · NaN/Inf 损失（每次梯度更新后，ERROR）
  · 梯度范数超过裁剪阈值 5 倍（每次梯度更新后，WARNING）
  · D3QN Q 值标准差过低（策略退化，每 CHECK_INTERVAL ep，WARNING）
  · 早停率长期为 0（tau/beta 配置问题，每 CHECK_INTERVAL ep，WARNING）
  · 奖励信号过弱（每 CHECK_INTERVAL ep，WARNING）
  · E-step / Q-step 损失上升（每 CHECK_INTERVAL ep，WARNING）

评估 (EvalMonitor)
  · Track A: MSE 过高、逐知识点准确率接近随机、早停率极低、F1 不足
  · Track B: RL ≈ random（策略未学习）、RL < 全量基线、AUC 过低、
             RL 与 Random 平均题数相同（从未自适应早停）

配置预检 (check_rl_config)
  · CDCAT_BETA × 典型熵减 << CDCAT_STEP_COST（奖励信号不可区分）
  · CDCAT_TAU ≥ 最大二值熵（早停永远无法触发）
  · RL_EPSILON_DECAY 超过总轮次 50%（探索过慢）
"""

import math
import sys
import logging
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 终端颜色支持检测
# ---------------------------------------------------------------------------
_USE_COLOR = sys.stdout.isatty() or hasattr(sys.stdout, "buffer")

_C_WARN  = "\033[93m"   # 黄色
_C_ERROR = "\033[91m"   # 红色
_C_OK    = "\033[92m"   # 绿色
_C_INFO  = "\033[96m"   # 青色
_C_RESET = "\033[0m"


def _c(code: str, text: str) -> str:
    """按需添加颜色转义。"""
    if _USE_COLOR:
        return f"{code}{text}{_C_RESET}"
    return text


def _fmt_warn(msg: str) -> str:
    return _c(_C_WARN, f"[⚠  WARN ] {msg}")


def _fmt_error(msg: str) -> str:
    return _c(_C_ERROR, f"[✖ ERROR ] {msg}")


def _fmt_ok(msg: str) -> str:
    return _c(_C_OK, f"[✔  OK   ] {msg}")


def _fmt_info(msg: str) -> str:
    return _c(_C_INFO, f"[ℹ  INFO ] {msg}")


# ===========================================================================
# 通用数值健康检查工具
# ===========================================================================

def check_tensor_health(tensor, name: str, phase: str = "") -> bool:
    """
    检查 tensor 是否含 NaN / Inf。
    返回 True 表示正常，False 表示异常（已记录 ERROR 日志）。
    """
    if not tensor.isfinite().all():
        nan_cnt = tensor.isnan().sum().item()
        inf_cnt = tensor.isinf().sum().item()
        logger.error(_fmt_error(
            f"{phase} 张量 '{name}' 含异常值: NaN={nan_cnt}, Inf={inf_cnt}。"
            "建议检查学习率、梯度裁剪阈值和输入数据归一化。"
        ))
        return False
    return True


def check_loss_health(loss_val: float, phase: str = "") -> bool:
    """
    检查标量损失是否为 NaN / Inf。
    返回 True 表示正常。
    """
    if not math.isfinite(loss_val):
        logger.error(_fmt_error(
            f"{phase} 损失异常: {loss_val}。"
            "训练可能已发散——建议降低学习率或增大梯度裁剪阈值。"
        ))
        return False
    return True


def compute_grad_norm(model) -> float:
    """计算模型所有参数的全局 L2 梯度范数（在 clip_grad_norm_ 之前调用）。"""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return math.sqrt(total)


def check_grad_norm(grad_norm: float, clip_value: float, phase: str = "") -> None:
    """若梯度范数超过 clip_value 的 5 倍则发出 WARNING。"""
    if grad_norm > clip_value * 5:
        logger.warning(_fmt_warn(
            f"{phase} 梯度范数 {grad_norm:.2f} 超过裁剪阈值 {clip_value} 的 5 倍，"
            "存在梯度爆炸风险。请检查奖励尺度（CDCAT_BETA）或网络结构。"
        ))


# ===========================================================================
# RL 超参数配置预检
# ===========================================================================

def check_rl_config(cfg) -> None:
    """
    在 RL 训练开始前验证关键超参数的合理性。
    发现问题时输出 WARNING，全部通过时输出 OK。

    Parameters
    ----------
    cfg : module
        已导入的 config 模块（需含 CDCAT_BETA/TAU/STEP_COST/RL_* 等属性）
    """
    import math as _math
    issues = []

    # 1. 报酬信号强度: CDCAT_BETA × 典型熵减 是否可超过 CDCAT_STEP_COST
    #    典型单步熵减估算：总熵下降 ≈ ln(2)/2，平均到 max_steps 步
    typical_entropy_reduction_per_step = math.log(2) / 2 / max(cfg.RL_MAX_STEPS, 1)
    expected_reward = cfg.CDCAT_BETA * typical_entropy_reduction_per_step
    if expected_reward < cfg.CDCAT_STEP_COST * 0.2:
        issues.append(
            f"CDCAT_BETA ({cfg.CDCAT_BETA}) × 典型单步熵减"
            f" ({typical_entropy_reduction_per_step:.5f})"
            f" ≈ {expected_reward:.5f}，远小于 CDCAT_STEP_COST ({cfg.CDCAT_STEP_COST})。"
            "D3QN 几乎无法区分好坏动作，建议增大 CDCAT_BETA 或减小 CDCAT_STEP_COST。"
        )

    # 2. 早停阈值：tau 若 ≥ 最大二值熵则永不触发
    max_binary_entropy = math.log(2)   # ≈ 0.693
    if cfg.CDCAT_TAU >= max_binary_entropy:
        issues.append(
            f"CDCAT_TAU ({cfg.CDCAT_TAU:.3f}) ≥ 最大二值熵 ({max_binary_entropy:.3f})，"
            "基于平均熵的早停永远无法触发，建议将 tau 设为 0.1–0.5。"
        )

    # 3. epsilon_decay 相对于总轮次是否合理
    if cfg.RL_EPSILON_DECAY > cfg.RL_MAX_EPISODES * 0.8:
        issues.append(
            f"RL_EPSILON_DECAY ({cfg.RL_EPSILON_DECAY}) 超过总轮次 80%，"
            "绝大多数训练时间处于高探索率，D3QN 可能来不及收敛。"
            "建议将 epsilon_decay 设为总轮次的 20%–50%。"
        )

    # 4. 缓冲区容量与 max_episodes × max_steps 的比例
    buffer_fill = cfg.RL_MAX_STEPS * cfg.RL_MAX_EPISODES
    if cfg.RL_BUFFER_CAPACITY > buffer_fill * 2:
        issues.append(
            f"RL_BUFFER_CAPACITY ({cfg.RL_BUFFER_CAPACITY}) 远大于"
            f" max_episodes × max_steps ({buffer_fill})，"
            "缓冲区可能永远不会被充分利用。"
        )

    if issues:
        logger.warning(_fmt_info("[配置预检] 发现以下潜在问题，建议在训练前确认："))
        for i, issue in enumerate(issues, 1):
            logger.warning(_fmt_warn(f"  [{i}] {issue}"))
    else:
        logger.info(_fmt_ok("RL 配置预检通过——超参数设置合理。"))


# ===========================================================================
# NCDM 训练监控
# ===========================================================================

class NCDMTrainingMonitor:
    """
    监控 NCDM 训练过程中的性能退化和数值异常。

    使用方法
    --------
    monitor = NCDMTrainingMonitor()
    for epoch in ...:
        ...
        monitor.check_epoch(epoch, total_epochs, train_loss, val_auc, val_rmse)
    monitor.final_check()
    """

    AUC_MIN_THRESHOLD = 0.65       # 最终 AUC 推荐下限
    PLATEAU_PATIENCE  = 3          # 连续不提升 epoch 数达到此值才告警

    def __init__(self):
        self.auc_history   = []
        self.best_auc      = 0.0
        self.plateau_count = 0     # 连续未提升 epoch 计数

    def check_epoch(self, epoch: int, total_epochs: int,
                    train_loss: float, val_auc: float, val_rmse: float) -> None:
        """每个 epoch 结束后调用。"""
        # 1. 损失 NaN/Inf
        check_loss_health(train_loss, phase=f"[NCDM Epoch {epoch}/{total_epochs}]")

        if not math.isfinite(val_auc) or not math.isfinite(val_rmse):
            logger.error(_fmt_error(
                f"Epoch {epoch}/{total_epochs}: 验证指标异常"
                f" (AUC={val_auc}, RMSE={val_rmse})，"
                "请检查数据集标签分布或模型初始化。"
            ))
            return

        # 2. AUC 提升跟踪
        if val_auc > self.best_auc:
            self.best_auc      = val_auc
            self.plateau_count = 0
        else:
            self.plateau_count += 1
            if self.plateau_count >= self.PLATEAU_PATIENCE:
                logger.warning(_fmt_warn(
                    f"Epoch {epoch}/{total_epochs}: 验证 AUC 已连续"
                    f" {self.plateau_count} 个 epoch 未提升"
                    f"（当前 {val_auc:.4f}，最佳 {self.best_auc:.4f}）。"
                    "学习率调度器应已触发；若仍不收敛，"
                    "建议检查 NCDM_LR、NCDM_EPOCHS 或数据质量。"
                ))

        self.auc_history.append(val_auc)

    def final_check(self) -> None:
        """训练结束后调用，输出整体质量摘要。"""
        if self.best_auc < self.AUC_MIN_THRESHOLD:
            logger.warning(_fmt_warn(
                f"NCDM 最终最佳 AUC {self.best_auc:.4f} 低于推荐阈值"
                f" {self.AUC_MIN_THRESHOLD}。"
                "后续 RL 训练中编码器无法从教师软标签获得有效监督信号，"
                "建议重新训练 NCDM（增加 epochs 或调整学习率）。"
            ))
        else:
            logger.info(_fmt_ok(
                f"NCDM 训练健康：最佳验证 AUC {self.best_auc:.4f}"
                f" ≥ 推荐阈值 {self.AUC_MIN_THRESHOLD}。"
            ))


# ===========================================================================
# RL 训练监控
# ===========================================================================

class RLTrainingMonitor:
    """
    监控 RL（D3QN + Encoder）全程训练健康状态。

    检测点
    ------
    1. 每次梯度更新   → check_update()：损失 NaN/Inf、梯度范数爆炸
    2. D3QN 贪心选择  → check_q_diversity()：Q 值是否退化
    3. 每个 episode   → record_episode()：积累奖励/损失/早停记录
    4. 定期综合检查   → periodic_check()：早停率、奖励强度、损失趋势

    使用方法
    --------
    monitor = RLTrainingMonitor(grad_clip=cfg.RL_GRAD_CLIP,
                                step_cost=cfg.CDCAT_STEP_COST,
                                max_steps=cfg.RL_MAX_STEPS)
    for episode in range(...):
        ...
        # 在贪心选择后（exploitation 路径）：
        monitor.check_q_diversity(q_values, episode)
        # 在每次 backward 后：
        monitor.check_update(loss_val, model, phase_name)
        ...
        monitor.record_episode(ep_reward, early_stopped, e_loss, q_loss)
        monitor.periodic_check(episode)
    """

    CHECK_INTERVAL      = 200      # 综合周期检查间隔（episodes）
    MIN_EARLY_STOP_RATE = 0.02     # 早停率低于此值才告警（近 500 ep 窗口）
    Q_VALUE_STD_MIN     = 0.05     # Q 值标准差低于此值认为策略退化
    REWARD_FLAT_STD     = 0.05     # 奖励标准差低于此值认为"奖励固定"

    def __init__(self, grad_clip: float, step_cost: float, max_steps: int):
        self.grad_clip = grad_clip
        self.step_cost = step_cost
        self.max_steps = max_steps

        # 滚动窗口（大小 = CHECK_INTERVAL）
        self.episode_rewards  = deque(maxlen=self.CHECK_INTERVAL)
        self.e_step_losses    = deque(maxlen=self.CHECK_INTERVAL)
        self.q_step_losses    = deque(maxlen=self.CHECK_INTERVAL)
        self.early_stop_flags = deque(maxlen=500)   # 更大窗口用于早停率

        # 连续 NaN 计数器（连续出现 NaN 才 ERROR，偶发仅 WARNING）
        self._nan_streak = 0

    # ------------------------------------------------------------------
    # 每次梯度更新后调用
    # ------------------------------------------------------------------

    def check_update(self, loss_val: float, model, phase: str) -> bool:
        """
        检查本次梯度更新健康状态。
        返回 False 表示发现严重问题（NaN/Inf），建议调用方记录但继续训练。

        Parameters
        ----------
        loss_val : float   本次 loss.item()
        model    : nn.Module  被优化的模型（用于计算梯度范数）
        phase    : str     如 "E-step" / "Q-step"
        """
        # 1. 损失 NaN/Inf
        if not check_loss_health(loss_val, phase=f"[RL {phase}]"):
            self._nan_streak += 1
            if self._nan_streak >= 5:
                logger.error(_fmt_error(
                    f"[RL {phase}] 已连续 {self._nan_streak} 次出现 NaN/Inf 损失，"
                    "训练极可能已发散。请检查 CDCAT_BETA、学习率和 Q 矩阵数据。"
                ))
            return False

        self._nan_streak = 0

        # 2. 梯度范数（clip 之前调用效果最佳）
        gn = compute_grad_norm(model)
        check_grad_norm(gn, self.grad_clip, phase=f"[RL {phase}]")
        return True

    # ------------------------------------------------------------------
    # D3QN 贪心选择后调用（每 CHECK_INTERVAL ep 检查一次）
    # ------------------------------------------------------------------

    def check_q_diversity(self, q_values, episode: int) -> None:
        """
        检查 Q 值是否退化（所有可用动作 Q 值几乎相同）。

        Parameters
        ----------
        q_values : Tensor  [1, num_items] 已施加掩码的 Q 值
        episode  : int
        """
        if episode % self.CHECK_INTERVAL != 0:
            return
        # 只考虑有效动作（未被 -1e9 掩码）
        valid_q = q_values[q_values > -1e8]
        if valid_q.numel() < 2:
            return
        std = valid_q.std().item()
        if std < self.Q_VALUE_STD_MIN:
            logger.warning(_fmt_warn(
                f"[Episode {episode}] D3QN Q 值标准差 {std:.5f}"
                f" < {self.Q_VALUE_STD_MIN}（可用动作数={valid_q.numel()}），"
                "策略可能已退化为均匀分配。"
                "建议检查奖励信号强度（CDCAT_BETA）或增加 RL 训练轮数。"
            ))

    # ------------------------------------------------------------------
    # 每个 episode 结束后调用
    # ------------------------------------------------------------------

    def record_episode(self, episode_total_reward: float,
                       early_stopped: bool,
                       e_loss: float = None,
                       q_loss: float = None) -> None:
        """积累指标，供 periodic_check 使用。"""
        self.episode_rewards.append(episode_total_reward)
        self.early_stop_flags.append(1 if early_stopped else 0)
        if e_loss is not None and math.isfinite(e_loss):
            self.e_step_losses.append(e_loss)
        if q_loss is not None and math.isfinite(q_loss):
            self.q_step_losses.append(q_loss)

    # ------------------------------------------------------------------
    # 定期综合健康检查（每 CHECK_INTERVAL ep）
    # ------------------------------------------------------------------

    def periodic_check(self, episode: int) -> None:
        """
        综合检查训练健康状态，每 CHECK_INTERVAL 个 episode 调用一次。
        至少积累 CHECK_INTERVAL 个 episode 数据后才开始检查。
        """
        if episode % self.CHECK_INTERVAL != 0 or episode < self.CHECK_INTERVAL:
            return

        issues_found = False

        # ── 1. 早停率 ────────────────────────────────────────────────
        if len(self.early_stop_flags) >= 100:
            esr = float(np.mean(list(self.early_stop_flags)))
            if esr < self.MIN_EARLY_STOP_RATE:
                logger.warning(_fmt_warn(
                    f"[Episode {episode}] 近 {len(self.early_stop_flags)} ep"
                    f" 早停率 {esr*100:.1f}% < {self.MIN_EARLY_STOP_RATE*100:.0f}%。"
                    "诊断成功奖励几乎从未触发，D3QN 缺乏正向学习信号。"
                    "建议：① 确认 CDCAT_TAU 使用的是平均熵；"
                    "② 增大 CDCAT_BETA；③ 增加 RL_MAX_EPISODES。"
                ))
                issues_found = True

        # ── 2. 奖励信号强度 ──────────────────────────────────────────
        if len(self.episode_rewards) >= 50:
            arr = np.array(list(self.episode_rewards))
            r_mean = float(arr.mean())
            r_std  = float(arr.std())
            # 若奖励的方差极低（所有 episode 奖励几乎相同），说明无差异化信号
            worst_ep = -self.step_cost * self.max_steps  # 最差 episode（无成功奖励）
            if r_std < self.REWARD_FLAT_STD and r_mean < worst_ep * 0.8:
                logger.warning(_fmt_warn(
                    f"[Episode {episode}] 近 {len(arr)} ep 平均奖励"
                    f" {r_mean:.3f}（标准差 {r_std:.4f}）极低且几乎固定。"
                    "说明 CDCAT_BETA × 熵减 << CDCAT_STEP_COST，"
                    "D3QN 无法通过奖励区分好坏选题动作。"
                    "建议增大 CDCAT_BETA 或减小 CDCAT_STEP_COST。"
                ))
                issues_found = True

        # ── 3. E-step 损失趋势 ───────────────────────────────────────
        if len(self.e_step_losses) >= self.CHECK_INTERVAL // 2:
            e_arr = np.array(list(self.e_step_losses))
            h = len(e_arr) // 2
            first_h, second_h = e_arr[:h].mean(), e_arr[h:].mean()
            if second_h > first_h * 1.15:      # 上升超 15%
                logger.warning(_fmt_warn(
                    f"[Episode {episode}] E-step 损失呈上升趋势"
                    f"（{first_h:.4f} → {second_h:.4f}）。"
                    "编码器表征学习不稳定，建议降低 RL_LR_ENCODER"
                    " 或减少 E-step 批次大小。"
                ))
                issues_found = True
            elif second_h < first_h * 0.5:
                # 下降超 50% 通常是好兆头，仅作信息输出
                pass

        # ── 4. Q-step 损失趋势 ───────────────────────────────────────
        if len(self.q_step_losses) >= self.CHECK_INTERVAL // 2:
            q_arr = np.array(list(self.q_step_losses))
            h = len(q_arr) // 2
            first_h, second_h = q_arr[:h].mean(), q_arr[h:].mean()
            if second_h > first_h * 1.3:       # 上升超 30%
                logger.warning(_fmt_warn(
                    f"[Episode {episode}] Q-step 损失呈上升趋势"
                    f"（{first_h:.4f} → {second_h:.4f}）。"
                    "D3QN 可能过拟合或目标网络 Polyak τ 过大，"
                    "建议降低 RL_LR_D3QN 或减小 RL_POLYAK_TAU。"
                ))
                issues_found = True

        # ── 汇总 ─────────────────────────────────────────────────────
        if not issues_found:
            esr_pct = (float(np.mean(list(self.early_stop_flags))) * 100
                       if self.early_stop_flags else 0.0)
            r_mean  = (float(np.mean(list(self.episode_rewards)))
                       if self.episode_rewards else 0.0)
            logger.info(_fmt_ok(
                f"[Episode {episode}] 训练健康检查通过。"
                f" 早停率={esr_pct:.1f}%，近期均奖励={r_mean:.3f}"
            ))


# ===========================================================================
# 评估监控
# ===========================================================================

class EvalMonitor:
    """
    评估结果健康检查。

    使用方法
    --------
    EvalMonitor.check_track_a(avg_mse, per_skill_acc, early_stop_rate, f1)
    EvalMonitor.check_track_b(results_dict)
    """

    # Track A 阈值
    MSE_HIGH         = 0.15
    SKILL_ACC_RANDOM = 0.55   # 低于此值视为接近随机猜测
    ESR_LOW          = 0.05   # 早停率低于 5% 则告警
    F1_LOW           = 0.60

    # Track B 阈值
    RL_RANDOM_MIN_DELTA = 0.002   # RL 与 Random AUC 差距小于此值则告警
    AUC_LOW             = 0.65

    @staticmethod
    def check_track_a(avg_mse: float, per_skill_acc: float,
                      early_stop_rate: float, f1: float) -> None:
        """Track A 结果验收，在 evaluate_track_a 输出日志后调用。"""
        issues = 0

        # MSE
        if avg_mse > EvalMonitor.MSE_HIGH:
            logger.warning(_fmt_warn(
                f"Track A | 状态预测 MSE {avg_mse:.4f} > {EvalMonitor.MSE_HIGH}："
                "编码器未能准确估计学生知识状态。"
                "建议：① 增加 RL 训练轮数；② 增大 RL_E_STEP_NEG_SAMPLES；"
                "③ 检查 NCDM 教师模型是否已充分训练。"
            ))
            issues += 1

        # 逐知识点准确率
        if per_skill_acc < EvalMonitor.SKILL_ACC_RANDOM:
            logger.warning(_fmt_warn(
                f"Track A | 逐知识点准确率 {per_skill_acc*100:.1f}%"
                f" < {EvalMonitor.SKILL_ACC_RANDOM*100:.0f}%（接近随机猜测）。"
                "编码器的掌握概率输出无辨别力，E-step 损失可能未有效收敛。"
                "建议检查训练日志中 E-step 损失趋势。"
            ))
            issues += 1

        # 早停率
        if early_stop_rate < EvalMonitor.ESR_LOW:
            logger.warning(_fmt_warn(
                f"Track A | 早停率 {early_stop_rate*100:.2f}%"
                f" < {EvalMonitor.ESR_LOW*100:.0f}%，智能体几乎从未实现高效早停。"
                "建议：① 确认 CDCAT_TAU 使用平均熵（非最大熵）；"
                "② 适当降低 CDCAT_TAU；③ 增加 RL 训练轮数。"
            ))
            issues += 1

        # F1
        if f1 < EvalMonitor.F1_LOW:
            logger.warning(_fmt_warn(
                f"Track A | 掌握判断 F1 {f1:.4f} < {EvalMonitor.F1_LOW}："
                "二值化后的掌握分类效果较差。"
                "建议增加训练轮数或优化编码器学习率。"
            ))
            issues += 1

        # 汇总
        if issues == 0:
            logger.info(_fmt_ok(
                f"Track A | 所有指标正常——"
                f"MSE={avg_mse:.4f}, SkillAcc={per_skill_acc*100:.1f}%,"
                f" ESR={early_stop_rate*100:.1f}%, F1={f1:.4f}"
            ))
        else:
            logger.warning(_fmt_warn(
                f"Track A | 共发现 {issues} 项异常，详见上方预警。"
            ))

    @staticmethod
    def check_track_b(results: dict) -> None:
        """
        Track B 三策略对比结果验收。

        Parameters
        ----------
        results : dict
            格式: {strategy_name: {auc, rmse, acc, f1, avg_steps, ...}}
        """
        issues = 0

        auc_rl     = results.get("rl",          {}).get("auc")
        auc_random = results.get("random",       {}).get("auc")
        auc_full   = results.get("full_static",  {}).get("auc")
        steps_rl   = results.get("rl",          {}).get("avg_steps")
        steps_rnd  = results.get("random",       {}).get("avg_steps")

        if auc_rl is None:
            logger.error(_fmt_error("Track B | 未找到 RL 策略评估结果，无法执行健康检查。"))
            return

        # ── RL vs Random AUC 差距 ─────────────────────────────────
        if auc_random is not None:
            delta = auc_rl - auc_random
            if abs(delta) < EvalMonitor.RL_RANDOM_MIN_DELTA:
                logger.warning(_fmt_warn(
                    f"Track B | RL AUC ({auc_rl:.4f}) 与随机策略 ({auc_random:.4f}) "
                    f"差距 |Δ|={abs(delta):.4f} < {EvalMonitor.RL_RANDOM_MIN_DELTA}，"
                    "D3QN 未能超越随机选题——训练效果不显著。"
                    "建议重新训练（确认 CDCAT_BETA / CDCAT_TAU 已修正）。"
                ))
                issues += 1
            elif delta < 0:
                logger.warning(_fmt_warn(
                    f"Track B | RL AUC ({auc_rl:.4f}) 低于随机策略 ({auc_random:.4f})，"
                    "RL 策略已退化，建议完整重新训练。"
                ))
                issues += 1

        # ── RL vs 全量静态 ────────────────────────────────────────
        if auc_full is not None and auc_rl < auc_full - 0.01:
            logger.warning(_fmt_warn(
                f"Track B | RL AUC ({auc_rl:.4f}) 低于全量静态测试"
                f" ({auc_full:.4f}) 超过 0.01，"
                "自适应选题未能有效利用信息优势。"
            ))
            issues += 1

        # ── 绝对 AUC ──────────────────────────────────────────────
        if auc_rl < EvalMonitor.AUC_LOW:
            logger.warning(_fmt_warn(
                f"Track B | RL AUC {auc_rl:.4f} < {EvalMonitor.AUC_LOW}，"
                "整体预测质量不足——可能 NCDM 训练不充分或数据质量问题。"
            ))
            issues += 1

        # ── 平均题数：RL 与 Random 完全相同 → 从未自适应早停 ────────
        if steps_rl is not None and steps_rnd is not None:
            if abs(steps_rl - steps_rnd) < 0.5:
                logger.warning(_fmt_warn(
                    f"Track B | RL ({steps_rl:.1f} 题) 与 Random ({steps_rnd:.1f} 题)"
                    " 平均测试长度完全相同，RL 未实现自适应早停。"
                    "建议确认 CDCAT_TAU 阈值合理，并检查 RL 模型是否已充分收敛。"
                ))
                issues += 1

        # ── 汇总 ─────────────────────────────────────────────────
        if issues == 0:
            logger.info(_fmt_ok(
                f"Track B | RL 策略表现正常——"
                f"AUC(RL)={auc_rl:.4f},"
                f" Δ(RL-Random)={auc_rl - (auc_random or 0):+.4f},"
                f" 平均题数={steps_rl:.1f}"
            ))
        else:
            logger.warning(_fmt_warn(
                f"Track B | 共发现 {issues} 项异常，详见上方预警。"
            ))
