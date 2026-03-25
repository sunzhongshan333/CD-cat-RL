import logging
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from collections import deque
from tqdm import tqdm  # 新增进度条包

# 导入我们的模块
from models.ncdm import NCDM
from models.encoder import StateEncoder
from agent.d3qn import D3QN
from agent.replay_buffer import PrioritizedReplayBuffer
from env.cdcat_env import CDCATEnv
import config as cfg

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def _check_plateau(loss_window, threshold):
    """
    检测损失窗口是否进入平台期。
    将窗口等分为前后两段，若后段均值相比前段均值的改善量低于 threshold，
    则认为当前阶段已停止有效优化。
    """
    if len(loss_window) < loss_window.maxlen:
        return False
    losses = list(loss_window)
    half = len(losses) // 2
    return (np.mean(losses[:half]) - np.mean(losses[half:])) < threshold


def train_rl_pipeline():
    # ==========================================
    # 1. 基础配置与路径设置
    # ==========================================
    # 全局随机种子，确保实验可复现
    seed = cfg.RANDOM_SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("当前使用的计算设备是: %s", device)

    data_dir = cfg.DATA_DIR
    models_dir = cfg.MODELS_DIR
    mastery_probs_path = os.path.join(data_dir, 'train_student_mastery_probs.npy')

    # 超参数设置（来自 config.py）
    max_steps = cfg.RL_MAX_STEPS
    batch_size = cfg.RL_BATCH_SIZE
    gamma = cfg.RL_GAMMA
    lr_encoder = cfg.RL_LR_ENCODER
    lr_d3qn = cfg.RL_LR_D3QN
    buffer_capacity = cfg.RL_BUFFER_CAPACITY
    polyak_tau = cfg.RL_POLYAK_TAU
    max_episodes = cfg.RL_MAX_EPISODES
    epsilon_start = cfg.RL_EPSILON_START
    epsilon_end = cfg.RL_EPSILON_END
    epsilon_decay = cfg.RL_EPSILON_DECAY
    grad_clip = cfg.RL_GRAD_CLIP

    # ==========================================
    # 2. 加载冻结的 NCDM 教师模型
    # ==========================================
    q_matrix = np.load(os.path.join(data_dir, 'q_matrix.npy'))
    num_items, num_skills = q_matrix.shape
    q_matrix_tensor = torch.tensor(q_matrix, dtype=torch.float32).to(device)

    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    valid_df = pd.read_csv(os.path.join(data_dir, 'valid.csv'))
    all_users = set(train_df['user_id'].values) | set(valid_df['user_id'].values)
    num_students = max(all_users) + 1

    ncdm = NCDM(num_students, num_items, num_skills).to(device)
    ncdm_ckpt = os.path.join(models_dir, 'ncdm_best.pth')
    try:
        ncdm.load_state_dict(torch.load(ncdm_ckpt, map_location=device, weights_only=True))
    except FileNotFoundError:
        logger.error("找不到 NCDM 权重文件: %s", ncdm_ckpt)
        raise
    except Exception as e:
        logger.error("加载 NCDM 权重失败: %s", e)
        raise
    ncdm.eval()  # 严格冻结教师模型
    for param in ncdm.parameters():
        param.requires_grad = False

    # 获取全部题目的冻结特征供 Encoder 使用
    all_items_idx = torch.arange(num_items).to(device)
    frozen_e_d, frozen_e_a = ncdm.get_frozen_item_features(all_items_idx)

    # ==========================================
    # 3. 初始化核心组件
    # ==========================================
    # 1. 先实例化置换不变编码器 (修复 AttributeError)
    encoder = StateEncoder(q_matrix_tensor, frozen_e_d, frozen_e_a, max_steps=max_steps).to(device)

    # 2. 再把完整的编码器传给环境
    env = CDCATEnv(ncdm, encoder_model=encoder, q_matrix=q_matrix,
                   mastery_probs_path=mastery_probs_path, max_steps=max_steps, device=device)

    # D3QN 策略网络 (主网络和目标网络)
    state_dim = encoder.state_dim
    action_dim = num_items
    main_d3qn = D3QN(state_dim, action_dim).to(device)
    target_d3qn = D3QN(state_dim, action_dim).to(device)
    target_d3qn.load_state_dict(main_d3qn.state_dict())  # 初始参数同步
    target_d3qn.eval()

    # 优化器
    opt_encoder = optim.Adam(encoder.parameters(), lr=lr_encoder)
    opt_d3qn = optim.Adam(main_d3qn.parameters(), lr=lr_d3qn)
    bce_loss = nn.BCEWithLogitsLoss()  # 内置 log-sum-exp trick，数值更稳定

    # 余弦退火学习率调度：从 lr 衰减到接近 0，兼顾前期快速收敛与后期精细调整
    scheduler_encoder = optim.lr_scheduler.CosineAnnealingLR(
        opt_encoder, T_max=max_episodes, eta_min=lr_encoder * 0.01)
    scheduler_d3qn = optim.lr_scheduler.CosineAnnealingLR(
        opt_d3qn, T_max=max_episodes, eta_min=lr_d3qn * 0.01)

    # 回放池（优先经验回放）
    buffer = PrioritizedReplayBuffer(
        buffer_capacity, max_steps, device,
        alpha=cfg.RL_PER_ALPHA,
        beta_start=cfg.RL_PER_BETA_START,
        beta_frames=cfg.RL_PER_BETA_FRAMES,
        eps=cfg.RL_PER_EPS,
    )

    # ==========================================
    # 4. 主训练循环 (自适应交替优化范式)
    # ==========================================
    total_steps = 0
    logger.info("开始强化学习范式训练...")

    # --- 自适应交替优化状态 ---
    current_phase = 'e_step'         # 从 E-step 开始
    phase_opt_steps = 0              # 当前阶段已完成的优化步数
    e_step_losses = deque(maxlen=cfg.RL_LOSS_PLATEAU_WINDOW)
    q_step_losses = deque(maxlen=cfg.RL_LOSS_PLATEAU_WINDOW)

    # 使用 tqdm 包裹训练循环，创建可视化进度条
    pbar = tqdm(range(1, max_episodes + 1), desc="RL Training")

    for episode in pbar:
        # 本 episode 所属阶段（上一步优化后可能已切换，这里生效）
        is_e_step = (current_phase == 'e_step')
        did_update = False  # 本 episode 是否执行了至少一次参数更新（用于守护 scheduler）

        if is_e_step:
            encoder.train()
            main_d3qn.eval()  # 冻结 D3QN
            phase_str = "E-Step"
        else:
            encoder.eval()  # 冻结 Encoder
            main_d3qn.train()
            phase_str = "Q-Step"

        # 环境复位
        s_t = env.reset()
        done = False

        # 计算当前探索率
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
                  np.exp(-1. * episode / epsilon_decay)

        while not done:
            mask_t = env.get_action_mask()

            # --- 动作选择（信息引导探索代替纯随机 ε-greedy）---
            if np.random.rand() < epsilon:
                # 信息引导探索：优先选覆盖当前高熵知识点的题目
                with torch.no_grad():
                    hat_alpha = env.current_hat_alpha          # [K]
                    p = torch.clamp(hat_alpha, 1e-7, 1.0 - 1e-7)
                    skill_entropy = -p * torch.log(p) - (1 - p) * torch.log(1 - p)  # [K]
                    valid_indices = torch.where(mask_t)[0]              # [num_valid]
                    valid_q = env.q_matrix[valid_indices]               # [num_valid, K]
                    item_scores = (valid_q * skill_entropy.unsqueeze(0)).sum(dim=1)  # [num_valid]
                    action = valid_indices[item_scores.argmax().item()].item()
            else:
                # D3QN 贪心选择
                with torch.no_grad():
                    q_values = main_d3qn(s_t.unsqueeze(0), mask_t.unsqueeze(0))
                    action = q_values.argmax(dim=1).item()

            # --- 执行动作，获取转移 ---
            h_items_t = list(env.history_item_ids)
            h_scores_t = list(env.history_scores)
            step_t = env.current_step
            true_alpha_t = env.alpha_star.clone()

            s_next, reward, done, info = env.step(action)
            mask_next = env.get_action_mask()
            total_steps += 1

            # --- 存入优先经验回放池 ---
            buffer.push(h_items_t, h_scores_t, step_t, action, reward,
                        env.history_item_ids, env.history_scores, env.current_step,
                        mask_t, mask_next, done, true_alpha_t)

            # --- Polyak 软更新目标网络（仅 Q-step 时执行，E-step 时 D3QN 未更新故跳过）---
            if not is_e_step:
                with torch.no_grad():
                    for param, target_param in zip(main_d3qn.parameters(), target_d3qn.parameters()):
                        target_param.lerp_(param, polyak_tau)

            s_t = s_next

            # ==========================================
            # 5. 核心优化步骤 (Batch 训练)
            # ==========================================
            if buffer.is_ready(batch_size):
                # 采样 Batch（含 PER 下标和 IS 权重）
                b_h_items, b_h_scores, b_steps, b_actions, b_rewards, \
                    b_next_h_items, b_next_h_scores, b_next_steps, \
                    b_masks, b_next_masks, b_dones, b_true_alphas, \
                    b_indices, b_is_weights = buffer.sample(batch_size)

                if is_e_step:
                    # ------------------------------------------
                    # E-step: 表征学习 (优化 Encoder)
                    # ------------------------------------------
                    opt_encoder.zero_grad()

                    # 1. 实时生成当前状态
                    s_batch, mastery_logits = encoder(b_h_items, b_h_scores, b_steps)
                    hat_alpha = torch.sigmoid(mastery_logits)  # [batch, K]

                    # 2a. 未答题目上的软标签损失（负采样，避免 [B, J, K] 超大张量）
                    # 从全量题库随机抽取 N_neg 道题，tensor 从 [B,J,K] 降至 [B,N_neg,K]
                    neg_idx = torch.randperm(num_items, device=device)[:cfg.RL_E_STEP_NEG_SAMPLES]
                    # [B, N_neg, K]
                    interaction_pred_neg = (hat_alpha.unsqueeze(1)
                                            * q_matrix_tensor[neg_idx].unsqueeze(0)
                                            * frozen_e_a[neg_idx].unsqueeze(0)
                                            - frozen_e_d[neg_idx].unsqueeze(0))
                    pred_logits_neg = ncdm.interaction_mlp(
                        interaction_pred_neg).squeeze(-1)  # [B, N_neg]

                    with torch.no_grad():
                        interaction_teacher_neg = (b_true_alphas.unsqueeze(1)
                                                   * q_matrix_tensor[neg_idx].unsqueeze(0)
                                                   * frozen_e_a[neg_idx].unsqueeze(0)
                                                   - frozen_e_d[neg_idx].unsqueeze(0))
                        target_y_neg = torch.sigmoid(
                            ncdm.interaction_mlp(interaction_teacher_neg).squeeze(-1))  # [B, N_neg]

                    # 仅对采样子集中未被作答的题目计算软标签损失
                    neg_unanswered = b_masks[:, neg_idx]  # [B, N_neg]，True=可选（未答）
                    loss_soft = bce_loss(pred_logits_neg[neg_unanswered],
                                        target_y_neg[neg_unanswered])

                    # 2b. 已答题目上的真实标签损失（自洽性约束，精确按索引计算）
                    valid_h_mask = (b_h_items != -1)                            # [B, T]
                    if valid_h_mask.any():
                        b_row = torch.where(valid_h_mask)[0]                    # batch 行索引 [N_ans]
                        i_col = b_h_items[valid_h_mask]                         # 题目索引   [N_ans]
                        # [N_ans, K]
                        interaction_ans = (hat_alpha[b_row]
                                           * q_matrix_tensor[i_col]
                                           * frozen_e_a[i_col]
                                           - frozen_e_d[i_col])
                        pred_logits_ans = ncdm.interaction_mlp(
                            interaction_ans).squeeze(-1)                        # [N_ans]
                        real_labels_ans = b_h_scores[valid_h_mask].clamp(min=0).float()  # [N_ans]
                        loss_real = bce_loss(pred_logits_ans, real_labels_ans)
                        loss_aux = loss_soft + loss_real
                    else:
                        loss_aux = loss_soft

                    loss_aux.backward()
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), grad_clip)
                    opt_encoder.step()
                    did_update = True

                    # 自适应交替：追踪 E-step 损失
                    e_step_losses.append(loss_aux.item())
                    phase_opt_steps += 1
                    if (phase_opt_steps >= cfg.RL_LOSS_PLATEAU_MIN_STEPS and
                            _check_plateau(e_step_losses, cfg.RL_LOSS_PLATEAU_THRESHOLD)):
                        current_phase = 'q_step'
                        phase_opt_steps = 0
                        e_step_losses.clear()

                else:
                    # ------------------------------------------
                    # Q-step: 策略学习 (优化 D3QN)
                    # ------------------------------------------
                    opt_d3qn.zero_grad()

                    with torch.no_grad():
                        all_h_items  = torch.cat([b_h_items,       b_next_h_items],  dim=0)
                        all_h_scores = torch.cat([b_h_scores,      b_next_h_scores], dim=0)
                        all_steps    = torch.cat([b_steps,         b_next_steps],    dim=0)
                        all_states, _ = encoder(all_h_items, all_h_scores, all_steps)
                        s_batch      = all_states[:batch_size]
                        s_next_batch = all_states[batch_size:]

                        # Double DQN 核心逻辑
                        argmax_a = main_d3qn(s_next_batch, b_next_masks).argmax(dim=1, keepdim=True)
                        max_q_next = target_d3qn(s_next_batch, b_next_masks).gather(1, argmax_a)
                        target_q = b_rewards + gamma * max_q_next * (1 - b_dones)

                    # 计算当前评估的 Q 值
                    q_eval = main_d3qn(s_batch, b_masks).gather(1, b_actions)

                    # IS 权重加权 TD 损失（PER 无偏修正）
                    td_errors_sq = (q_eval - target_q) ** 2        # [batch, 1]
                    loss_td = (b_is_weights * td_errors_sq).mean()
                    loss_td.backward()
                    torch.nn.utils.clip_grad_norm_(main_d3qn.parameters(), grad_clip)
                    opt_d3qn.step()
                    did_update = True
                    with torch.no_grad():
                        td_errors_np = (q_eval - target_q).abs().squeeze(1).cpu().numpy()
                    buffer.update_priorities(b_indices, td_errors_np)

                    # 自适应交替：追踪 Q-step 损失
                    q_step_losses.append(loss_td.item())
                    phase_opt_steps += 1
                    if (phase_opt_steps >= cfg.RL_LOSS_PLATEAU_MIN_STEPS and
                            _check_plateau(q_step_losses, cfg.RL_LOSS_PLATEAU_THRESHOLD)):
                        current_phase = 'e_step'
                        phase_opt_steps = 0
                        q_step_losses.clear()

        # ==========================================
        # 6. 周期性日志打印与模型保存
        # ==========================================
        # 每个 episode 结束后推进学习率调度；仅在本 episode 有参数更新时才推进，
        # 避免缓冲区未满的冷启动阶段触发 "scheduler before optimizer" 警告
        if did_update:
            scheduler_encoder.step()
            scheduler_d3qn.step()

        # 实时将当前状态更新到进度条尾部，方便监控
        pbar.set_postfix({'Phase': phase_str, 'Steps': env.current_step, 'Epsilon': f"{epsilon:.3f}"})

        # 每 1000 轮保存一次 Checkpoint
        if episode % 1000 == 0:
            os.makedirs(models_dir, exist_ok=True)
            torch.save(encoder.state_dict(), os.path.join(models_dir, f'encoder_ep{episode}.pth'))
            torch.save(main_d3qn.state_dict(), os.path.join(models_dir, f'd3qn_ep{episode}.pth'))
            tqdm.write(f"--> [Checkpoint] 成功保存 Episode {episode} 的模型权重！")

    logger.info("强化学习范式训练圆满结束！")

    # 无条件保存最终模型，确保 max_episodes 不能整除 1000 时也不会丢失权重
    if max_episodes % 1000 != 0:
        os.makedirs(models_dir, exist_ok=True)
        torch.save(encoder.state_dict(), os.path.join(models_dir, f'encoder_ep{max_episodes}.pth'))
        torch.save(main_d3qn.state_dict(), os.path.join(models_dir, f'd3qn_ep{max_episodes}.pth'))
        logger.info("最终权重已保存（encoder_ep%d.pth / d3qn_ep%d.pth）。", max_episodes, max_episodes)


if __name__ == "__main__":
    train_rl_pipeline()