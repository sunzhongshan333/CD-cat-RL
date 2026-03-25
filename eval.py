import logging
import os
import random
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, mean_squared_error,
    accuracy_score, f1_score, precision_score, recall_score, log_loss,
)
from sklearn.model_selection import train_test_split

from models.ncdm import NCDM
from models.encoder import StateEncoder
from agent.d3qn import D3QN
from env.cdcat_env import CDCATEnv

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def load_models(device, data_dir, models_dir, max_steps, checkpoint_ep=5000):
    """统一加载所有预训练好的权重，并设置为严格的 eval 模式"""
    q_matrix = np.load(os.path.join(data_dir, 'q_matrix.npy'))
    num_items, num_skills = q_matrix.shape
    q_matrix_tensor = torch.tensor(q_matrix, dtype=torch.float32).to(device)
    mastery_probs_path = os.path.join(data_dir, 'train_student_mastery_probs.npy')

    # 从原始训练/验证集 CSV 中推断最大 user_id，与 NCDM 训练时保持一致
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    valid_df = pd.read_csv(os.path.join(data_dir, 'valid.csv'))
    num_students = int(max(train_df['user_id'].max(), valid_df['user_id'].max())) + 1

    # 1. 加载 NCDM
    ncdm = NCDM(num_students, num_items, num_skills).to(device)
    ncdm_ckpt = os.path.join(models_dir, 'ncdm_best.pth')
    try:
        ncdm.load_state_dict(torch.load(ncdm_ckpt, map_location=device))
    except FileNotFoundError:
        logger.error("找不到 NCDM 权重文件: %s", ncdm_ckpt)
        raise
    except Exception as e:
        logger.error("加载 NCDM 权重失败: %s", e)
        raise
    ncdm.eval()

    # 获取冻结特征
    all_items_idx = torch.arange(num_items).to(device)
    frozen_e_d, frozen_e_a = ncdm.get_frozen_item_features(all_items_idx)

    # 2. 加载 Encoder
    encoder = StateEncoder(q_matrix_tensor, frozen_e_d, frozen_e_a, max_steps=max_steps).to(device)
    encoder_ckpt = os.path.join(models_dir, f'encoder_ep{checkpoint_ep}.pth')
    try:
        encoder.load_state_dict(torch.load(encoder_ckpt, map_location=device))
    except FileNotFoundError:
        logger.error("找不到 Encoder 权重文件: %s", encoder_ckpt)
        raise
    except Exception as e:
        logger.error("加载 Encoder 权重失败: %s", e)
        raise
    encoder.eval()

    # 3. 加载 D3QN
    state_dim = encoder.state_dim
    d3qn = D3QN(state_dim, action_dim=num_items).to(device)
    d3qn_ckpt = os.path.join(models_dir, f'd3qn_ep{checkpoint_ep}.pth')
    try:
        d3qn.load_state_dict(torch.load(d3qn_ckpt, map_location=device))
    except FileNotFoundError:
        logger.error("找不到 D3QN 权重文件: %s", d3qn_ckpt)
        raise
    except Exception as e:
        logger.error("加载 D3QN 权重失败: %s", e)
        raise
    d3qn.eval()

    return ncdm, encoder, d3qn, q_matrix_tensor, mastery_probs_path


def evaluate_track_a(env, encoder, d3qn, num_simulated_students=500, device='cpu',
                     curve_save_path=None):
    """
    Track A: 模拟数据轨 (真实知识状态 α* 已知)

    评估指标:
      - 平均测试长度 / 早停率 (entropy < τ 触发)
      - MSE / MAE (知识状态诊断误差)
      - PAR (模式准确率，全部知识点完全匹配)
      - 逐知识点准确率 (Per-Skill Accuracy)
      - Precision / Recall / F1 (二值掌握判断)

    可选: 将 MSE@K 效率曲线保存到 curve_save_path (CSV 格式)
    """
    logger.info("\n" + "=" * 50)
    logger.info("开始 Track A (模拟数据) 评估，共 %d 名虚拟学生...", num_simulated_students)

    total_steps = 0
    early_stop_count = 0   # 因熵达标提前终止的学生数

    per_student_mse = []
    per_student_mae = []
    par_hits = 0

    # 用于全局 Precision / Recall / F1 / Per-Skill Accuracy
    all_true_flat = []   # 所有学生所有知识点的真实二值标签
    all_pred_flat = []   # 所有学生所有知识点的预测二值标签

    # MSE@K 效率曲线: 第 k 步时各学生 MSE 的累计量
    max_steps_int = env.max_steps
    step_mse_sums = np.zeros(max_steps_int + 1)
    step_mse_counts = np.zeros(max_steps_int + 1, dtype=int)

    for i in range(num_simulated_students):
        s_t = env.reset()
        done = False
        # alpha_star 在 reset() 后立即固定，整个 episode 不变
        true_alpha = env.alpha_star.cpu().numpy()

        # 记录每一步结束后 encoder 对知识状态的预测 (供 MSE@K 使用)
        step_preds = {}

        while not done:
            mask_t = env.get_action_mask()
            with torch.no_grad():
                q_values = d3qn(s_t.unsqueeze(0), mask_t.unsqueeze(0))
                action = q_values.argmax(dim=1).item()

            s_t, reward, done, info = env.step(action)
            # env.current_step 已在 step() 内自增，记录本步预测
            step_preds[env.current_step] = info['pred_alpha']

        final_step = env.current_step
        total_steps += final_step

        # 终止原因判断：熵达标 vs 到达最大步数
        if info['max_entropy'] < env.tau:
            early_stop_count += 1

        pred_alpha = info['pred_alpha']

        # MSE / MAE
        per_student_mse.append(float(np.mean((true_alpha - pred_alpha) ** 2)))
        per_student_mae.append(float(np.mean(np.abs(true_alpha - pred_alpha))))

        # PAR (所有知识点完全吻合才算 1 分)
        pred_binary = (pred_alpha > 0.5).astype(np.float32)
        if np.array_equal(true_alpha.astype(np.float32), pred_binary):
            par_hits += 1

        # 累计展平标签，供全局二分类指标计算
        all_true_flat.extend(true_alpha.astype(int).tolist())
        all_pred_flat.extend(pred_binary.astype(int).tolist())

        # MSE@K 曲线累计
        for k, p_alpha in step_preds.items():
            if 1 <= k <= max_steps_int:
                step_mse_sums[k] += float(np.mean((true_alpha - p_alpha) ** 2))
                step_mse_counts[k] += 1

    # ── 汇总指标 ────────────────────────────────────────────────────
    avg_steps = total_steps / num_simulated_students
    early_stop_rate = early_stop_count / num_simulated_students
    avg_mse = float(np.mean(per_student_mse))
    avg_mae = float(np.mean(per_student_mae))
    par = par_hits / num_simulated_students

    all_true_arr = np.array(all_true_flat)
    all_pred_arr = np.array(all_pred_flat)
    per_skill_acc = accuracy_score(all_true_arr, all_pred_arr)
    prec = precision_score(all_true_arr, all_pred_arr, zero_division=0)
    rec  = recall_score(all_true_arr, all_pred_arr, zero_division=0)
    f1   = f1_score(all_true_arr, all_pred_arr, zero_division=0)

    logger.info("[Track A 结果] 平均测试长度:       %.2f 题", avg_steps)
    logger.info("[Track A 结果] 早停率:             %.2f%% (%d/%d 提前终止)",
                early_stop_rate * 100, early_stop_count, num_simulated_students)
    logger.info("[Track A 结果] 状态预测 MSE:       %.4f", avg_mse)
    logger.info("[Track A 结果] 状态预测 MAE:       %.4f", avg_mae)
    logger.info("[Track A 结果] 模式准确率 (PAR):   %.2f%%", par * 100)
    logger.info("[Track A 结果] 逐知识点准确率:     %.2f%%", per_skill_acc * 100)
    logger.info("[Track A 结果] 掌握判断 Precision: %.4f", prec)
    logger.info("[Track A 结果] 掌握判断 Recall:    %.4f", rec)
    logger.info("[Track A 结果] 掌握判断 F1:        %.4f", f1)

    # ── MSE@K 效率曲线 (可选保存) ────────────────────────────────────
    if curve_save_path is not None:
        valid_ks = np.where(step_mse_counts > 0)[0]
        if len(valid_ks) > 0:
            avg_mse_at_k = np.where(
                step_mse_counts > 0,
                step_mse_sums / np.maximum(step_mse_counts, 1),
                np.nan,
            )
            curve_df = pd.DataFrame({
                'step': valid_ks,
                'avg_mse': avg_mse_at_k[valid_ks],
            })
            curve_df.to_csv(curve_save_path, index=False)
            logger.info("[Track A 效率曲线] MSE@K 已保存至: %s", curve_save_path)


def _predict_holdout(ncdm, hat_alpha, holdout_items_tensor, q_matrix_tensor):
    """
    使用 NCDM 前向计算保留集题目的作答概率。

    Args:
        hat_alpha: [1, K] 当前诊断的知识掌握概率向量
        holdout_items_tensor: [n_holdout] 保留集题目 ID
        q_matrix_tensor: [J, K] Q 矩阵

    Returns:
        pred_probs: (n_holdout,) numpy 数组，预测答对概率
    """
    with torch.no_grad():
        e_d, e_a = ncdm.get_frozen_item_features(holdout_items_tensor)
        q_vec = q_matrix_tensor[holdout_items_tensor]
        interaction = hat_alpha * q_vec * e_a - e_d
        # NCDM interaction_mlp 输出原始 logit，加 sigmoid 转为概率
        pred_probs = torch.sigmoid(ncdm.interaction_mlp(interaction)).squeeze(-1).cpu().numpy()
    return pred_probs


def _run_track_b_policy(policy, ncdm, encoder, d3qn, grouped,
                        q_matrix_tensor, max_steps, tau, device):
    """
    在 Track B 真实数据上运行单一选题策略，返回汇总指标和 AUC@K 曲线。

    policy:
        'rl'          — D3QN 自适应选题（含熵达标提前终止）
        'random'      — 随机选题（等长，同样有熵达标提前终止）
        'full_static' — 遍历全部可用池题目（无自适应，最多 max_steps 题）

    Returns:
        dict 包含: valid_students, avg_steps, auc, rmse, acc, f1, log_loss,
                   auc_curve ({step_k: auc_value})
    """
    all_holdout_y_true = []
    all_holdout_y_pred = []
    total_steps = 0
    valid_students = 0

    # AUC@K 曲线: step k -> {'y_true': [...], 'y_pred': [...]}
    step_auc_data = {}

    for user_id, user_data in grouped:
        if len(user_data) < 10:
            continue  # 样本过少无法有效划分 70/30

        valid_students += 1
        items = user_data['problem_id'].values
        scores = user_data['correct'].values

        pool_items, holdout_items, pool_scores, holdout_scores = train_test_split(
            items, scores, test_size=0.3, random_state=42)

        pool_dict = dict(zip(pool_items.tolist(), pool_scores.tolist()))
        holdout_items_tensor = torch.tensor(holdout_items, dtype=torch.long).to(device)

        current_step = 0
        pad_items  = torch.full((1, max_steps), -1, dtype=torch.long).to(device)
        pad_scores = torch.full((1, max_steps), -1, dtype=torch.long).to(device)
        step_tensor = torch.tensor([0], dtype=torch.long).to(device)

        if policy == 'full_static':
            # 按原始顺序遍历可用池（最多 max_steps 题）
            for action, real_score in zip(pool_items[:max_steps],
                                          pool_scores[:max_steps]):
                current_step += 1
                pad_items[0,  current_step - 1] = action
                pad_scores[0, current_step - 1] = real_score
                step_tensor = torch.tensor([current_step], dtype=torch.long).to(device)

                # AUC@K 快照
                with torch.no_grad():
                    _, mastery_logits_k = encoder(pad_items, pad_scores, step_tensor)
                    hat_alpha_k = torch.sigmoid(mastery_logits_k)   # [1, K]
                pred_k = _predict_holdout(ncdm, hat_alpha_k,
                                          holdout_items_tensor, q_matrix_tensor)
                entry = step_auc_data.setdefault(current_step, {'y_true': [], 'y_pred': []})
                entry['y_true'].extend(holdout_scores.tolist())
                entry['y_pred'].extend(pred_k.tolist())

        else:  # 'rl' 或 'random'
            while current_step < max_steps and len(pool_dict) > 0:
                with torch.no_grad():
                    s_t, mastery_logits = encoder(pad_items, pad_scores, step_tensor)
                    hat_alpha = torch.sigmoid(mastery_logits).squeeze(0)   # [K]

                # 熵达标提前终止
                p = torch.clamp(hat_alpha, 1e-7, 1.0 - 1e-7)
                entropy = -p * torch.log(p) - (1 - p) * torch.log(1 - p)
                if torch.max(entropy).item() < tau:
                    break

                available_pool_items = list(pool_dict.keys())
                mask_t = torch.zeros(ncdm.num_items, dtype=torch.bool).to(device)
                mask_t[available_pool_items] = True

                if policy == 'rl':
                    with torch.no_grad():
                        q_values = d3qn(s_t, mask_t.unsqueeze(0))
                        action = q_values.argmax(dim=1).item()
                else:  # 'random'
                    action = random.choice(available_pool_items)

                real_score = pool_dict.pop(action)
                current_step += 1
                pad_items[0,  current_step - 1] = action
                pad_scores[0, current_step - 1] = real_score
                step_tensor = torch.tensor([current_step], dtype=torch.long).to(device)

                # AUC@K 快照
                with torch.no_grad():
                    _, mastery_logits_k = encoder(pad_items, pad_scores, step_tensor)
                    hat_alpha_k = torch.sigmoid(mastery_logits_k)   # [1, K]
                pred_k = _predict_holdout(ncdm, hat_alpha_k,
                                          holdout_items_tensor, q_matrix_tensor)
                entry = step_auc_data.setdefault(current_step, {'y_true': [], 'y_pred': []})
                entry['y_true'].extend(holdout_scores.tolist())
                entry['y_pred'].extend(pred_k.tolist())

        total_steps += current_step

        # 最终诊断状态 → 保留集预测
        with torch.no_grad():
            _, mastery_logits = encoder(pad_items, pad_scores, step_tensor)
            hat_alpha_final = torch.sigmoid(mastery_logits)   # [1, K]
        pred_probs = _predict_holdout(ncdm, hat_alpha_final,
                                      holdout_items_tensor, q_matrix_tensor)
        all_holdout_y_true.extend(holdout_scores.tolist())
        all_holdout_y_pred.extend(pred_probs.tolist())

    if valid_students == 0:
        raise RuntimeError("Track B: 没有满足条件（答题数≥10）的学生，无法评估。")

    y_true_arr  = np.array(all_holdout_y_true)
    y_pred_arr  = np.array(all_holdout_y_pred)
    y_pred_bin  = (y_pred_arr > 0.5).astype(int)

    auc  = roc_auc_score(y_true_arr, y_pred_arr)
    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    acc  = accuracy_score(y_true_arr, y_pred_bin)
    f1   = f1_score(y_true_arr, y_pred_bin, zero_division=0)
    ll   = log_loss(y_true_arr, y_pred_arr)
    avg_steps = total_steps / valid_students

    # 计算各步 AUC（至少包含两类标签才有意义）
    auc_curve = {}
    for k, data in sorted(step_auc_data.items()):
        yt = np.array(data['y_true'])
        yp = np.array(data['y_pred'])
        if len(np.unique(yt)) > 1:
            auc_curve[k] = float(roc_auc_score(yt, yp))

    return {
        'valid_students': valid_students,
        'avg_steps':      avg_steps,
        'auc':            auc,
        'rmse':           rmse,
        'acc':            acc,
        'f1':             f1,
        'log_loss':       ll,
        'auc_curve':      auc_curve,
    }


def evaluate_track_b(ncdm, encoder, d3qn, test_csv_path, q_matrix_tensor,
                     max_steps, tau, device, curve_save_path=None):
    """
    Track B: 真实数据轨

    对比三种策略:
        rl          — D3QN 自适应选题
        random      — 随机选题基线（等长对照）
        full_static — 全量静态测试基线（性能上界参考）

    评估指标: 平均测试长度 / AUC / RMSE / Accuracy / F1 / Log-Loss
    可选: 将 AUC@K 效率曲线保存到 curve_save_path (CSV 格式)
    """
    logger.info("\n" + "=" * 50)
    logger.info("开始 Track B (真实数据) 评估，共三种策略对比...")

    test_df = pd.read_csv(test_csv_path)
    grouped = list(test_df.groupby('user_id'))

    results = {}
    for policy in ('rl', 'random', 'full_static'):
        logger.info("  运行策略: %s ...", policy)
        results[policy] = _run_track_b_policy(
            policy, ncdm, encoder, d3qn, grouped,
            q_matrix_tensor, max_steps, tau, device,
        )

    # ── 打印三策略对比表格 ────────────────────────────────────────────
    col_w = 13
    header = (f"{'策略':<{col_w}} {'参与学生':>8} {'平均题数':>8} "
              f"{'AUC':>8} {'RMSE':>8} {'Acc':>8} {'F1':>8} {'LogLoss':>9}")
    sep = "-" * 75
    logger.info("\n" + "=" * 75)
    logger.info("[Track B 对比结果]")
    logger.info(header)
    logger.info(sep)
    for policy, r in results.items():
        row = (f"{policy:<{col_w}} {r['valid_students']:>8d} {r['avg_steps']:>8.2f} "
               f"{r['auc']:>8.4f} {r['rmse']:>8.4f} {r['acc']:>8.4f} "
               f"{r['f1']:>8.4f} {r['log_loss']:>9.4f}")
        logger.info(row)

    # ── AUC@K 效率曲线 (可选保存) ─────────────────────────────────────
    if curve_save_path is not None:
        all_ks = sorted(set().union(*[set(r['auc_curve'].keys()) for r in results.values()]))
        if all_ks:
            curve_rows = []
            for k in all_ks:
                row_dict = {'step': k}
                for policy, r in results.items():
                    row_dict[f'auc_{policy}'] = r['auc_curve'].get(k, float('nan'))
                curve_rows.append(row_dict)
            curve_df = pd.DataFrame(curve_rows)
            curve_df.to_csv(curve_save_path, index=False)
            logger.info("[Track B 效率曲线] AUC@K 已保存至: %s", curve_save_path)


if __name__ == "__main__":
    from config import DATA_DIR, MODELS_DIR, EVAL_MAX_STEPS, EVAL_TAU, EVAL_NUM_SIMULATED, EVAL_CHECKPOINT_EP

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("评估环境设备: %s", device)

    TEST_CSV = os.path.join(DATA_DIR, 'test.csv')

    # 统一加载权重
    logger.info("正在加载 NCDM、Encoder 和 D3QN 的网络权重...")
    ncdm, encoder, d3qn, q_mat, m_probs_path = load_models(
        device, DATA_DIR, MODELS_DIR, EVAL_MAX_STEPS, checkpoint_ep=EVAL_CHECKPOINT_EP
    )

    # 初始化 Track A 所需的 Env
    env = CDCATEnv(ncdm, encoder, q_mat.cpu().numpy(), m_probs_path,
                   max_steps=EVAL_MAX_STEPS, tau=EVAL_TAU, device=device)

    # 运行双轨评估
    evaluate_track_a(
        env, encoder, d3qn,
        num_simulated_students=EVAL_NUM_SIMULATED,
        device=device,
        curve_save_path=os.path.join(DATA_DIR, 'eval_track_a_mse_curve.csv'),
    )
    evaluate_track_b(
        ncdm, encoder, d3qn, TEST_CSV, q_mat, EVAL_MAX_STEPS, EVAL_TAU,
        device=device,
        curve_save_path=os.path.join(DATA_DIR, 'eval_track_b_auc_curve.csv'),
    )