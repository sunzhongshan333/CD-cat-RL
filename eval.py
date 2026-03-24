import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import train_test_split

from models.ncdm import NCDM
from models.encoder import StateEncoder
from agent.d3qn import D3QN
from env.cdcat_env import CDCATEnv


def load_models(device, data_dir, models_dir, max_steps):
    """统一加载所有预训练好的权重，并设置为严格的 eval 模式"""
    q_matrix = np.load(os.path.join(data_dir, 'q_matrix.npy'))
    num_items, num_skills = q_matrix.shape
    q_matrix_tensor = torch.tensor(q_matrix, dtype=torch.float32).to(device)
    mastery_probs_path = os.path.join(data_dir, 'train_student_mastery_probs.npy')
    num_students = np.load(mastery_probs_path).shape[0]

    # 1. 加载 NCDM
    ncdm = NCDM(num_students, num_items, num_skills).to(device)
    ncdm.load_state_dict(torch.load(os.path.join(models_dir, 'ncdm_best.pth'), map_location=device))
    ncdm.eval()

    # 获取冻结特征
    all_items_idx = torch.arange(num_items).to(device)
    frozen_e_d, frozen_e_a = ncdm.get_frozen_item_features(all_items_idx)

    # 2. 加载 Encoder
    encoder = StateEncoder(q_matrix_tensor, frozen_e_d, frozen_e_a, max_steps=max_steps).to(device)
    # 请根据你实际训练保存的 epoch 数修改文件名，这里假设加载第 5000 轮
    encoder.load_state_dict(torch.load(os.path.join(models_dir, 'encoder_ep5000.pth'), map_location=device))
    encoder.eval()

    # 3. 加载 D3QN
    state_dim = encoder.d2 + num_skills + 1
    d3qn = D3QN(state_dim, action_dim=num_items).to(device)
    d3qn.load_state_dict(torch.load(os.path.join(models_dir, 'd3qn_ep5000.pth'), map_location=device))
    d3qn.eval()

    return ncdm, encoder, d3qn, q_matrix_tensor, mastery_probs_path


def evaluate_track_a(env, encoder, d3qn, num_simulated_students=500, device='cpu'):
    """
    Track A: 模拟数据轨 (真实知识状态 \alpha^* 已知)
    评估指标: 均方误差 (MSE), 模式准确率 (PAR), 平均测试长度
    """
    print("\n" + "=" * 50)
    print(f"开始 Track A (模拟数据) 评估，共 {num_simulated_students} 名虚拟学生...")

    total_steps = 0
    total_mse = 0.0
    par_hits = 0

    for i in range(num_simulated_students):
        s_t = env.reset()
        done = False

        while not done:
            mask_t = env.get_action_mask()
            with torch.no_grad():
                q_values = d3qn(s_t.unsqueeze(0), mask_t.unsqueeze(0))
                action = q_values.argmax(dim=1).item()

            s_t, reward, done, info = env.step(action)

        # 记录结果
        total_steps += env.current_step
        true_alpha = info['true_alpha']
        pred_alpha = info['pred_alpha']

        total_mse += np.mean((true_alpha - pred_alpha) ** 2)

        # 计算模式准确率 PAR (将预测概率二值化后，如果全相等则记为 1)
        pred_binary = (pred_alpha > 0.5).astype(np.float32)
        if np.array_equal(true_alpha, pred_binary):
            par_hits += 1

    avg_steps = total_steps / num_simulated_students
    avg_mse = total_mse / num_simulated_students
    par = par_hits / num_simulated_students

    print(f"[Track A 结果] 平均测试长度: {avg_steps:.2f} 题")
    print(f"[Track A 结果] 状态预测 MSE: {avg_mse:.4f}")
    print(f"[Track A 结果] 模式准确率 (PAR): {par * 100:.2f}%")


def evaluate_track_b(ncdm, encoder, d3qn, test_csv_path, q_matrix_tensor, max_steps, tau, device):
    """
    Track B: 真实数据轨 (利用可用池 70% 选题，在全局保留集 30% 上预测)
    评估指标: AUC, RMSE, 平均测试长度
    """
    print("\n" + "=" * 50)
    print("开始 Track B (真实数据) 评估...")

    test_df = pd.read_csv(test_csv_path)
    grouped = test_df.groupby('user_id')

    all_holdout_y_true = []
    all_holdout_y_pred = []
    total_steps = 0
    valid_students = 0

    for user_id, user_data in grouped:
        if len(user_data) < 10:
            continue  # 跳过答题过少的学生，无法有效划分 70/30

        valid_students += 1
        items = user_data['problem_id'].values
        scores = user_data['correct'].values

        # 将考生的真实作答划分为 70% 可用池 和 30% 保留集
        pool_items, holdout_items, pool_scores, holdout_scores = train_test_split(
            items, scores, test_size=0.3, random_state=42)

        pool_dict = dict(zip(pool_items, pool_scores))

        # ------------------- 真实数据上的 CD-CAT 推断循环 -------------------
        history_items = []
        history_scores = []
        current_step = 0

        # 初始状态
        pad_items = torch.full((1, max_steps), -1, dtype=torch.long).to(device)
        pad_scores = torch.full((1, max_steps), -1, dtype=torch.long).to(device)
        step_tensor = torch.tensor([0], dtype=torch.long).to(device)

        while current_step < max_steps and len(pool_dict) > 0:
            # 1. 实时编码状态
            with torch.no_grad():
                s_t, mastery_logits = encoder(pad_items, pad_scores, step_tensor)
                hat_alpha = torch.sigmoid(mastery_logits).squeeze(0)  # [K]

            # 计算熵，判断是否达标终止
            p = torch.clamp(hat_alpha, 1e-7, 1.0 - 1e-7)
            entropy = -p * torch.log(p) - (1 - p) * torch.log(1 - p)
            if torch.max(entropy).item() < tau:
                break  # 熵达标，提前终止

            # 2. 获取掩码 (只能选 pool_dict 里还没做过的题)
            mask_t = torch.zeros(ncdm.num_items, dtype=torch.bool).to(device)
            available_pool_items = list(pool_dict.keys())
            mask_t[available_pool_items] = True

            # 3. D3QN 动作选择
            with torch.no_grad():
                q_values = d3qn(s_t, mask_t.unsqueeze(0))
                action = q_values.argmax(dim=1).item()

            # 4. 执行动作，获取真实作答
            real_score = pool_dict.pop(action)
            history_items.append(action)
            history_scores.append(real_score)
            current_step += 1

            # 更新 padded history 供下一步使用
            pad_items[0, current_step - 1] = action
            pad_scores[0, current_step - 1] = real_score
            step_tensor = torch.tensor([current_step], dtype=torch.long).to(device)

        # 循环结束，记录步数
        total_steps += current_step

        # 获取最终诊断状态 hat_alpha_final
        with torch.no_grad():
            _, mastery_logits = encoder(pad_items, pad_scores, step_tensor)
            hat_alpha_final = torch.sigmoid(mastery_logits)  # [1, K]

        # ------------------- 在保留集上验证预测精度 -------------------
        holdout_items_tensor = torch.tensor(holdout_items, dtype=torch.long).to(device)

        with torch.no_grad():
            # 提取保留集题目的冻结参数
            e_d, e_a = ncdm.get_frozen_item_features(holdout_items_tensor)
            q_vec = q_matrix_tensor[holdout_items_tensor]

            # NCDM 前向预测：hat_alpha_final * Q * e_a - e_d
            interaction = hat_alpha_final * q_vec * e_a - e_d
            pred_probs = ncdm.interaction_mlp(interaction).squeeze(-1).cpu().numpy()

        all_holdout_y_true.extend(holdout_scores)
        all_holdout_y_pred.extend(pred_probs)

    # 汇总计算全局 AUC 和 RMSE
    auc = roc_auc_score(all_holdout_y_true, all_holdout_y_pred)
    rmse = np.sqrt(mean_squared_error(all_holdout_y_true, all_holdout_y_pred))
    avg_steps = total_steps / valid_students

    print(f"[Track B 结果] 参与评估真实学生数: {valid_students}")
    print(f"[Track B 结果] 平均测试长度: {avg_steps:.2f} 题")
    print(f"[Track B 结果] 保留集作答预测 AUC: {auc:.4f}")
    print(f"[Track B 结果] 保留集作答预测 RMSE: {rmse:.4f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"评估环境设备: {device}")

    # 路径配置 (与你的 Windows 11 工程结构保持一致)
    DATA_DIR = r"C:\Users\95215\PycharmProjects\CD_CAT_RL\data\processed"
    MODELS_DIR = r"C:\Users\95215\PycharmProjects\CD_CAT_RL\models\saved"
    TEST_CSV = os.path.join(DATA_DIR, 'test.csv')

    # 全局参数
    MAX_STEPS = 50
    TAU = 0.3  # 诊断终止的不确定性阈值

    # 统一加载权重
    print("正在加载 NCDM、Encoder 和 D3QN 的网络权重...")
    ncdm, encoder, d3qn, q_mat, m_probs_path = load_models(device, DATA_DIR, MODELS_DIR, MAX_STEPS)

    # 初始化 Track A 所需的 Env
    env = CDCATEnv(ncdm, encoder, q_mat.cpu().numpy(), m_probs_path,
                   max_steps=MAX_STEPS, tau=TAU, device=device)

    # 运行双轨评估
    evaluate_track_a(env, encoder, d3qn, num_simulated_students=500, device=device)
    evaluate_track_b(ncdm, encoder, d3qn, TEST_CSV, q_mat, MAX_STEPS, TAU, device=device)