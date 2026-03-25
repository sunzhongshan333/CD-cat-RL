import logging
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm  # 新增进度条包

# 导入我们的模块
from models.ncdm import NCDM
from models.encoder import StateEncoder
from agent.d3qn import D3QN
from agent.replay_buffer import ReplayBuffer
from env.cdcat_env import CDCATEnv
import config as cfg

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


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
    T_update = cfg.RL_T_UPDATE
    N_alt = cfg.RL_N_ALT
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
        ncdm.load_state_dict(torch.load(ncdm_ckpt, map_location=device))
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
    mse_loss = nn.MSELoss()

    # 余弦退火学习率调度：从 lr 衰减到接近 0，兼顾前期快速收敛与后期精细调整
    scheduler_encoder = optim.lr_scheduler.CosineAnnealingLR(
        opt_encoder, T_max=max_episodes, eta_min=lr_encoder * 0.01)
    scheduler_d3qn = optim.lr_scheduler.CosineAnnealingLR(
        opt_d3qn, T_max=max_episodes, eta_min=lr_d3qn * 0.01)

    # 回放池
    buffer = ReplayBuffer(buffer_capacity, max_steps, device)

    # ==========================================
    # 4. 主训练循环 (交替优化范式)
    # ==========================================
    total_steps = 0
    logger.info("开始强化学习范式训练...")

    # 使用 tqdm 包裹训练循环，创建可视化进度条
    pbar = tqdm(range(1, max_episodes + 1), desc="RL Training")

    for episode in pbar:
        # 决定当前是 E-step 还是 Q-step (严格执行框架 7.3)
        is_e_step = ((episode - 1) // N_alt) % 2 == 0

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

            # --- 动作选择 (epsilon-greedy) ---
            if np.random.rand() < epsilon:
                # 随机选择一个有效的题目
                valid_actions = torch.where(mask_t)[0].cpu().numpy()
                action = np.random.choice(valid_actions)
            else:
                # D3QN 贪心选择
                with torch.no_grad():
                    q_values = main_d3qn(s_t.unsqueeze(0), mask_t.unsqueeze(0))
                    action = q_values.argmax(dim=1).item()

            # --- 执行动作，获取转移 ---
            # 记录执行前的状态变量供回放池使用
            h_items_t = list(env.history_item_ids)
            h_scores_t = list(env.history_scores)
            step_t = env.current_step
            true_alpha_t = env.alpha_star.clone()

            s_next, reward, done, info = env.step(action)
            mask_next = env.get_action_mask()
            total_steps += 1

            # --- 存入延迟编码回放池 ---
            buffer.push(h_items_t, h_scores_t, step_t, action, reward,
                        env.history_item_ids, env.history_scores, env.current_step,
                        mask_t, mask_next, done, true_alpha_t)

            s_t = s_next

            # ==========================================
            # 5. 核心优化步骤 (Batch 训练)
            # ==========================================
            if buffer.is_ready(batch_size):
                # 采样 Batch
                b_h_items, b_h_scores, b_steps, b_actions, b_rewards, \
                    b_next_h_items, b_next_h_scores, b_next_steps, \
                    b_masks, b_next_masks, b_dones, b_true_alphas = buffer.sample(batch_size)

                if is_e_step:
                    # ------------------------------------------
                    # E-step: 表征学习 (优化 Encoder)
                    # ------------------------------------------
                    opt_encoder.zero_grad()

                    # 1. 实时生成当前状态
                    s_batch, mastery_logits = encoder(b_h_items, b_h_scores, b_steps)
                    hat_alpha = torch.sigmoid(mastery_logits)  # [batch, K]

                    # 2. 构建学生预测的 interaction logits（未答题目上计算）
                    interaction_pred = hat_alpha.unsqueeze(1) * q_matrix_tensor.unsqueeze(0) * frozen_e_a.unsqueeze(
                        0) - frozen_e_d.unsqueeze(0)
                    pred_logits = ncdm.interaction_mlp(interaction_pred).squeeze(-1)  # [batch, num_items]

                    # 3. 构建教师给出的真实软标签 (使用模拟的真实状态 alpha^*)
                    with torch.no_grad():
                        interaction_teacher = b_true_alphas.unsqueeze(1) * q_matrix_tensor.unsqueeze(
                            0) * frozen_e_a.unsqueeze(0) - frozen_e_d.unsqueeze(0)
                        # 教师输出加 sigmoid，作为 [0,1] 软标签供 BCEWithLogitsLoss 使用
                        target_y = torch.sigmoid(ncdm.interaction_mlp(interaction_teacher).squeeze(-1))

                    # 4. 计算辅助 BCE 损失 (BCEWithLogitsLoss 接收 logit 输入，数值稳定)
                    loss_aux = bce_loss(pred_logits[b_masks], target_y[b_masks])
                    loss_aux.backward()
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), grad_clip)
                    opt_encoder.step()

                else:
                    # ------------------------------------------
                    # Q-step: 策略学习 (优化 D3QN)
                    # ------------------------------------------
                    opt_d3qn.zero_grad()

                    with torch.no_grad():
                        # 将当前状态与下一状态拼接，一次 encoder forward 完成两次编码
                        # 节省约 30~40% 的 encoder 推断时间（LayerNorm 等算子可并行）
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

                    loss_td = mse_loss(q_eval, target_q)
                    loss_td.backward()
                    torch.nn.utils.clip_grad_norm_(main_d3qn.parameters(), grad_clip)
                    opt_d3qn.step()

            # --- 目标网络同步 ---
            if total_steps % T_update == 0:
                target_d3qn.load_state_dict(main_d3qn.state_dict())

        # ==========================================
        # 6. 周期性日志打印与模型保存
        # ==========================================
        # 每个 episode 结束后推进学习率调度
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