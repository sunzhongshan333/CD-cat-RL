import numpy as np
import torch
import config as cfg


class CDCATEnv:
    def __init__(self, ncdm_model, encoder_model, q_matrix, mastery_probs_path,
                 max_steps=50, tau=None, beta=None, epsilon=None, device='cpu'):
        """
        阶段三/四：CD-CAT 强化学习交互环境 (Gymnasium-style)
        """
        self.device = device
        self.ncdm = ncdm_model.to(self.device)
        self.encoder = encoder_model.to(self.device)
        self.ncdm.eval()  # 严格冻结教师模型
        self.encoder.eval()  # 初始化为 eval；_get_current_state_and_entropy 在推断时会按需还原

        self.q_matrix = torch.tensor(q_matrix, dtype=torch.float32).to(self.device)
        self.num_items, self.num_skills = self.q_matrix.shape

        # 框架配置参数
        self.max_steps = max_steps
        self.tau = tau if tau is not None else cfg.CDCAT_TAU
        self.beta = beta if beta is not None else cfg.CDCAT_BETA
        self.epsilon = epsilon if epsilon is not None else cfg.CDCAT_EPSILON

        # 加载阶段一生成的训练集学生先验经验分布 (Shape: [num_train_users, K])
        self.empirical_mastery_probs = np.load(mastery_probs_path)
        self.num_empirical_students = self.empirical_mastery_probs.shape[0]

        # 内部状态变量
        self.alpha_star = None  # 模拟学生的真实知识状态
        self.history_item_ids = []  # 答题历史 ID
        self.history_scores = []  # 答题历史 结果
        self.available_items = set(range(self.num_items))  # 剩余可选题目池
        self.current_step = 0
        self.current_mean_entropy = None  # \bar{H}_t

        # 预分配热路径张量，避免每步重复申请 GPU 内存
        self._padded_items = torch.full((1, self.max_steps), -1, dtype=torch.long, device=self.device)
        self._padded_scores = torch.full((1, self.max_steps), -1, dtype=torch.long, device=self.device)
        self._step_tensor = torch.zeros(1, dtype=torch.long, device=self.device)

    def _sample_simulated_student(self):
        """
        严格对应框架 7.2 节：鲁棒经验采样生成器
        """
        if np.random.rand() < self.epsilon:
            # \epsilon 概率退化为全空间均匀采样，保证探索多样性
            probs = np.random.uniform(0, 1, size=(self.num_skills,))
        else:
            # 1-\epsilon 概率从真实经验分布中有放回重采样
            idx = np.random.randint(0, self.num_empirical_students)
            probs = self.empirical_mastery_probs[idx]

        # 独立伯努利采样生成真实的二值知识状态 \alpha^* \in {0,1}^K
        alpha_star = np.random.binomial(1, probs).astype(np.float32)
        return torch.tensor(alpha_star).to(self.device)

    def _calculate_entropy(self, mastery_probs):
        """
        计算掌握概率的二值熵平均值 \bar{H}_t
        """
        # 截断以防止 log(0)
        p = torch.clamp(mastery_probs, 1e-7, 1.0 - 1e-7)
        entropy = -p * torch.log(p) - (1 - p) * torch.log(1 - p)
        # 返回最大熵和平均熵
        return torch.max(entropy).item(), torch.mean(entropy).item()

    def _get_current_state_and_entropy(self):
        """
        调用编码器生成当前状态 s_t，并计算当前的系统不确定性。
        数据收集阶段强制使用 eval 模式，确保 dropout 不干扰经验质量。
        """
        # 复用预分配的 padded tensor，避免热路径上反复申请内存
        self._padded_items.fill_(-1)
        self._padded_scores.fill_(-1)

        if self.current_step > 0:
            t = self.current_step
            self._padded_items[0, :t] = torch.tensor(
                self.history_item_ids, dtype=torch.long, device=self.device)
            self._padded_scores[0, :t] = torch.tensor(
                self.history_scores, dtype=torch.long, device=self.device)

        self._step_tensor.fill_(self.current_step)

        # 环境推断阶段强制 eval 模式（防止 E-step 训练时 dropout 污染状态）
        was_training = self.encoder.training
        self.encoder.eval()
        with torch.no_grad():
            s_t, mastery_logits = self.encoder(
                self._padded_items, self._padded_scores, self._step_tensor)
            hat_alpha_t = torch.sigmoid(mastery_logits).squeeze(0)  # [num_skills]
        if was_training:
            self.encoder.train()

        max_entropy, mean_ent = self._calculate_entropy(hat_alpha_t)
        return s_t.squeeze(0), hat_alpha_t, max_entropy, mean_ent

    def reset(self):
        """
        重置环境，开始一个新的 Episode
        """
        # 1. 生成模拟学生
        self.alpha_star = self._sample_simulated_student()

        # 2. 清空历史
        self.history_item_ids = []
        self.history_scores = []
        self.available_items = set(range(self.num_items))
        self.current_step = 0

        # 3. 获取初始状态 s_0 (全空历史)
        s_0, _, _, mean_ent = self._get_current_state_and_entropy()
        self.current_mean_entropy = mean_ent

        return s_0

    def step(self, action_item_id):
        """
        执行选题动作，模拟作答，返回下一步转移
        """
        action_item_id = int(action_item_id)
        if action_item_id not in self.available_items:
            raise ValueError(f"动作 {action_item_id} 已经选过或不在题库中！")

        # ==========================================
        # 1. 模拟环境转移 (学生作答) -> 框架 2.3 节
        # ==========================================
        item_tensor = torch.tensor([action_item_id], dtype=torch.long).to(self.device)
        q_vec = self.q_matrix[item_tensor]  # [1, K]

        with torch.no_grad():
            # 获取冻结的题目参数
            e_d, e_a = self.ncdm.get_frozen_item_features(item_tensor)

            # 核心机制：绕过 embedding，直接用 alpha_star 与题目参数交互
            interaction = self.alpha_star.unsqueeze(0) * q_vec * e_a - e_d
            # 真实作答模拟 (NCDM 输出 logit，加 sigmoid 转为概率后伯努利采样)
            pred_prob = torch.sigmoid(self.ncdm.interaction_mlp(interaction)).squeeze(-1).item()

            # 伯努利硬采样：将概率转为 0/1 作答
            y_t = np.random.binomial(1, pred_prob)

        # ==========================================
        # 2. 更新系统信念状态
        # ==========================================
        self.history_item_ids.append(action_item_id)
        self.history_scores.append(y_t)
        self.available_items.remove(action_item_id)
        self.current_step += 1

        s_next, hat_alpha_next, max_entropy, mean_ent_next = self._get_current_state_and_entropy()

        # ==========================================
        # 3. 计算混合奖励 -> 框架 2.4 节
        # ==========================================
        entropy_reduction = self.current_mean_entropy - mean_ent_next
        self.current_mean_entropy = mean_ent_next  # 更新状态

        # 检查终止条件 (框架 2.5 节)
        done = False

        if max_entropy < self.tau:
            # 熵达标，诊断成功提前终止，给予正奖励激励智能体尽早完成
            done = True
            reward = 1.0
        else:
            # 未达标（含步数耗尽）：硬截断防止刷正奖励
            if self.current_step >= self.max_steps:
                done = True
            reward = min(-1.0 + self.beta * entropy_reduction, 0.0)

        info = {
            'y_t': y_t,
            'true_alpha': self.alpha_star.cpu().numpy(),
            'pred_alpha': hat_alpha_next.cpu().numpy(),
            'entropy_reduction': entropy_reduction,
            'max_entropy': max_entropy
        }

        return s_next, reward, done, info

    def get_action_mask(self):
        """
        获取当前可用动作的掩码，用于给 Q 网络的输出施加 -10^9 惩罚
        返回: [num_items] 的布尔型 tensor，True 表示可选，False 表示不可选
        """
        mask = torch.zeros(self.num_items, dtype=torch.bool, device=self.device)
        valid_indices = list(self.available_items)
        if valid_indices:
            mask[valid_indices] = True
        return mask