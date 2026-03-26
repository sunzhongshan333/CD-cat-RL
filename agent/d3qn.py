import torch
import torch.nn as nn
import torch.nn.functional as F


class D3QN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=None):
        """
        阶段三：Double Dueling DQN 决策网络

        Args:
            state_dim: 状态维度 (d2 + K + 1)
            action_dim: 动作空间维度 (总题库大小 J)
            hidden_dims: 共享隐藏层和分支隐藏层的维度配置
        """
        super(D3QN, self).__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]
        self.state_dim = state_dim
        self.action_dim = action_dim

        # 1. 共享特征提取层 (Shared Representation)
        shared_layers = []
        input_dim = state_dim
        for h_dim in hidden_dims:
            shared_layers.append(nn.Linear(input_dim, h_dim))
            shared_layers.append(nn.ReLU())
            input_dim = h_dim
        self.shared_net = nn.Sequential(*shared_layers)

        # 2. 状态价值分支 (Value Stream): V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # 输出标量 V(s)
        )

        # 3. 动作优势分支 (Advantage Stream): A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)  # 输出 J 维向量 A(s, \cdot)
        )

        # 4. 执行 Xavier 均匀初始化 (框架 6.2 节要求)
        self._initialize_weights()

    def _initialize_weights(self):
        """对所有线性层使用 Xavier Uniform 初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, state, action_mask):
        """
        前向传播，计算掩码后的 Q 值。

        Args:
            state: [batch_size, state_dim] 拼接好的状态 s_t
            action_mask: [batch_size, action_dim] 布尔张量，True 表示题目可选，False 表示已做过

        Returns:
            q_values: [batch_size, action_dim] 最终的 Q 值向量 (无效动作已被赋为 -1e9)
        """
        # 1. 提取共享特征
        features = self.shared_net(state)

        # 2. 计算 V(s) 和 A(s, a)
        value = self.value_stream(features)  # [batch_size, 1]
        advantage = self.advantage_stream(features)  # [batch_size, action_dim]

        # ==========================================
        # 3. 严格执行 6.1 节要求：仅在可选题目 R_t 上计算优势均值
        # ==========================================
        # 将不可选动作的优势值置为 0，避免干扰求和
        valid_advantage = advantage * action_mask.float()

        # 计算每个样本当前可选动作的数量
        # clamp(min=1.0) 是为了防止全被掩码时除以 0 导致 NaN
        num_valid_actions = action_mask.sum(dim=1, keepdim=True).float()
        num_valid_actions = torch.clamp(num_valid_actions, min=1.0)

        # 计算有效动作的平均优势值
        mean_valid_advantage = valid_advantage.sum(dim=1, keepdim=True) / num_valid_actions

        # 4. Dueling 聚合公式: Q(s,a) = V(s) + (A(s,a) - mean_valid_A)
        q_values = value + (advantage - mean_valid_advantage)

        # 5. 施加极大负值掩码，确保 argmax 永远不会选到已答题目 (框架 2.2 节要求)
        q_values = q_values.masked_fill(~action_mask, -1e9)

        return q_values