import torch
import torch.nn as nn
import torch.nn.functional as F


class StateEncoder(nn.Module):
    def __init__(self, q_matrix_tensor, frozen_item_diff_tensor, frozen_item_disc_tensor,
                 d1=256, d2=256, max_steps=50):
        super(StateEncoder, self).__init__()
        """
        阶段二：置换不变状态编码器 (包含掌握概率映射头)

        Args:
            q_matrix_tensor: [J, K] 冻结的 Q矩阵
            frozen_item_diff_tensor: [J, K] 冻结的 NCDM题目难度嵌入
            frozen_item_disc_tensor: [J, K] 冻结的 NCDM题目区分度嵌入 (已加sigmoid)
            d1: 元素级变换 phi 的隐层和输出维度
            d2: 整体变换 rho 的隐层和输出维度 (h_t 的维度)
            max_steps: 测试最大题数上限 H_max
        """

        # 1. 注册基础数据 (注册为 buffer，跟随模型 save/load，且不作为需要优化的参数)
        self.register_buffer('q_matrix', q_matrix_tensor)  # [num_items, num_skills]
        self.register_buffer('item_diff', frozen_item_diff_tensor)  # [num_items, num_skills]
        # 注意：这里接收的应当是 train_ncdm.py 中处理好的、数值在 sigmoid 后的区分度
        self.register_buffer('item_disc', frozen_item_disc_tensor)  # [num_items, num_skills]

        num_items, num_skills = self.q_matrix.shape
        self.num_skills = num_skills
        self.max_steps = float(max_steps)

        # 特征构造维度定义: [Q_j + e_a_j + e_d_j + y_h] -> K + K + K + 1 = 3K+1
        self.feature_dim = 3 * num_skills + 1
        self.d1 = d1
        self.d2 = d2

        # 2. Deep Sets 结构定义

        # 2.1 元素级编码器 (MLP_phi): input x_h (3K+1) -> output phi_h (d1)
        self.phi_mlp = nn.Sequential(
            nn.Linear(self.feature_dim, d1),
            nn.ReLU(),
            nn.Dropout(0.1),  # E-step 训练时增加鲁棒性
            nn.Linear(d1, d1),
            nn.ReLU()
        )

        # 2.2 LayerNorm (关键！配合 Sum-Pooling 解决数值随答题数爆炸的问题)
        self.layer_norm = nn.LayerNorm(d1)

        # 2.3 整体变换器 (MLP_rho): input g_t_norm (d1) -> output h_t (d2)
        self.rho_mlp = nn.Sequential(
            nn.Linear(d1, d2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d2, d2),
            nn.ReLU()
        )

        # 3. 掌握概率映射头 (Mapping Head)
        # 输入 h_t (d2) -> 输出各知识点的掌握 Logit -> 后续加 Sigmoid 为概率
        # 对应框架 5.4 节：在线毫秒级推断的核心路径
        self.mapping_head = nn.Linear(d2, num_skills)

    def forward(self, history_item_ids, history_scores, current_steps):
        """
        接收变长的答题历史，进行置换不变编码，并输出完整状态表示 s_t。
        该函数支持批处理 (Batch Processing)，可以处理 Padding 过的序列。

        Args:
            history_item_ids: [batch_size, padded_len] - 历史题目ID。
                               由于 batch 内不同样本长度不同，不足 padded_len 的用 -1 填充。
            history_scores: [batch_size, padded_len]   - 历史作答结果(0/1)。
                               不足 padded_len 的用 -1 填充。
            current_steps: [batch_size] - 真实答题数 (t)。

        Returns:
            final_state: [batch_size, d2 + num_skills + 1] - 完整的 s_t，供 Q 网络输入。
            mastery_logits: [batch_size, num_skills] - 掌握概率的原始 Logits，
                                                     供 E-step 计算 BCEWithLogitsLoss 使用。
        """
        batch_size, padded_len = history_item_ids.shape
        device = history_item_ids.device

        # 0. 创建掩码：指示哪些位置是有效的历史记录 (ID 不为 -1)
        # shape: [batch_size, padded_len]
        valid_mask = (history_item_ids != -1)

        # 为了高效计算，将 batch 展平，只处理 valid 位置，最后再还原
        valid_mask_flat = valid_mask.view(-1)

        # ==========================================
        # 阶段二 A: 严格执行框架 5.2 节 特征构造 (x_h)
        # ==========================================
        # 只提取有效位的数据
        valid_item_ids = history_item_ids[valid_mask]  # [num_valid_history_points]
        valid_scores = history_scores[valid_mask]  # [num_valid_history_points]

        # 提取冻结的 NCDM 特征
        # shape: [num_valid_points, K]
        valid_q = self.q_matrix[valid_item_ids]
        valid_diff = self.item_diff[valid_item_ids]
        valid_disc = self.item_disc[valid_item_ids]

        # 格式化 scores 维度
        # shape: [num_valid_points, 1]
        valid_scores_vec = valid_scores.unsqueeze(-1).float()

        # 拼接特征 x_h = [Q, a, d, y]
        # x_h shape: [num_valid_history_points, 3K+1]
        x_h_valid = torch.cat([valid_q, valid_diff, valid_disc, valid_scores_vec], dim=1)

        # ==========================================
        # 阶段二 B: 执行 5.3 节 置换不变编码 (Deep Sets)
        # ==========================================
        # 2.1 元素级编码: phi_h = MLP_phi(x_h)
        phi_h_valid = self.phi_mlp(x_h_valid)  # [num_valid_points, d1]

        # 将计算完的 valid phi 还原回带 Padding 的形状
        # shape: [batch_size * padded_len, d1]
        phi_h_padded_flat = torch.zeros(batch_size * padded_len, self.d1, device=device)
        # scatter 散布回去
        phi_h_padded_flat[valid_mask_flat] = phi_h_valid
        # shape: [batch_size, padded_len, d1]
        phi_h_padded = phi_h_padded_flat.view(batch_size, padded_len, self.d1)

        # 2.2 Sum-Pooling 聚合: g_t = \sum phi_h
        # Padding 处全为 0，求和不贡献
        # shape: [batch_size, d1]
        g_t = torch.sum(phi_h_padded, dim=1)

        # 2.3 LayerNorm 归一化: g_t_norm
        # 这确保了数值稳定性，无论答了3题还是30题，输出都在合理范围
        # [batch_size, d1]
        g_t_norm = self.layer_norm(g_t)

        # 2.4 整体变换: h_t = MLP_rho(g_t_norm)
        # 对应框架 5.3 节：捕捉答题路径历史结构信息
        # h_t shape: [batch_size, d2]
        h_t = self.rho_mlp(g_t_norm)

        # ==========================================
        # 阶段二 C: 执行 5.4 节 掌握概率映射头
        # ==========================================
        # 这里输出 Logits (sigmoid 之前)
        # 用于 E-step 计算 BCE (数值更稳定) 以及推理时加 sigmoid 变成概率
        # mastery_logits shape: [batch_size, num_skills]
        mastery_logits = self.mapping_head(h_t)

        # 部署推断用的概率表示 alpha_hat_t
        hat_alpha_t = torch.sigmoid(mastery_logits)

        # ==========================================
        # 阶段二 D: 执行 5.5 节 构造完整状态表示 s_t
        # ==========================================
        # 归一化步数 (t / H_max)
        # shape: [batch_size, 1]
        norm_timesteps = (current_steps.float() / self.max_steps).unsqueeze(-1)

        # 最终状态拼接: s_t = [h_t | hat_alpha_t | t/H_max]
        # s_t 维度应当是: d2 + num_skills + 1
        # shape: [batch_size, d2 + num_skills + 1]
        final_state = torch.cat([h_t, hat_alpha_t, norm_timesteps], dim=1)

        return final_state, mastery_logits

    def get_hat_alpha(self, history_item_ids, history_scores, current_steps):
        """
        在线部署流程专用便捷方法 (8.2/8.7 节)
        仅输出掌握概率 \alpha_hat_t，不需要完整的状态向量。
        """
        with torch.no_grad():
            self.eval()  # 开启 eval 模式，关闭 Dropout
            _, mastery_logits = self.forward(history_item_ids, history_scores, current_steps)
            return torch.sigmoid(mastery_logits)