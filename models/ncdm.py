import torch
import torch.nn as nn
import torch.nn.functional as F


class NCDM(nn.Module):
    def __init__(self, num_students, num_items, num_skills, hidden_dims=[512, 256]):
        super(NCDM, self).__init__()
        self.num_students = num_students
        self.num_items = num_items
        self.num_skills = num_skills

        # 学生掌握程度嵌入 (e_s)
        self.student_emb = nn.Embedding(num_students, num_skills)

        # 题目难度嵌入 (e_d) 和 区分度嵌入 (e_a)
        self.item_diff = nn.Embedding(num_items, num_skills)
        self.item_disc = nn.Embedding(num_items, num_skills)

        # 交互函数 MLP
        layers = []
        input_dim = num_skills
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.Tanh())  # 或者 ReLU，但 Tanh 在认知诊断中通常更稳定
            layers.append(nn.Dropout(0.2))
            input_dim = h_dim
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())  # 输出 0-1 的作答概率

        self.interaction_mlp = nn.Sequential(*layers)

        # 初始化权重 (非常重要，保证初始概率合理)
        nn.init.xavier_uniform_(self.student_emb.weight)
        nn.init.xavier_uniform_(self.item_diff.weight)
        nn.init.xavier_uniform_(self.item_disc.weight)

    def forward(self, student_ids, item_ids, q_matrix):
        """
        student_ids: [batch_size]
        item_ids: [batch_size]
        q_matrix: [num_items, num_skills] 完整的 Q 矩阵
        """
        # 1. 获取对应的 Embedding
        # shape: [batch_size, num_skills]
        e_s = torch.sigmoid(self.student_emb(student_ids))
        e_d = self.item_diff(item_ids)
        e_a = torch.sigmoid(self.item_disc(item_ids))  # 区分度通常限制为正数

        # 2. 获取当前 batch 题目的 Q 矩阵行向量
        # shape: [batch_size, num_skills]
        q_vec = q_matrix[item_ids]

        # 3. 核心交互逻辑: e_s ⊙ Q ⊙ e_a - e_d
        # 只有该题考查的知识点 (q_vec=1) 才会参与运算
        interaction = e_s * q_vec * e_a - e_d

        # 4. MLP 预测答对概率
        pred_prob = self.interaction_mlp(interaction)

        return pred_prob.squeeze(-1)

    def get_frozen_item_features(self, item_ids):
        """
        部署到 RL 阶段时，用于提取冻结的题目特征，供特征工程使用
        """
        with torch.no_grad():
            e_d = self.item_diff(item_ids)
            e_a = torch.sigmoid(self.item_disc(item_ids))
        return e_d, e_a