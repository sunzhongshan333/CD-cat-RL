import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, mean_squared_error
from models.ncdm import NCDM  # 引入我们刚才写的模型


# 1. 定义 PyTorch Dataset
class ASSISTmentsDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.users = torch.tensor(self.data['user_id'].values, dtype=torch.long)
        self.items = torch.tensor(self.data['problem_id'].values, dtype=torch.long)
        self.labels = torch.tensor(self.data['correct'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]


# 2. 评估函数 (计算 AUC 和 RMSE)
def evaluate(model, dataloader, q_matrix, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for users, items, labels in dataloader:
            users, items = users.to(device), items.to(device)
            # NCDM 前向传播
            preds = model(users, items, q_matrix)

            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    auc = roc_auc_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return auc, rmse


# 3. 主训练循环
def train_ncdm_pipeline(data_dir, save_dir, batch_size=256, epochs=10, lr=0.002):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用设备: {device}")

    # 加载数据和 Q 矩阵
    train_dataset = ASSISTmentsDataset(os.path.join(data_dir, 'train.csv'))
    valid_dataset = ASSISTmentsDataset(os.path.join(data_dir, 'valid.csv'))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    q_matrix = np.load(os.path.join(data_dir, 'q_matrix.npy'))
    q_matrix_tensor = torch.tensor(q_matrix, dtype=torch.float32).to(device)

    # 统计全局数量以初始化模型
    # 注意：为了防止 Embedding 越界，总人数和题数应该从全部数据集的最大 ID 中推断，
    # 但由于我们之前做了连续重映射，这里直接取 Q 矩阵的 shape 和全体去重 user 数即可。
    num_items, num_skills = q_matrix.shape
    # 获取所有的 user_id 最大值以确定 Embedding 大小
    all_users = set(train_dataset.users.numpy()) | set(valid_dataset.users.numpy())
    num_students = max(all_users) + 1

    print(f"初始化 NCDM: 学生数={num_students}, 题目数={num_items}, 知识点数={num_skills}")
    model = NCDM(num_students, num_items, num_skills).to(device)

    # 损失函数与优化器 (严格的二元交叉熵)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_auc = 0.0

    print("开始训练 NCDM 教师模型...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for users, items, labels in train_loader:
            users, items, labels = users.to(device), items.to(device), labels.to(device)

            optimizer.zero_grad()
            preds = model(users, items, q_matrix_tensor)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 验证集评估
        val_auc, val_rmse = evaluate(model, valid_loader, q_matrix_tensor, device)
        print(
            f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss / len(train_loader):.4f} | Val AUC: {val_auc:.4f} | Val RMSE: {val_rmse:.4f}")

        # 保存最佳模型
        if val_auc > best_auc:
            best_auc = val_auc
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, 'ncdm_best.pth'))
            print("  --> 发现更优模型，已保存。")

    # ==========================================
    # 关键步骤：提取并保存训练集学生的先验知识分布
    # 对应框架 4.5 节：推断训练集学生的知识状态
    # ==========================================
    print("\n训练结束。开始提取训练集学生的知识掌握经验分布...")
    model.load_state_dict(torch.load(os.path.join(save_dir, 'ncdm_best.pth')))
    model.eval()

    with torch.no_grad():
        # 获取训练集中出现过的 user_ids
        train_user_ids = torch.tensor(list(set(train_dataset.users.numpy())), dtype=torch.long).to(device)
        # NCDM 的 student_emb 经过 sigmoid 就是掌握概率 [num_train_users, num_skills]
        mastery_probs = torch.sigmoid(model.student_emb(train_user_ids)).cpu().numpy()

    save_path = os.path.join(data_dir, 'train_student_mastery_probs.npy')
    np.save(save_path, mastery_probs)
    print(f"经验分布已保存至: {save_path} (Shape: {mastery_probs.shape})")
    print("阶段一圆满完成！")


if __name__ == "__main__":
    DATA_DIR = r"C:\Users\95215\PycharmProjects\CD_CAT_RL\data\processed"
    SAVE_DIR = r"C:\Users\95215\PycharmProjects\CD_CAT_RL\models\saved"

    train_ncdm_pipeline(data_dir=DATA_DIR, save_dir=SAVE_DIR)