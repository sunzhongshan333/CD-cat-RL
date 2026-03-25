import random
import torch
import numpy as np
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity, max_steps, device='cpu'):
        """
        阶段四：延迟编码经验回放池

        Args:
            capacity: 回放池最大容量
            max_steps: 历史记录的最大长度 (H_max)，用于采样时动态 Padding
            device: 张量所在的计算设备
        """
        self.buffer = deque(maxlen=capacity)
        self.max_steps = max_steps
        self.device = device

    def push(self, h_items, h_scores, step, action, reward,
             next_h_items, next_h_scores, next_step, mask, next_mask, done, true_alpha):
        """
        存入原始交互元组，而不是编码后的 s_t，严格遵循延迟编码设计。
        """
        self.buffer.append((
            h_items,  # 调用方已保证是快照（list 拷贝），无需再次复制
            h_scores,  # 同上
            step,  # 当前步数 t
            action,  # a_t
            reward,  # r_t
            list(next_h_items),  # Y_{t+1} 题目（env 引用，需在此处拷贝）
            list(next_h_scores),  # Y_{t+1} 结果（同上）
            next_step,  # 下一步数 t+1
            mask,  # R_t (可选题目掩码)
            next_mask,  # R_{t+1}
            done,  # 终止标志
            true_alpha  # 真实的 \alpha^* (用于 E-step 软标签生成)
        ))

    def sample(self, batch_size):
        """
        随机采样并动态 Padding 转化为 Tensor
        """
        batch = random.sample(self.buffer, batch_size)

        # 解包 Batch
        h_items_batch, h_scores_batch, step_batch, action_batch, reward_batch, \
            next_h_items_batch, next_h_scores_batch, next_step_batch, \
            mask_batch, next_mask_batch, done_batch, true_alpha_batch = zip(*batch)

        # ==========================================
        # 核心逻辑：动态 Padding，生成变长历史的 Tensor 表示
        # ==========================================
        pad_h_items = torch.full((batch_size, self.max_steps), -1, dtype=torch.long, device=self.device)
        pad_h_scores = torch.full((batch_size, self.max_steps), -1, dtype=torch.long, device=self.device)
        pad_next_h_items = torch.full((batch_size, self.max_steps), -1, dtype=torch.long, device=self.device)
        pad_next_h_scores = torch.full((batch_size, self.max_steps), -1, dtype=torch.long, device=self.device)

        for i in range(batch_size):
            len_t = len(h_items_batch[i])
            if len_t > 0:
                pad_h_items[i, :len_t] = torch.tensor(h_items_batch[i])
                pad_h_scores[i, :len_t] = torch.tensor(h_scores_batch[i])

            len_next_t = len(next_h_items_batch[i])
            if len_next_t > 0:
                pad_next_h_items[i, :len_next_t] = torch.tensor(next_h_items_batch[i])
                pad_next_h_scores[i, :len_next_t] = torch.tensor(next_h_scores_batch[i])

        # 转换为计算设备上的 Tensor
        step_tensor = torch.tensor(step_batch, dtype=torch.long).to(self.device)
        action_tensor = torch.tensor(action_batch, dtype=torch.long).unsqueeze(1).to(self.device)
        reward_tensor = torch.tensor(reward_batch, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_step_tensor = torch.tensor(next_step_batch, dtype=torch.long).to(self.device)
        done_tensor = torch.tensor(done_batch, dtype=torch.float32).unsqueeze(1).to(self.device)
        true_alpha_tensor = torch.stack(true_alpha_batch).to(self.device)

        mask_tensor = torch.stack(mask_batch).to(self.device)
        next_mask_tensor = torch.stack(next_mask_batch).to(self.device)

        return (pad_h_items, pad_h_scores, step_tensor,
                action_tensor, reward_tensor,
                pad_next_h_items, pad_next_h_scores, next_step_tensor,
                mask_tensor, next_mask_tensor, done_tensor, true_alpha_tensor)

    def __len__(self):
        return len(self.buffer)