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
        存入原始交互元组，在 push 时即转换为固定长度 numpy 数组，
        从而在高频采样时避免逐样本循环创建临时张量。
        """
        def _to_fixed(lst):
            arr = np.full(self.max_steps, -1, dtype=np.int64)
            if lst:
                arr[:len(lst)] = lst
            return arr

        self.buffer.append((
            _to_fixed(h_items),           # [max_steps] int64，当前历史（调用方已保证为快照）
            _to_fixed(h_scores),          # [max_steps] int64
            step,                         # 当前步数 t
            action,                       # a_t
            reward,                       # r_t
            _to_fixed(next_h_items),      # [max_steps] int64，下一时刻历史（env 引用，_to_fixed 在此复制）
            _to_fixed(next_h_scores),     # [max_steps] int64
            next_step,                    # 下一步数 t+1
            mask,                         # R_t (可选题目掩码)
            next_mask,                    # R_{t+1}
            done,                         # 终止标志
            true_alpha,                   # 真实的 \alpha^* (用于 E-step 软标签生成)
        ))

    def sample(self, batch_size):
        """
        随机采样并批量转化为 Tensor。
        由于 push 时已做固定长度 numpy 数组转换，
        这里直接 np.stack + 一次性 to(device)，消除逐样本循环。
        """
        batch = random.sample(self.buffer, batch_size)

        # 解包 Batch
        h_items_batch, h_scores_batch, step_batch, action_batch, reward_batch, \
            next_h_items_batch, next_h_scores_batch, next_step_batch, \
            mask_batch, next_mask_batch, done_batch, true_alpha_batch = zip(*batch)

        # ==========================================
        # numpy stack → 一次性创建设备张量，无循环 overhead
        # ==========================================
        pad_h_items = torch.tensor(
            np.stack(h_items_batch), dtype=torch.long, device=self.device)
        pad_h_scores = torch.tensor(
            np.stack(h_scores_batch), dtype=torch.long, device=self.device)
        pad_next_h_items = torch.tensor(
            np.stack(next_h_items_batch), dtype=torch.long, device=self.device)
        pad_next_h_scores = torch.tensor(
            np.stack(next_h_scores_batch), dtype=torch.long, device=self.device)

        step_tensor      = torch.tensor(step_batch,      dtype=torch.long,    device=self.device)
        action_tensor    = torch.tensor(action_batch,    dtype=torch.long,    device=self.device).unsqueeze(1)
        reward_tensor    = torch.tensor(reward_batch,    dtype=torch.float32, device=self.device).unsqueeze(1)
        next_step_tensor = torch.tensor(next_step_batch, dtype=torch.long,    device=self.device)
        done_tensor      = torch.tensor(done_batch,      dtype=torch.float32, device=self.device).unsqueeze(1)
        true_alpha_tensor = torch.stack(true_alpha_batch).to(self.device)

        mask_tensor      = torch.stack(mask_batch).to(self.device)
        next_mask_tensor = torch.stack(next_mask_batch).to(self.device)

        return (pad_h_items, pad_h_scores, step_tensor,
                action_tensor, reward_tensor,
                pad_next_h_items, pad_next_h_scores, next_step_tensor,
                mask_tensor, next_mask_tensor, done_tensor, true_alpha_tensor)

    def is_ready(self, batch_size):
        """回放池是否已积累足够的经验供训练（warm-up 阈值）"""
        return len(self.buffer) >= batch_size * 2

    def __len__(self):
        return len(self.buffer)
