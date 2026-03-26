import random
import logging
import torch
import numpy as np

logger = logging.getLogger(__name__)


class SumTree:
    """
    二叉和树：叶节点存储优先级，内部节点存储子树优先级之和。
    支持 O(log n) 比例采样和优先级更新。
    """

    def __init__(self, capacity):
        self.capacity = capacity
        # 树共 2*capacity-1 个节点；叶节点从下标 capacity-1 开始
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.n_entries = 0
        self.write = 0  # 循环写指针

    def _propagate(self, leaf_idx, change):
        """从叶节点向上传播优先级变化量。"""
        idx = leaf_idx
        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def _retrieve(self, s):
        """查找累积优先级为 s 对应的叶节点下标。"""
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                return idx
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right

    def total(self):
        return float(self.tree[0])

    def add(self, priority, data):
        leaf_idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(leaf_idx, priority)
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, leaf_idx, priority):
        change = priority - self.tree[leaf_idx]
        self.tree[leaf_idx] = priority
        self._propagate(leaf_idx, change)

    def get(self, s):
        """返回 (leaf_idx, priority, data)。"""
        leaf_idx = self._retrieve(s)
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    def __init__(self, capacity, max_steps, device='cpu',
                 alpha=0.6, beta_start=0.4, beta_frames=100000, eps=1e-6):
        """
        优先经验回放（PER）缓冲区，基于 SumTree 实现 O(log n) 比例采样。

        Args:
            capacity:     缓冲区最大容量
            max_steps:    历史记录最大长度（H_max），用于 Padding
            device:       张量所在计算设备
            alpha:        优先级指数（0=均匀，1=完全按优先级）
            beta_start:   IS 权重初始 β
            beta_frames:  β 从 beta_start 线性退火到 1.0 的帧数
            eps:          优先级平滑常数，防止零优先级
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.max_steps = max_steps
        self.device = device
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.eps = eps
        self.frame = 0
        self._max_priority = 1.0

    def push(self, h_items, h_scores, step, action, reward,
             next_h_items, next_h_scores, next_step, mask, next_mask, done, true_alpha):
        """
        存入原始交互元组（push 时即转换为固定长度 numpy 数组）。
        新经验以当前最大优先级入队，保证至少被采样一次。
        """
        def _to_fixed(lst):
            arr = np.full(self.max_steps, -1, dtype=np.int64)
            if lst:
                arr[:len(lst)] = lst
            return arr

        transition = (
            _to_fixed(h_items),
            _to_fixed(h_scores),
            step,
            action,
            reward,
            _to_fixed(next_h_items),
            _to_fixed(next_h_scores),
            next_step,
            mask,
            next_mask,
            done,
            true_alpha,
        )
        self.tree.add(self._max_priority, transition)

    def sample(self, batch_size):
        """
        分层 PER 采样，返回批数据、叶节点下标和 IS 权重。

        Returns:
            (batch tensors..., indices: list[int], is_weights: [batch, 1] FloatTensor)
        """
        self.frame += 1
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

        indices = []
        priorities = []
        batch_data = []

        # 分层采样：将优先级总和等分为 batch_size 段，每段内均匀采样
        total = self.tree.total()
        segment = total / batch_size
        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            s = random.uniform(lo, hi)
            leaf_idx, priority, data = self.tree.get(s)
            if data is None:
                # 安全回退（理论上 is_ready 保护后不会触发，若触发说明存在 bug）
                logger.warning(
                    "PER SumTree: sampled None data at leaf_idx=%d; falling back to midpoint.", leaf_idx)
                leaf_idx, priority, data = self.tree.get(total * 0.5)
            indices.append(leaf_idx)
            priorities.append(max(float(priority), self.eps))
            batch_data.append(data)

        # 计算 IS 权重并归一化到 [0, 1]
        probs = np.array(priorities, dtype=np.float64) / total
        is_weights = (self.tree.n_entries * probs) ** (-beta)
        is_weights /= is_weights.max()
        is_weights_tensor = torch.tensor(
            is_weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        # 解包批数据
        h_items_batch, h_scores_batch, step_batch, action_batch, reward_batch, \
            next_h_items_batch, next_h_scores_batch, next_step_batch, \
            mask_batch, next_mask_batch, done_batch, true_alpha_batch = zip(*batch_data)

        pad_h_items = torch.tensor(
            np.stack(h_items_batch), dtype=torch.long, device=self.device)
        pad_h_scores = torch.tensor(
            np.stack(h_scores_batch), dtype=torch.long, device=self.device)
        pad_next_h_items = torch.tensor(
            np.stack(next_h_items_batch), dtype=torch.long, device=self.device)
        pad_next_h_scores = torch.tensor(
            np.stack(next_h_scores_batch), dtype=torch.long, device=self.device)

        step_tensor       = torch.tensor(step_batch,      dtype=torch.long,    device=self.device)
        action_tensor     = torch.tensor(action_batch,    dtype=torch.long,    device=self.device).unsqueeze(1)
        reward_tensor     = torch.tensor(reward_batch,    dtype=torch.float32, device=self.device).unsqueeze(1)
        next_step_tensor  = torch.tensor(next_step_batch, dtype=torch.long,    device=self.device)
        done_tensor       = torch.tensor(done_batch,      dtype=torch.float32, device=self.device).unsqueeze(1)
        true_alpha_tensor = torch.stack(true_alpha_batch).to(self.device)

        mask_tensor       = torch.stack(mask_batch).to(self.device)
        next_mask_tensor  = torch.stack(next_mask_batch).to(self.device)

        return (pad_h_items, pad_h_scores, step_tensor,
                action_tensor, reward_tensor,
                pad_next_h_items, pad_next_h_scores, next_step_tensor,
                mask_tensor, next_mask_tensor, done_tensor, true_alpha_tensor,
                indices, is_weights_tensor)

    def update_priorities(self, indices, td_errors):
        """
        根据 TD 误差更新对应经验的优先级。
        priority = (|td_error| + eps)^alpha
        """
        for leaf_idx, td_error in zip(indices, td_errors):
            priority = (float(abs(td_error)) + self.eps) ** self.alpha
            self.tree.update(leaf_idx, priority)
            self._max_priority = max(self._max_priority, priority)

    def is_ready(self, batch_size):
        """回放池是否已积累足够经验（warm-up 阈值）"""
        return self.tree.n_entries >= batch_size * 2

    def __len__(self):
        return self.tree.n_entries
