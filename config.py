import os

# ==========================================
# 项目根目录与路径配置（跨平台，无硬编码）
# ==========================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 全局随机种子（保证实验可复现）
# ==========================================
RANDOM_SEED = 42

# 数据目录
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'skill_builder_data.csv')

# 模型保存目录
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models', 'saved')

# ==========================================
# NCDM 训练超参数
# ==========================================
NCDM_BATCH_SIZE = 256
NCDM_EPOCHS = 10
NCDM_LR = 0.002

# ==========================================
# RL 训练超参数
# ==========================================
RL_MAX_STEPS = 50           # H_max：每轮测试最大题数
RL_BATCH_SIZE = 128
RL_GAMMA = 0.99             # 奖励折扣因子
RL_LR_ENCODER = 1e-4        # 编码器学习率
RL_LR_D3QN = 1e-4           # 策略网络学习率
RL_BUFFER_CAPACITY = 50000
RL_T_UPDATE = 500           # 目标网络同步频率（步数）
RL_N_ALT = 10               # 交替优化频率：每 N 个 Episode 切换一次阶段
RL_MAX_EPISODES = 5000      # 总训练轮数
RL_EPSILON_START = 1.0      # 初始探索率
RL_EPSILON_END = 0.05       # 最低探索率
RL_EPSILON_DECAY = 2000     # 探索率衰减控制（单位：episode）
RL_GRAD_CLIP = 10.0         # 梯度裁剪最大范数（DQN 场景 Q 值梯度量级较大，10.0 为常用经验值）

# ==========================================
# CD-CAT 环境超参数
# ==========================================
CDCAT_TAU = 0.3             # 诊断终止的不确定性（最大熵）阈值
CDCAT_BETA = 0.05           # 奖励中的熵减权重
CDCAT_EPSILON = 0.1         # 经验分布退化为均匀采样的概率

# ==========================================
# 评估超参数
# ==========================================
EVAL_MAX_STEPS = 50
EVAL_TAU = CDCAT_TAU        # 与训练环境保持一致
EVAL_NUM_SIMULATED = 500    # Track A 虚拟学生数量
EVAL_CHECKPOINT_EP = 5000   # 加载第几轮的 checkpoint 权重
