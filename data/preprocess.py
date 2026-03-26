import logging
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def preprocess_assistments(raw_path, processed_dir):
    logger.info("1. 读取原始数据...")
    # ASSISTments 2009 的标准列名
    df = pd.read_csv(raw_path, usecols=['order_id', 'user_id', 'problem_id', 'skill_id', 'correct'],
                     encoding='ISO-8859-1')

    # 丢弃没有技能标签的题目
    df = df.dropna(subset=['skill_id', 'problem_id', 'user_id', 'correct'])

    logger.info("2. 执行清洗规则：保留首次尝试...")
    # 按时间顺序排序，确保 keep='first' 拿到的是真实的首次作答
    df = df.sort_values(by=['order_id'])
    df = df.drop_duplicates(subset=['user_id', 'problem_id'], keep='first')

    # 将 correct 严格二值化（向量化，避免逐行 apply）
    df['correct'] = (df['correct'] >= 1).astype(int)

    logger.info("3. 执行清洗规则：过滤低频数据...")
    # 丢弃被作答少于 10 次的题目
    item_counts = df['problem_id'].value_counts()
    valid_items = item_counts[item_counts >= 10].index
    df = df[df['problem_id'].isin(valid_items)]

    # 丢弃作答记录少于 5 条的学生
    user_counts = df['user_id'].value_counts()
    valid_users = user_counts[user_counts >= 5].index
    df = df[df['user_id'].isin(valid_users)]

    logger.info(
        "清洗后剩余: %d 名学生, %d 道题目, %d 个知识点。",
        df['user_id'].nunique(), df['problem_id'].nunique(), df['skill_id'].nunique()
    )

    logger.info("4. ID 重新映射 (连续化，供 PyTorch Embedding 使用)...")
    user_mapping = {u: i for i, u in enumerate(df['user_id'].unique())}
    item_mapping = {p: i for i, p in enumerate(df['problem_id'].unique())}
    skill_mapping = {s: i for i, s in enumerate(df['skill_id'].unique())}

    df['user_id'] = df['user_id'].map(user_mapping)
    df['problem_id'] = df['problem_id'].map(item_mapping)
    df['skill_id'] = df['skill_id'].map(skill_mapping)

    logger.info("5. 构建 Q 矩阵...")
    num_items = len(item_mapping)
    num_skills = len(skill_mapping)
    q_matrix = np.zeros((num_items, num_skills), dtype=int)

    # 向量化填充：直接用 numpy 花式索引，避免逐行 iterrows（O(N) Python 循环）
    q_matrix[df['problem_id'].astype(int).values, df['skill_id'].astype(int).values] = 1

    # 步骤 2 已对 (user_id, problem_id) 去重，此处直接投影即可
    df_final = df[['user_id', 'problem_id', 'correct']]

    logger.info("6. Student-level 7:1:2 数据划分...")
    users = df_final['user_id'].unique()

    # 先分出 70% 训练集，剩下 30%
    train_users, temp_users = train_test_split(users, test_size=0.3, random_state=42)
    # 再把剩下的 30% 按 1:2 划分为验证集(10%)和测试集(20%)
    valid_users, test_users = train_test_split(temp_users, test_size=2 / 3, random_state=42)

    train_df = df_final[df_final['user_id'].isin(train_users)]
    valid_df = df_final[df_final['user_id'].isin(valid_users)]
    test_df = df_final[df_final['user_id'].isin(test_users)]

    logger.info("7. 保存处理结果...")
    os.makedirs(processed_dir, exist_ok=True)
    train_df.to_csv(os.path.join(processed_dir, 'train.csv'), index=False)
    valid_df.to_csv(os.path.join(processed_dir, 'valid.csv'), index=False)
    test_df.to_csv(os.path.join(processed_dir, 'test.csv'), index=False)
    np.save(os.path.join(processed_dir, 'q_matrix.npy'), q_matrix)

    logger.info("预处理完成！全部文件已保存至: %s", processed_dir)


if __name__ == "__main__":
    # 将项目根目录加入 sys.path，以便从 data/ 子目录中直接运行脚本时能找到 config.py
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import RAW_DATA_PATH, DATA_DIR

    preprocess_assistments(RAW_DATA_PATH, DATA_DIR)