import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split


def preprocess_assistments(raw_path, processed_dir):
    print("1. 读取原始数据...")
    # ASSISTments 2009 的标准列名
    df = pd.read_csv(raw_path, usecols=['order_id', 'user_id', 'problem_id', 'skill_id', 'correct'],
                     encoding='ISO-8859-1')

    # 丢弃没有技能标签的题目
    df = df.dropna(subset=['skill_id', 'problem_id', 'user_id', 'correct'])

    print("2. 执行清洗规则：保留首次尝试...")
    # 按时间顺序排序，确保 keep='first' 拿到的是真实的首次作答
    df = df.sort_values(by=['order_id'])
    df = df.drop_duplicates(subset=['user_id', 'problem_id'], keep='first')

    # 将 correct 严格二值化
    df['correct'] = df['correct'].apply(lambda x: 1 if x >= 1 else 0)

    print("3. 执行清洗规则：过滤低频数据...")
    # 丢弃被作答少于 10 次的题目
    item_counts = df['problem_id'].value_counts()
    valid_items = item_counts[item_counts >= 10].index
    df = df[df['problem_id'].isin(valid_items)]

    # 丢弃作答记录少于 5 条的学生
    user_counts = df['user_id'].value_counts()
    valid_users = user_counts[user_counts >= 5].index
    df = df[df['user_id'].isin(valid_users)]

    print(
        f"清洗后剩余: {df['user_id'].nunique()} 名学生, {df['problem_id'].nunique()} 道题目, {df['skill_id'].nunique()} 个知识点。")

    print("4. ID 重新映射 (连续化，供 PyTorch Embedding 使用)...")
    user_mapping = {u: i for i, u in enumerate(df['user_id'].unique())}
    item_mapping = {p: i for i, p in enumerate(df['problem_id'].unique())}
    skill_mapping = {s: i for i, s in enumerate(df['skill_id'].unique())}

    df['user_id'] = df['user_id'].map(user_mapping)
    df['problem_id'] = df['problem_id'].map(item_mapping)
    df['skill_id'] = df['skill_id'].map(skill_mapping)

    print("5. 构建 Q 矩阵...")
    num_items = len(item_mapping)
    num_skills = len(skill_mapping)
    q_matrix = np.zeros((num_items, num_skills), dtype=int)

    # ASSISTments 中一道题可能对应多行（不同 skill），遍历填充
    for _, row in df.iterrows():
        q_matrix[int(row['problem_id']), int(row['skill_id'])] = 1

    # 为了后续方便，把每人每题的作答压成单行（因为前面去重了，这里直接 groupby 即可）
    df_final = df[['user_id', 'problem_id', 'correct']].drop_duplicates()

    print("6. Student-level 7:1:2 数据划分...")
    users = df_final['user_id'].unique()

    # 先分出 70% 训练集，剩下 30%
    train_users, temp_users = train_test_split(users, test_size=0.3, random_state=42)
    # 再把剩下的 30% 按 1:2 划分为验证集(10%)和测试集(20%)
    valid_users, test_users = train_test_split(temp_users, test_size=2 / 3, random_state=42)

    train_df = df_final[df_final['user_id'].isin(train_users)]
    valid_df = df_final[df_final['user_id'].isin(valid_users)]
    test_df = df_final[df_final['user_id'].isin(test_users)]

    print("7. 保存处理结果...")
    os.makedirs(processed_dir, exist_ok=True)
    train_df.to_csv(os.path.join(processed_dir, 'train.csv'), index=False)
    valid_df.to_csv(os.path.join(processed_dir, 'valid.csv'), index=False)
    test_df.to_csv(os.path.join(processed_dir, 'test.csv'), index=False)
    np.save(os.path.join(processed_dir, 'q_matrix.npy'), q_matrix)

    print("预处理完成！全部文件已保存至:", processed_dir)


if __name__ == "__main__":
    # 使用你的绝对路径
    raw_csv_path = r"C:\Users\95215\PycharmProjects\CD_CAT_RL\data\raw\skill_builder_data.csv"
    processed_output_dir = r"C:\Users\95215\PycharmProjects\CD_CAT_RL\data\processed"

    preprocess_assistments(raw_csv_path, processed_output_dir)