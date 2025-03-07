import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 读取 Excel 文件
file_path = '风险计算.xlsx'  # Excel 文件路径
df = pd.read_excel(file_path)

# 提取年份列
years = df.columns[4:]  # 假设第一列是子类别、第二列是指标名称、第三列是单位、第四列是正负性

# 定义两个空的 DataFrame，用来存储得分和权重
scores_df = pd.DataFrame(index=years)  # 存储每年每个类别的得分
weights_list = []  # 存储每个类别的指标和权重

# 计算信息熵并确定权重
def calculate_entropy(df_norm):
        k = 1 / np.log(len(df_norm))  # 熵的常数因子
        p = df_norm / df_norm.sum(axis=0)  # 计算每个值的概率
        entropy = -k * (p * np.log(p)).sum(axis=0)  # 计算熵
        return entropy

# 定义一个函数用于熵权TOPSIS计算
def entropy_topsis(category):
    global scores_df, weights_list  # 引用外部的 DataFrame

    # 筛选出特定类别的指标
    df_category = df[df['子类别'] == category]
    
    if df_category.empty:
        print(f"类别 {category} 没有数据。")
        return

    # 将数据从 Excel 中的子类别、指标、单位列提取出来
    df_data = df_category.iloc[:, 4:].copy()  # 提取所有年份的指标数据
    df_data = df_data.apply(pd.to_numeric, errors='coerce')  # 确保数据是数值类型
    df_t = df_data.T  # 转置，便于后续处理

    # 标准化数据
    scaler = MinMaxScaler()
    df_t_norm = pd.DataFrame(scaler.fit_transform(df_t), columns=df_t.columns, index=df_t.index)
    df_data_norm = df_t_norm.T  # 转置回原来的格式

    # 处理正负性
    for i, row in df_category.iterrows():
        if row['方向'] == '-':  # 对负向指标取反
            df_data_norm.loc[i] = 1 - df_data_norm.loc[i]

    df_data_norm += 0.00001  # 防止除以零
    df_data_norm = df_data_norm.T  # 转置回原格式

    # 计算每个指标的熵值
    entropy = calculate_entropy(df_data_norm)
    print(f"\n类别 {category} 的各指标的熵值：\n", entropy)

    # 计算每个指标的权重
    weights = (1 - entropy) / (1 - entropy).sum()
    print(f"\n类别 {category} 的各指标的权重：\n", weights)

    # 将权重添加到weights_list中，记录每个类别的指标和权重
    for idx, weight in zip(df_category['指标名称'], weights):
        weights_list.append([category, idx, weight])
    
    Q=weights.T*(df_data_norm-0.00001)

    # 计算理想解和负理想解
    ideal_solution = Q.max(axis=0)  # 正理想解（最大值）
    negative_ideal_solution = Q.min(axis=0)  # 负理想解（最小值）

    # 计算每个选项与理想解和负理想解的距离
    distance_to_ideal = np.sqrt(((Q - ideal_solution) ** 2).sum(axis=1))  # 距离正理想解
    distance_to_negative_ideal = np.sqrt(((Q - negative_ideal_solution) ** 2).sum(axis=1))  # 距离负理想解

    # 计算每个选项的综合评分
    scores = distance_to_negative_ideal / (distance_to_ideal + distance_to_negative_ideal)

    # 将得分添加到scores_df中
    scores_df[category] = scores

# 定义需要计算的类别
categories = ['社会', '经济', '制度', '生态', '设施']

# 对每个类别分别计算熵权TOPSIS结果
for category in categories:
    entropy_topsis(category)


# 将权重结果转换为DataFrame，并输出
weights_df = pd.DataFrame(weights_list, columns=['类别', '指标', '指标权重'])

# 计算每个指标的熵值
entropy = calculate_entropy(scores_df)

# 计算每个指标的权重
sector_weights = (1 - entropy) / (1 - entropy).sum()

#计算加权后的矩阵
Q_sector=sector_weights.T*scores_df

# 计算理想解和负理想解
ideal_solution = Q_sector.max(axis=0)  # 正理想解（最大值）
negative_ideal_solution = Q_sector.min(axis=0)  # 负理想解（最小值）

# 计算每个选项与理想解和负理想解的距离
distance_to_ideal = np.sqrt(((Q_sector - ideal_solution) ** 2).sum(axis=1))  # 距离正理想解
distance_to_negative_ideal = np.sqrt(((Q_sector - negative_ideal_solution) ** 2).sum(axis=1))  # 距离负理想解

# 计算每个选项的综合评分
total_scores = distance_to_negative_ideal / (distance_to_ideal + distance_to_negative_ideal)

#合并各类别评分
year_df=scores_df.assign(综合=total_scores)

#计算每个指标的权重
# 通过类别名将df1（Series）中的权重值添加到df2中
weights_df['类别权重'] = weights_df['类别'].map(sector_weights)

# 计算调整后的指标权重
weights_df['调整后指标权重'] = weights_df['指标权重'] * weights_df['类别权重']

#输出各类别评分和综合评分
year_df.to_excel('scores.xlsx', index=True, sheet_name='Sheet1')

#输出指标权重和类别权重
weights_df.to_excel('weights.xlsx', index=False, sheet_name='Sheet1')
