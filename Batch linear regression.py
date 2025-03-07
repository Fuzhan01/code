import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

# 读取数据
df = pd.read_excel('数据需求.xlsx', sheet_name="插值",index_col=0)
df_interpolated = df.copy()

# 定义外推的时间范围
output_years = np.arange(2000, 2024)

for index, row in df.iterrows():
    # 提取非缺失数据并排序
    non_missing = row.dropna()
    if len(non_missing) >= 2:
        non_missing_years = non_missing.index.astype(int)
        non_missing_values = non_missing.values
        
        # 按年份递增排序
        sorted_idx = np.argsort(non_missing_years)
        years_sorted = non_missing_years[sorted_idx]
        values_sorted = non_missing_values[sorted_idx]
        
        # 创建线性插值函数，设置允许外推
        linear_interp = interp1d(years_sorted, values_sorted, kind='linear', fill_value='extrapolate')
        
        # 计算输出时间范围内的所有年份值
        interpolated_values = linear_interp(output_years)
        
        # 确保拟合结果大于等于0
        interpolated_values = np.maximum(interpolated_values, 0)  # 将负值替换为0
        
        # 将原始数据放回到结果中（未缺失的数据不改变）
        df_interpolated.loc[index, row.index] = row.values
        
        # 填充缺失数据的位置为插值或外推结果
        missing_mask = row.isna()
        if missing_mask.any():
            # 计算缺失年份的插值结果，并更新缺失值
            missing_years = output_years[missing_mask.values]
            interpolated = linear_interp(missing_years)
            interpolated = np.maximum(interpolated, 0)  # 确保拟合结果大于等于0
            df_interpolated.loc[index, missing_years] = interpolated
            
# 输出结果到 Excel 文件
print("已完成")
df_interpolated.to_excel('data_interpolate.xlsx')
