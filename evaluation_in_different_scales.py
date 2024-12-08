import pandas as pd
import numpy as np
import os
import re
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

# 指定目录路径
directory_path = "E:/evaluation/BHNet/19_10分街道数据"

# 创建一个空的 DataFrame 用于存储结果
results = pd.DataFrame(columns=["县级", "MAE", "RMSE", "num_samplepoint"])

# 遍历目录中的所有 CSV 文件
for filename in os.listdir(directory_path):
    if filename.endswith(".csv"):
        # 构建文件路径
        file_path = os.path.join(directory_path, filename)

        # 读取 CSV 文件
        df = pd.read_csv(file_path)
        # 过滤掉包含NaN值的行
        df_filtered = df.dropna(subset=['height', 'RASTERVALU_Gee'])
        # 提取两列数据
        observed_values = df_filtered['height']  # 替换为你的第一列名称
        predicted_values = df_filtered['RASTERVALU_Gee']  # 替换为你的第二列名称

        # 检查数据长度
        if len(observed_values) >= 2 and len(predicted_values) >= 2:
            # 检查是否为常数数组
            if observed_values.nunique() == 1 or predicted_values.nunique() == 1:
                correlation_coefficient = np.nan
                r_squared = np.nan
            else:
                # 计算相关系数 R
                correlation_coefficient, _ = pearsonr(observed_values, predicted_values)
                # 计算 R²
                r_squared = correlation_coefficient ** 2

            # 计算 MAE
            absolute_errors = abs(observed_values - predicted_values)
            mae = absolute_errors.mean()

            # 计算 RMSE
            rmse = np.sqrt(mean_squared_error(observed_values, predicted_values))

            # 统计参与 RMSE 和 MAE 计算的数据点数量
            num_samplepoint = len(observed_values)
        else:
            # 如果数据长度不足，设置相应的值为 NaN
            correlation_coefficient = np.nan
            mae = np.nan
            r_squared = np.nan
            rmse = np.nan
            num_samplepoint = 0

        # 从文件名中提取中文字符
        chinese_characters = re.findall(r'[\u4e00-\u9fff]+', filename)
        chinese_name = ''.join(chinese_characters) if chinese_characters else filename

        # 创建一个新的 DataFrame 包含当前文件的结果
        new_row = pd.DataFrame({
            "县级": [chinese_name],
            "MAE": [mae],
            # "R": [correlation_coefficient],
            # "R²": [r_squared],
            "RMSE": [rmse],
            "num_samplepoint": [num_samplepoint]
        })

        # 如果 new_row 不为空且不全为 NaN，才添加到 results 中
        if not new_row.empty and not new_row.isna().all().all():
            results = pd.concat([results, new_row], ignore_index=True)

# 输出最终的表到指定路径
output_file = os.path.join(directory_path, "summary_results_19_10%.csv")
results.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"Results have been saved to {output_file}")
