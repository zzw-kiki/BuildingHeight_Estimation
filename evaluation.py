import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

df = pd.read_csv("E:/evaluation/19_10万点.csv")  # 替换为你的文件路径
# 过滤掉包含NaN值的行
df_filtered = df.dropna(subset=['height', 'RASTERVALU'])
# 提取两列数据
observed_values = df_filtered['height']  # 替换为你的第一列名称
predicted_values = df_filtered['RASTERVALU']  # 替换为你的第二列名称

# 计算相关系数 R
correlation_coefficient, _ = pearsonr(observed_values, predicted_values)
# 输出结果
print(f"Correlation Coefficient R: {correlation_coefficient}")

absolute_errors = abs(observed_values - predicted_values)
# 计算 MAE
mae = absolute_errors.mean()
# 输出结果
print(f"Mean Absolute Error (MAE): {mae}")

r_squared = correlation_coefficient**2
print(f"R-squared (R²): {r_squared}")

rmse = np.sqrt(mean_squared_error(observed_values, predicted_values))

# 输出结果
print(f"RMSE: {rmse}")

plt.rcParams['font.family'] = 'Times New Roman'
plt.scatter(observed_values, predicted_values, alpha=0.5)
# plt.title('Scatter Plot of Observed vs Predicted Heights')
plt.xlabel('Reference Height(m)')
plt.ylabel('Predicted Height(m)')

# 最小二乘回归模型
regression_model = LinearRegression()
regression_model.fit(observed_values.values.reshape(-1, 1), predicted_values.values.reshape(-1, 1))
x_range = np.linspace(min(observed_values), max(observed_values), 100).reshape(-1, 1)
y_range = regression_model.predict(x_range)

# 绘制最小二乘回归线
plt.plot(x_range, y_range, color='red', linewidth=2)

# 绘制一对一的白线
plt.plot([min(observed_values), max(observed_values)], [min(observed_values), max(observed_values)], linestyle='--', color='black')

# 显示图例
legend = plt.legend(['Data Points','Regression Line', 'One-to-One Line'])
for text in legend.get_texts():
    text.set_fontweight('bold')

equation = f"y = {regression_model.coef_[0][0]:.2f}x + {regression_model.intercept_[0]:.2f}"
plt.text(min(observed_values), max(predicted_values) + 0.5, equation, color='black',weight='bold')
plt.text(min(observed_values), max(predicted_values) - 2.5, f"R= {correlation_coefficient:.2f}", color='black',weight='bold')
# plt.text(min(observed_values), max(predicted_values) + 100, f"R²= {r_squared:.2f}", color='black',weight='bold')
plt.text(min(observed_values), max(predicted_values) - 5, f"RMSE= {rmse:.2f} m", color='black',weight='bold')
# plt.savefig('E:/evaluation/图表/19_10万点.png', dpi=600, bbox_inches='tight')
# 显示图形
plt.show()


