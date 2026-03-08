import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# 数据
G_values = ['None', '2', '4', '8', '16', '32']
accuracy  = [95.50, 96.25, 96.13, 97.25, 96.50, 96.30]
precision = [95.50, 96.29, 96.11, 97.26, 96.52, 96.28]
recall    = [95.50, 96.25, 96.13, 97.25, 96.50, 96.30]
f1        = [95.49, 96.25, 96.08, 97.25, 96.50, 96.31]

x = np.arange(len(G_values))

# 增加图形高度，使纵轴拉长
fig, ax = plt.subplots(figsize=(9, 6))

# 使用不同线型和标记
ax.plot(x, accuracy,  marker='o', linestyle='-',  linewidth=2, label='Accuracy')
ax.plot(x, precision, marker='s', linestyle='--', linewidth=2, label='Precision')
ax.plot(x, recall,    marker='^', linestyle=':',  linewidth=2, label='Recall')
ax.plot(x, f1,        marker='D', linestyle='-.', linewidth=2, label='F1-score')

# 设置坐标轴：进一步缩小y轴范围，放大局部差异
ax.set_xlabel('Group Number G')
ax.set_ylabel('Percentage (%)')
ax.set_title('Effect of Group Number G in EMA Module', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(G_values)
ax.set_ylim(95.0, 97.5)   # 更紧凑的范围
ax.legend(loc='lower right', frameon=True)

# 可选：在最大值点添加标注
max_idx = 3  # G=8
ax.annotate(f'Best: {accuracy[max_idx]:.2f}%',
            xy=(x[max_idx], accuracy[max_idx]), xytext=(x[max_idx]+0.5, accuracy[max_idx]-0.2),
            arrowprops=dict(arrowstyle='->', color='gray'), fontsize=10)

plt.tight_layout()
plt.savefig('ema_group_line_zoomed.png', dpi=600)
plt.show()