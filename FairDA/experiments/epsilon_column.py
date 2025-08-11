import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
# 设置数据
metric = ("$\epsilon=0.1$", "$\epsilon=0.01$", "$\epsilon=0.001$")
values = {
    "Acc.": (0.774, 0.781, 0.782),
    "EO": (0.087, 0.027, 0.024),
    "DP": (0.069, 0.034, 0.013),
}
x = np.arange(len(metric))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax1 = plt.subplots(figsize=(10, 3))
# ax2 = ax1.twinx()  # 创建共享x轴的第二个坐标轴

for attribute, measurement in values.items():
    offset = width * multiplier
    rects = ax1.bar(x + offset, measurement, width, label=attribute)
    ax1.bar_label(rects, padding=3)
    multiplier += 1

# 设置坐标轴的标题和刻度

# ax1.set_ylabel("Acc.")  # 左侧y轴标题
# ax2.set_ylabel("EO and DP")  # 右侧y轴标题
ax1.set_xticks(x + width, metric)
ax1.set_ylim(0, 1)
# ax2.set_ylim(0, 0.1)
# 设置图例和标题
fig.legend(loc="upper center", ncol=3)  # 图例位置和列数

# 显示图形
plt.savefig("fig/epsilon_column.jpg")
plt.savefig("fig/epsilon_column.svg", format="svg")
print("end")
