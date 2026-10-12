import matplotlib.pyplot as plt
import numpy as np

# 设置画图风格 (兼容不同版本matplotlib)
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('seaborn')

# ==================== 数据准备 (严格对照日志) ====================
# 1. 2000步的数据点 (每200步取一个点，对应 epoch 0.1 ~ 1.0)
steps_2000 = list(range(0, 2001, 200))
adamw_2000 = [11.29, 7.132, 6.473, 6.292, 6.155, 5.955, 5.860, 5.839, 5.769, 5.676, 5.737]
lora_2000  = [11.29, 6.641, 6.573, 6.500, 6.438, 6.290, 6.320, 6.313, 6.226, 6.176, 6.299]

# 2. 5000步的数据点 (每500步取一个点，对应 epoch 0.1 ~ 1.0)
steps_5000 = list(range(0, 5001, 500))
adamw_5000 = [11.29, 6.484, 5.991, 5.699, 5.552, 5.340, 5.261, 5.207, 5.149, 5.148, 5.276]
lora_5000  = [11.29, 6.430, 6.340, 6.300, 6.250, 6.160, 6.100, 6.080, 6.090, 6.090, 6.156]

# ==================== 图1：Loss vs Steps 对比图 ====================
plt.figure(figsize=(10, 6))

plt.plot(steps_2000, adamw_2000, label='AdamW-2000 Steps', marker='o', color='#1f77b4', linewidth=2, alpha=0.6)
plt.plot(steps_2000, lora_2000, label='LoRA-Pre-2000 Steps', marker='s', color='#ff7f0e', linewidth=2, alpha=0.6)

plt.plot(steps_5000, adamw_5000, label='AdamW-5000 Steps', marker='d', color='#d62728', linewidth=2)
plt.plot(steps_5000, lora_5000, label='LoRA-Pre-5000 Steps', marker='^', color='#2ca02c', linewidth=2)

plt.title('Training Loss vs. Steps (Pre-training SmolLM2-135M)', fontsize=14)
plt.xlabel('Training Steps', fontsize=12)
plt.ylabel('Training Loss', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.7)

# 保存图片
plt.tight_layout()
plt.savefig('loss_comparison.png', dpi=300)
print("图1已保存为 loss_comparison.png")

# ==================== 图2：峰值显存柱状图 ====================
plt.figure(figsize=(8, 5))

labels = ['AdamW\n(2000)', 'LoRA-Pre\n(2000)', 'AdamW\n(5000)', 'LoRA-Pre\n(5000)']
memory_usage = [3864.4, 3479.2, 3864.4, 3439.7] 
# 修正：4个柱子需要4种颜色
colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c'] 

bars = plt.bar(labels, memory_usage, color=colors, width=0.5)

# 在柱状图上方标注具体数值
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 50, f'{yval} MB', ha='center', va='bottom', fontsize=11)

plt.title('Peak GPU Memory Usage (Allocated)', fontsize=14)
plt.xlabel('Optimizer & Training Steps', fontsize=12)
plt.ylabel('Peak Memory (MB)', fontsize=12)
plt.ylim(0, 4500) 

# 保存图片
plt.tight_layout()
plt.savefig('memory_comparison.png', dpi=300)
print("图2已保存为 memory_comparison.png")