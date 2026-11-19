import matplotlib.pyplot as plt
import numpy as np
import os
os.makedirs('figures', exist_ok=True)

# 数据准备
labels = ['AdamW', 'LoRA-Pre\nAdamW', 'Standard\nMuon', 'LoRA-Pre\nMuon']

arc_easy = [45.0, 44.0, 40.0, 36.0]
tqa_mc1  = [24.0, 25.0, 33.0, 30.0]
tqa_mc2  = [43.1, 43.1, 52.5, 51.2]

x = np.arange(len(labels))  # 标签位置
width = 0.25  # 柱子宽度

fig, ax = plt.subplots(figsize=(10, 6))

# 画三组柱子
rects1 = ax.bar(x - width, arc_easy, width, label='ARC-Easy (Common Sense)', color='#1f77b4')
rects2 = ax.bar(x, tqa_mc1, width, label='TruthfulQA MC1 (Strict)', color='#ff7f0e')
rects3 = ax.bar(x + width, tqa_mc2, width, label='TruthfulQA MC2 (Confidence)', color='#2ca02c')

# 添加标签和标题
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Fine-tuning Capability Comparison Across Optimizers', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.legend(fontsize=11)
ax.set_ylim(0, 65)

# 在柱子上标注数值
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()
plt.savefig('figures/finetune_accuracy_combined.png', dpi=300)
print("图已保存为 finetune_accuracy_combined.png")