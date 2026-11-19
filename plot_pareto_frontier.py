import matplotlib.pyplot as plt
import os
os.makedirs('figures', exist_ok=True)

data = {
    'AdamW': (1321.1, 45.0),
    'LoRA-Pre-AdamW': (1125.8, 44.0),
    'Muon': (1111.7, 40.0),
    'LoRA-Pre-Muon': (934.6, 36.0)
}
colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c']

plt.figure(figsize=(9, 6))

# 1. 先画 Win Zone (背景)
# 稍微加深的浅蓝色: #d4e6f1 或 #cce5ff
plt.axvspan(850, 1150, ymin=0.59, ymax=1.0, color='#cce5ff', alpha=0.6, zorder=1)
plt.text(1000, 48, "Win Zone\n(Pareto Optimal)", ha='center', fontsize=14, color='#1f77b4', fontweight='bold', style='italic', zorder=4)

# 2. 再画散点图 (前景)
for i, (name, (x, y)) in enumerate(data.items()):
    # zorder=3 确保点在 zone 上面
    plt.scatter(x, y, s=800, color=colors[i], alpha=0.9, edgecolors='black', linewidth=1.5, zorder=3)
    plt.annotate(name, (x, y), textcoords="offset points", xytext=(0, 20), ha='center', fontsize=12, fontweight='bold', zorder=4)

# 设置坐标轴
plt.title('The Pareto Frontier: Capability vs. Memory (Fine-tuning)', fontsize=15, pad=15)
plt.xlabel('Peak Memory Usage (MB) ↓', fontsize=13)
plt.ylabel('ARC-Easy Accuracy (%) ↑', fontsize=13)
plt.xlim(850, 1450)
plt.ylim(30, 52)
plt.grid(True, linestyle='--', alpha=0.6, zorder=0)

# 3. 修改文字颜色为红色
plt.text(1300, 32, "Lower Memory\n& Higher Accuracy\nis Better", fontsize=12, color='red', fontweight='bold', ha='center')

plt.tight_layout()
plt.savefig('figures/pareto_frontier.png', dpi=300)
print("图已保存为 pareto_frontier.png")