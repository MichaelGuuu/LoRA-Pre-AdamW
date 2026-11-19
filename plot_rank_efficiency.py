import matplotlib.pyplot as plt
import os
os.makedirs('figures', exist_ok=True)

# 秩消融实验数据
ranks = [16, 32, 64]
losses = [6.614, 6.478, 6.350]

plt.figure(figsize=(7, 5))
plt.plot(ranks, losses, marker='o', markersize=10, linewidth=3, color='#2ca02c',
         label='LoRA-Pre-AdamW (r=16 = default)')
plt.axhline(6.257, linestyle='--', linewidth=1.8, color='#1f77b4',
            label='AdamW full-rank, run avg (6.257)')
plt.title('Rank Efficiency: LoRA-Pre-AdamW (2000 Steps, run avg)')
plt.xlabel('Rank Size (r)')
plt.ylabel('Run-average Training Loss')        
plt.xticks(ranks)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=10, loc='upper right')
plt.tight_layout()
plt.savefig('figures/rank_efficiency.png', dpi=300)
print("图已保存为 rank_efficiency.png")