import matplotlib.pyplot as plt

# 秩消融实验数据
ranks = [16, 32, 64]
losses = [6.614, 6.478, 6.350]

plt.figure(figsize=(7, 5))
plt.plot(ranks, losses, marker='o', markersize=10, linewidth=3, color='#2ca02c')
plt.title('Rank Efficiency: LoRA-Pre-AdamW (2000 Steps)')
plt.xlabel('Rank Size (r)')
plt.ylabel('Final Training Loss')
plt.xticks(ranks)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('rank_efficiency.png', dpi=300)
print("图已保存为 rank_efficiency.png")