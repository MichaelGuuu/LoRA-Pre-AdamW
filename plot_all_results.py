import matplotlib.pyplot as plt
import numpy as np
import os
os.makedirs('figures', exist_ok=True)

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('seaborn')
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

# ===== 数据准备 =====
opt_names = ['AdamW', 'LoRA-Pre\nAdamW', 'Muon', 'LoRA-Pre\nMuon']
colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c']

# 1. 预训练数据
pretrain_loss = [6.257, 6.614, 5.491, 5.975]
pretrain_mem = [3864.4, 3479.2, 3664.3, 3487.7]

# 2. 微调数据
ft_loss = [1.124, 1.030, 1.595, 1.572]
ft_mem = [1321.1, 1125.8, 1111.7, 934.6]

# ===================================
# 图 1 & 2：预训练对比
# ===================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.bar(opt_names, pretrain_loss, color=colors)
ax1.set_title('Pre-training train_loss (2000 Steps, run avg)')
ax1.set_ylim(5.0, 7.0)

ax2.bar(opt_names, pretrain_mem, color=colors)
ax2.set_title('Pre-training Peak Memory (MB)')
ax2.set_ylim(3000, 4000)
plt.tight_layout()
plt.savefig('figures/pretrain_family.png', dpi=300)

# ===================================
# 图 3 & 4：微调对比
# ===================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.bar(opt_names, ft_loss, color=colors)
ax1.set_title('Fine-tuning Loss (250 Steps)')

ax2.bar(opt_names, ft_mem, color=colors)
ax2.set_title('Fine-tuning Peak Memory (MB)')
plt.tight_layout()
plt.savefig('figures/finetune_family.png', dpi=300)

# ===================================
print("所有终极图表已生成！")