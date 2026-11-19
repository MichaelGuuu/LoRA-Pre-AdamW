import matplotlib.pyplot as plt
import numpy as np

# 设置画图风格
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('seaborn')

# 1. 提取数据（从你发我的 002.txt 和 AdamW 日志中提取的全局最小值）
steps = [50, 100, 150, 200, 250]

# Muon 训练时的最小特征值
muon_min_eigs = [0.000002, 0.000001, 0.000001, 0.000001, 0.000001]

# AdamW 训练时的最小特征值
adamw_min_eigs = [0.000002, 0.000002, 0.000003, 0.000002, 0.000001]

plt.figure(figsize=(10, 6))

# 2. 画图
plt.plot(steps, muon_min_eigs, marker='o', markersize=10, linewidth=3, 
         color='#d62728', label='LoRA-Pre-Muon (Min Eigenvalue)')
plt.plot(steps, adamw_min_eigs, marker='s', markersize=10, linewidth=3, 
         color='#1f77b4', label='LoRA-Pre-AdamW (Min Eigenvalue)')

# 3. 标出安全底线 (eps = 1e-6)
plt.axhline(y=1e-6, color='black', linestyle='--', linewidth=2, label='Safety Floor (\(\lambda = 10^{-6}\))')

# 4. 加上注解
plt.text(130, 1.5e-6, "Without + λI, values drop to 0\n-> Matrix Singular -> NaN Loss", 
         fontsize=12, color='black', fontweight='bold',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

# 5. 设置坐标轴
plt.yscale('log') # 使用对数坐标轴，让 \(10^{-6}\) 看起来更直观
plt.ylim(5e-7, 1e-5)
plt.title('Tracking the Minimum Eigenvalue of $M_A$ During Fine-tuning', fontsize=16)
plt.xlabel('Training Steps', fontsize=14)
plt.ylabel('Global Min Eigenvalue (Log Scale)', fontsize=14)
plt.legend(fontsize=12, loc='upper right')
plt.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.savefig('eigenvalue_stability.png', dpi=300)
print("特征值稳定性图已保存为 eigenvalue_stability.png")