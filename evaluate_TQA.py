import subprocess
import os

# 你要测试的 4 个模型路径列表
models = {
    "AdamW Baseline": "./smollm2-finetuned-final",
    "LoRA-Pre-AdamW": "./smollm2-finetuned-lora-pre-final",
    "Standard Muon": "./smollm2-finetuned-muon-final",
    "LoRA-Pre-Muon": "./smollm2-finetuned-lora-pre-muon-final"
}

# 检查文件夹是否存在
for name, path in models.items():
    if not os.path.exists(path):
        print(f"⚠️ 警告：找不到文件夹 {path}，请检查名字是否正确！")
        exit(1)

print("==================================================")
print("开始 ARC 评估矩阵，这可能需要一段时间...")
print("==================================================")

for name, path in models.items():
    print(f"\n正在评估: {name} (路径: {path})")
    print("-" * 50)
    
    # 构建 lm_eval 命令
    command = [
        "lm_eval", 
        "--model", "hf",
        "--model_args", f"pretrained={path}",
        "--tasks", "truthfulqa_mc1,truthfulqa_mc2",  # 同时跑两个指标
        "--device", "cuda:0",
        "--batch_size", "1",
        "--limit", "100",      # ARC 的题比较短，跑 100 道也就几分钟
        "--gen_kwargs", "max_gen_toks=10" # ARC 选 A/B/C/D，给 10 个 token 足够
    ]
    
    # 运行命令并等待完成
    result = subprocess.run(command, text=True)
    
    if result.returncode != 0:
        print(f"❌ {name} 评估失败，请检查报错信息。")
    else:
        print(f"✅ {name} 评估完成！请记录上面输出的 prompt_level_strict_acc 数值。")

print("\n🎉 全部评估结束！现在你可以把数据填入表格 C 了！")