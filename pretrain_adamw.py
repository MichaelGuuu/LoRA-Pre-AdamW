import torch
torch.cuda.reset_peak_memory_stats()
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import sys

# ---- step budget from CLI:  python pretrain_adamw.py 2000 | 5000 ----
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
if STEPS not in (2000, 5000):
    raise SystemExit("usage: python pretrain_adamw.py [2000|5000]")
WARMUP = 200 if STEPS <= 2000 else 500
SAVE_EVERY = 1000 if STEPS <= 2000 else 2500

# 1. 设置
model_name = "HuggingFaceTB/SmolLM2-135M"
output_dir = f"./smollm2-pretrained-adamw-{STEPS}steps"     # STEPS = 2000步；5000步
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 2. 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 3. 从配置初始化模型（随机权重！这是从头预训练的关键）
config = AutoConfig.from_pretrained(model_name)
model = AutoModelForCausalLM.from_config(config).to(device)
print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")

# 4. 流式加载数据集（避免占满硬盘）
dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text", "timestamp", "url"])

# 5. 数据整理器
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 6. 训练参数
training_args = TrainingArguments(
    output_dir=output_dir,
    max_steps=STEPS,                    # 来自命令行；默认 5000
    per_device_train_batch_size=4,      # 6GB显存，设4比较安全
    gradient_accumulation_steps=4,
    learning_rate=5e-4,                 # 预训练学习率
    warmup_steps=WARMUP,                # 200步对应2000步，500步对应5000步
    logging_steps=10,
    save_steps=SAVE_EVERY,              # 1000只在2000步时保存一次检查点；2500只在5000步时保存一次
    save_total_limit=1,                 # 只保留最近 1 个检查点，旧的自动删除
    bf16=True,                          # 开启BF16节省显存
    report_to="none",
)

# 7. 创建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
)

# 8. 开始训练
print("开始从零预训练...")
trainer.train()
trainer.save_model(output_dir)
print(f"预训练完成！模型已保存到 {output_dir}")
print(f"峰值显存占用: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
print(f"峰值显存（含缓存）: {torch.cuda.max_memory_reserved() / 1024**2:.1f} MB")