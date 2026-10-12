import torch
torch.cuda.reset_peak_memory_stats()   
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# 1. 设置
model_name = "HuggingFaceTB/SmolLM2-135M"
output_dir = "./smollm2-pretrained-lora-pre-adamw-5000steps"
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
    max_steps=5000,                      # 再跑5000步看看效果
    per_device_train_batch_size=4,       # 6GB显存，设4比较安全
    gradient_accumulation_steps=4,
    learning_rate=5e-4,                  # 预训练学习率
    warmup_steps=500,
    logging_steps=10,
    save_steps=2500,
    save_total_limit=1,    # 只保留最近 1 个检查点，旧的自动删除
    bf16=True,                           # 开启BF16节省显存
    report_to="none",
)

# 7. 导入并实例化 LoRA-Pre 优化器
from lora_pre_adamw_optimizer import LoRAPreAdamW

lora_pre_adamw_optimizer = LoRAPreAdamW(
    model.parameters(), 
    lr=5e-4,           # 和原来预训练保持一致
    weight_decay=0.0,  # 和原来保持一致
    rank=16            # 压缩秩，先设16，以后可以调
)

# 8. 创建 Trainer，通过 optimizers 参数传入自定义优化器
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
    optimizers=(lora_pre_adamw_optimizer, None), # 第二个参数传 None 表示用默认学习率调度
)

# 9. 开始训练
print("开始从零预训练 (使用 LoRA-Pre)...")
trainer.train()
trainer.save_model(output_dir)
print(f"预训练完成！模型已保存到 {output_dir}")
print(f"峰值显存占用: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
print(f"峰值显存（含缓存）: {torch.cuda.max_memory_reserved() / 1024**2:.1f} MB")