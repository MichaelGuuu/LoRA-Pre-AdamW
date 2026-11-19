import torch
torch.cuda.reset_peak_memory_stats()   
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from muon_optimizers import HybridLoRAPreMuon # 导入 LoRA-Pre-Muon

# 1. 设置
model_name = "HuggingFaceTB/SmolLM2-135M"
output_dir = "./smollm2-pretrained-lora-pre-muon-2000steps"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 2. 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 3. 从配置初始化模型
config = AutoConfig.from_pretrained(model_name)
model = AutoModelForCausalLM.from_config(config).to(device)
print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")

# 4. 流式加载数据集
dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text", "timestamp", "url"])

# 5. 数据整理器
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 6. 训练参数
training_args = TrainingArguments(
    output_dir=output_dir,
    max_steps=2000,                      
    per_device_train_batch_size=4,       
    gradient_accumulation_steps=4,
    learning_rate=0.005,                  
    warmup_steps=200,
    logging_steps=10,
    save_steps=2500,
    save_total_limit=1,
    bf16=True,                           
    report_to="none",
)

# 7. 实例化 HybridLoRAPreMuon
print("Initializing Hybrid LoRA-Pre-Muon optimizer...")
optimizer = HybridLoRAPreMuon(
    model, 
    lr=0.005, 
    momentum=0.95, 
    weight_decay=0.0,
    rank=16  # 保持 rank=16 与基准对齐
)

# 8. 创建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
    optimizers=(optimizer, None), 
)

# 9. 开始训练
print("开始从零预训练 (使用 LoRA-Pre-Muon)...")
trainer.train()
trainer.save_model(output_dir)
print(f"预训练完成！模型已保存到 {output_dir}")
print(f"峰值显存占用: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")