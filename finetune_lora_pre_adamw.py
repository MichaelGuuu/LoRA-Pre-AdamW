import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from lora_pre_adamw_optimizer import LoRAPreAdamW # 导入你写的优化器

torch.cuda.reset_peak_memory_stats() # 监控显存

model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

print("Loading dataset...")
dataset = load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations", split="train[:2000]")

def format_chat(example):
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}

dataset = dataset.map(format_chat)

training_args = SFTConfig(
    output_dir="./smollm2-finetuned-lora-pre", # 改了输出路径
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-5, # 微调学习率
    logging_steps=10,
    save_steps=250,
    save_total_limit=1,
    max_length=256,
    bf16=True,
    report_to="none",
)

# 实例化 LoRA-Pre-AdamW 优化器
print("Initializing LoRA-Pre-AdamW optimizer...")
lora_pre_optimizer = LoRAPreAdamW(
    model.parameters(), 
    lr=2e-5, 
    weight_decay=0.0, 
    rank=16
)

# 创建 Trainer，传入自定义优化器
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    optimizers=(lora_pre_optimizer, None), # 替换默认的 AdamW
)

print("Start Fine-tuning...")
trainer.train()
trainer.save_model("./smollm2-finetuned-lora-pre-final")

print(f"峰值显存占用: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
print("微调完成！")