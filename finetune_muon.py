import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from muon_optimizers import HybridMuon # 导入你写的优化器

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
    output_dir="./smollm2-finetuned-muon", # 注意改了输出路径
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=0.005, # 标准muon是0.02，通常比 Adamw 大；我们先试0.005
    logging_steps=10,
    save_steps=250,
    save_total_limit=1,
    max_length=256,
    bf16=True,
    report_to="none",
)

# 实例化 HybridMuon 优化器
print("Initializing Hybrid Muon optimizer...")
optimizer = HybridMuon(
    model, 
    lr=0.005, 
    momentum=0.95, 
    weight_decay=0.0
)

# 创建 Trainer，传入自定义优化器
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    optimizers=(optimizer, None), # 替换默认的 AdamW
)

print("Start Fine-tuning...")
trainer.train()
trainer.save_model("./smollm2-finetuned-muon-final")

print(f"峰值显存占用: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
print("微调完成！")