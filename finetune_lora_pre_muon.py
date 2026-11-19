import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from muon_optimizers import HybridLoRAPreMuon

torch.cuda.reset_peak_memory_stats()

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
    output_dir="./smollm2-finetuned-lora-pre-muon", # 改输出路径
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=0.005, 
    logging_steps=10,
    save_steps=250,
    save_total_limit=1,
    max_length=256,
    bf16=True,
    report_to="none",
)

print("Initializing Hybrid LoRA-Pre-Muon optimizer...")
optimizer = HybridLoRAPreMuon( # 实例化 LoRA-Pre-Muon
    model, 
    lr=0.005, 
    momentum=0.95, 
    weight_decay=0.0,
    rank=16 # 加入 rank 参数
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    optimizers=(optimizer, None),
)

print("Start Fine-tuning with LoRA-Pre-Muon...")
trainer.train()
trainer.save_model("./smollm2-finetuned-lora-pre-muon-final") # 保存路径

print(f"峰值显存占用: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
print("微调完成！")