import torch
torch.cuda.reset_peak_memory_stats()
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

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
    output_dir="./smollm2-finetuned",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    logging_steps=10,
    save_steps=200,
    max_length=256,
    bf16=True,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

print("Start training...")
trainer.train()
trainer.save_model("./smollm2-finetuned-final")
print(f"峰值显存占用: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
print(f"峰值显存（含缓存）: {torch.cuda.max_memory_reserved() / 1024**2:.1f} MB")
print("训练完成！模型已保存到 ./smollm2-finetuned-final")
