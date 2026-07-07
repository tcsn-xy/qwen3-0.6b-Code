"""快速测试训练流程是否可运行"""
import time, sys, os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

print("Stage 0: Checking CUDA...", flush=True)
print(f"  CUDA available: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}", flush=True)

print("Stage 1: Loading tokenizer...", flush=True)
tok = AutoTokenizer.from_pretrained("model_cache/Qwen/Qwen3-0.6B-Base", trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
print("  Tokenizer loaded", flush=True)

print("Stage 2: Loading model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    "model_cache/Qwen/Qwen3-0.6B-Base",
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
print("  Model loaded", flush=True)

print("Stage 3: Adding LoRA...", flush=True)
lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("Stage 4: Loading small dataset...", flush=True)
# 只用10条数据快速验证
with open("data/corpus/train.jsonl", "r", encoding="utf-8") as f:
    lines = [next(f) for _ in range(10)]
with open("data/corpus/test_small.jsonl", "w", encoding="utf-8") as f:
    for l in lines:
        f.write(l)
with open("data/corpus/valid_small.jsonl", "w", encoding="utf-8") as f:
    f.write(lines[0])

ds = load_dataset("json", data_files={
    "train": "data/corpus/test_small.jsonl",
    "validation": "data/corpus/valid_small.jsonl",
})
print(f"  Loaded {len(ds['train'])} train, {len(ds['validation'])} valid", flush=True)

print("Stage 5: Tokenizing...", flush=True)
def tok_fn(ex):
    return tok(ex["text"], truncation=True, max_length=2048, padding=False)

tok_ds = ds.map(tok_fn, batched=True, remove_columns=["text"])
print(f"  Tokenized: {len(tok_ds['train'])}", flush=True)

print("Stage 6: Running 1 training step...", flush=True)
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

args = TrainingArguments(
    output_dir="checkpoints/test_cpt",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=1,
    logging_steps=1,
    bf16=True,
    max_steps=2,  # 只跑2步
    report_to=[],
    remove_unused_columns=False,
    ddp_find_unused_parameters=False,
    gradient_checkpointing=True,
    seed=42,
)

collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
trainer = Trainer(
    model=model, args=args,
    train_dataset=tok_ds["train"],
    eval_dataset=tok_ds["validation"],
    processing_class=tok,
    data_collator=collator,
)

print("  Training...", flush=True)
trainer.train()
print("  Training completed!", flush=True)

# 清理测试文件
os.remove("data/corpus/test_small.jsonl")
os.remove("data/corpus/valid_small.jsonl")
print("SUCCESS: Training pipeline works!", flush=True)
