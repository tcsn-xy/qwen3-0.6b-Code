"""
第五步：SFT 监督微调
在代码问答数据上用 LoRA 微调 Qwen3-0.6B-Base + CPT LoRA
"""

import os
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="model_cache/Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--cpt_adapter", type=str, default="checkpoints/qwen-code-cpt-lora-v1",
                        help="CPT LoRA adapter 路径，可选")
    parser.add_argument("--data_dir", type=str, default="data/sft")
    parser.add_argument("--output_dir", type=str, default="checkpoints/qwen-code-sft-lora-v1")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载 tokenizer
    print(f"加载 tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 加载模型
    print(f"加载模型: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 3. 可选：加载 CPT LoRA
    if args.cpt_adapter and os.path.exists(args.cpt_adapter):
        print(f"加载 CPT LoRA adapter: {args.cpt_adapter}")
        model = PeftModel.from_pretrained(model, args.cpt_adapter)
        model = model.merge_and_unload()  # 合并到 base 再新加 LoRA
        print("CPT LoRA 已合并到 base model")

    # 4. 配置 SFT LoRA
    print("配置 SFT LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. 加载 SFT 数据
    print(f"加载 SFT 数据: {args.data_dir}")
    train_path = os.path.join(args.data_dir, "train.jsonl")
    valid_path = os.path.join(args.data_dir, "valid.jsonl")

    data_files = {"train": train_path, "validation": valid_path}
    dataset = load_dataset("json", data_files=data_files, streaming=False)

    # 6. 格式化数据（ChatML）
    def format_chat(example):
        """将 messages 转为 ChatML 格式"""
        formatted_text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": formatted_text}

    print("格式化 ChatML 数据...")
    formatted_dataset = dataset.map(format_chat, remove_columns=["messages", "category"])

    def tokenize_function(examples):
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
        )
        return outputs

    print("Tokenizing...")
    tokenized_dataset = formatted_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )

    # 7. 训练配置
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        seed=args.seed,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # 8. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # 9. 训练
    print("开始 SFT 训练...")
    train_result = trainer.train()

    # 10. 保存
    trainer.save_model()
    trainer.save_state()
    print(f"SFT LoRA 保存到: {args.output_dir}")

    # 打印结果
    final_loss = train_result.training_loss
    eval_results = trainer.evaluate()
    print(f"最终训练 loss: {final_loss:.4f}")
    print(f"最终验证 loss: {eval_results['eval_loss']:.4f}")

    with open(os.path.join(args.output_dir, "training_results.json"), "w") as f:
        json.dump({
            "train_loss": final_loss,
            "eval_loss": eval_results["eval_loss"],
        }, f, indent=2)


if __name__ == "__main__":
    main()
