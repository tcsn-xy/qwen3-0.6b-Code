"""
交互式推理脚本
支持 Base / CPT / SFT 三种模式
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_model(model_type="sft"):
    base_model = "Qwen/Qwen3-0.6B-Base"
    cpt_adapter = "checkpoints/qwen-code-cpt-lora-v1"
    sft_adapter = "checkpoints/qwen-code-sft-lora-v1"

    print(f"加载 base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_type == "cpt" and os.path.exists(cpt_adapter):
        print(f"加载 CPT LoRA: {cpt_adapter}")
        model = PeftModel.from_pretrained(model, cpt_adapter)
    elif model_type == "sft" and os.path.exists(sft_adapter):
        print(f"加载 SFT LoRA: {sft_adapter}")
        model = PeftModel.from_pretrained(model, sft_adapter)
    else:
        print("使用 base model（未加载 LoRA）")

    model.eval()
    return model, tokenizer


def chat():
    print("Qwen3-0.6B-Code 交互式推理")
    print("=" * 50)
    print("选择模式: base / cpt / sft (默认: sft)")
    model_type = input("模式: ").strip() or "sft"

    model, tokenizer = load_model(model_type)
    device = next(model.parameters()).device

    print("\n输入 /exit 退出，/clear 清屏\n")

    while True:
        user_input = input("\n>>> ")
        if user_input.lower() in ["/exit", "/quit", "exit"]:
            break
        if user_input.lower() == "/clear":
            print("\033[2J\033[H", end="")
            continue

        messages = [{"role": "user", "content": user_input}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )

        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        print(f"\nAssistant: {response}")


if __name__ == "__main__":
    chat()
