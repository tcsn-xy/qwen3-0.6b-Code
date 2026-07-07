"""
合并 LoRA 权重到 base model 并保存完整模型
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel


def merge_lora(base_model_name="model_cache/Qwen/Qwen3-0.6B-Base",
               cpt_adapter="checkpoints/qwen-code-cpt-lora-v1",
               sft_adapter="checkpoints/qwen-code-sft-lora-v1",
               output_dir="checkpoints/qwen-code-merged"):
    print(f"加载 base model: {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 合并 CPT LoRA（如果存在）
    if os.path.exists(cpt_adapter):
        print(f"加载并合并 CPT LoRA: {cpt_adapter}")
        model = PeftModel.from_pretrained(model, cpt_adapter)
        model = model.merge_and_unload()

    # 合并 SFT LoRA（如果存在）
    if os.path.exists(sft_adapter):
        print(f"加载并合并 SFT LoRA: {sft_adapter}")
        model = PeftModel.from_pretrained(model, sft_adapter)
        model = model.merge_and_unload()

    # 保存
    print(f"保存合并后模型到: {output_dir}")
    model.save_pretrained(output_dir, safe_serialization=True)

    # 保存 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    # 保存 config
    config = AutoConfig.from_pretrained(base_model_name, trust_remote_code=True)
    config.save_pretrained(output_dir)

    print(f"[OK] Merge complete! Saved to: {output_dir}")


if __name__ == "__main__":
    merge_lora()
