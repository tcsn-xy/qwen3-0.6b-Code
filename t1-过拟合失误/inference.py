"""
交互式推理脚本
支持 Base / CPT / SFT 三种模式
用法: python inference.py [--mode sft|cpt|base] [--max-tokens 1024] [--temp 0.7]
"""

import os
import sys
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel


class StopOnQuestion(StoppingCriteria):
    \"\"\"当模型开始自己编下一个「问题：」时停止生成\"\"\"
    def __init__(self, tokenizer, prompt_tokens, stop_patterns=None):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_tokens
        self.stop_patterns = stop_patterns or ["问题：", "\n问题", "Q:", "Question:"]

    def __call__(self, input_ids, scores, **kwargs):
        # 只检查新生成的部分
        new_ids = input_ids[0][self.prompt_len:]
        if len(new_ids) < 3:
            return False
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return any(p in text for p in self.stop_patterns)


def load_model(model_type="sft"):
    base_model = "Qwen/Qwen3-0.6B-Base"
    merged_model = "checkpoints/qwen-code-merged"
    cpt_adapter = "checkpoints/qwen-code-cpt-lora-v1"

    # SFT 模式直接用合并好的完整模型（base + CPT + SFT 已合并）
    if model_type == "sft" and os.path.exists(merged_model):
        print(f"加载合并模型: {merged_model}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            merged_model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(merged_model, trust_remote_code=True)

    # CPT 模式：base + CPT LoRA
    elif model_type == "cpt" and os.path.exists(cpt_adapter):
        print(f"加载 base model: {base_model}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        print(f"加载 CPT LoRA: {cpt_adapter}", flush=True)
        model = PeftModel.from_pretrained(model, cpt_adapter)

    # Base 模式：仅加载原始模型
    else:
        print(f"加载 base model: {base_model}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


def chat(args):
    print("Qwen3-0.6B-Code 交互式推理", flush=True)
    print("=" * 50, flush=True)
    print(f"模式: {args.mode}  |  max_tokens: {args.max_tokens}  |  temperature: {args.temp}", flush=True)

    model, tokenizer = load_model(args.mode)
    device = next(model.parameters()).device

    print(f"设备: {device}", flush=True)
    print("输入 /exit 退出，/clear 清屏", flush=True)

    while True:
        try:
            sys.stdout.write("\n>>> ")
            sys.stdout.flush()
            user_input = input()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！", flush=True)
            break

        if user_input.lower() in ["/exit", "/quit", "exit"]:
            break
        if user_input.lower() == "/clear":
            os.system('cls' if os.name == 'nt' else 'clear')
            continue
        if not user_input.strip():
            continue

        # 用简洁 prompt 格式（chat template 在 116 条 SFT 数据下没学好）
        text = f"问题：{user_input}\n回答："
        inputs = tokenizer(text, return_tensors="pt").to(device)

        try:
            # 带停止策略的流式输出
            sys.stdout.write("\nAssistant: ")
            sys.stdout.flush()
            stop_criteria = StoppingCriteriaList([StopOnQuestion(tokenizer, inputs.input_ids.shape[1])])
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            with torch.no_grad():
                model.generate(
                    **inputs,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temp,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    stopping_criteria=stop_criteria,
                    streamer=streamer,
                )
            sys.stdout.write("\n")
            sys.stdout.flush()
        except Exception as e:
            print(f"\n[生成出错: {e}]", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-0.6B-Code 交互式推理")
    parser.add_argument("--mode", default="sft", choices=["base", "cpt", "sft"],
                        help="模型模式 (default: sft)")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="最大生成 token 数 (default: 512)")
    parser.add_argument("--temp", type=float, default=0.3,
                        help="采样温度，越低越聚焦 (default: 0.3)")
    args = parser.parse_args()
    chat(args)
