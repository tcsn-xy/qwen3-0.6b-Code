"""
第六步：评测脚本
全维度评测 Qwen3-0.6B-Base + SFT LoRA
"""

import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ========== 测试集 ==========
PASS_AT_1_TESTS = [
    {
        "prompt": "用 Python 写一个函数，判断一个字符串是否是回文",
        "check": lambda out: "def " in out and ("return" in out),
    },
    {
        "prompt": "用 Python 写一个计算阶乘的函数",
        "check": lambda out: "def " in out and "return" in out,
    },
    {
        "prompt": "写一个 SQL 查询，查询所有年龄大于 18 岁的用户",
        "check": lambda out: "SELECT" in out.upper() and "FROM" in out.upper(),
    },
    {
        "prompt": "用 Java 写一个 Hello World 程序",
        "check": lambda out: "public class" in out and "main" in out,
    },
]

BUG_FIX_TESTS = [
    {
        "code": "def add(a, b):\n    return a + b\n\nprint(add(1))\n\n这段代码有什么问题？",
        "check": lambda out: len(out) > 20,
    },
]

CODE_EXPLAIN_TESTS = [
    {
        "code": "def decorator(func):\n    @wraps(func)\n    def wrapper(*args, **kwargs):\n        return func(*args, **kwargs)\n    return wrapper",
        "question": "请解释这段代码的作用",
        "check": lambda out: len(out) > 30,
    },
]


def test_model(model, tokenizer, device):
    results = {
        "code_gen": {"pass": 0, "total": 0},
        "bug_fix": {"pass": 0, "total": 0},
        "explain": {"pass": 0, "total": 0},
    }

    print("\n" + "="*60)
    print("代码生成测试")
    print("="*60)
    for t in PASS_AT_1_TESTS:
        results["code_gen"]["total"] += 1
        prompt = t["prompt"]
        # Simple Q&A format instead of ChatML
        text = f"Q: {prompt}\nA:"
        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        passed = t["check"](response)
        if passed:
            results["code_gen"]["pass"] += 1
        status = "[PASS]" if passed else "[FAIL]"
        print(f"\n{status} Prompt: {prompt[:60]}...")
        safe_response = response[:100].encode('gbk', errors='replace').decode('gbk', errors='replace')
        print(f"   Response: {safe_response}...")

    print("\n" + "="*60)
    print("Bug 修复测试")
    print("="*60)
    for t in BUG_FIX_TESTS:
        results["bug_fix"]["total"] += 1
        text = f"Fix the bug:\n{t['code']}\n\nFixed version:"
        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        passed = t["check"](response)
        if passed:
            results["bug_fix"]["pass"] += 1
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} Response: {response[:150]}...")

    print("\n" + "="*60)
    print("代码解释测试")
    print("="*60)
    for t in CODE_EXPLAIN_TESTS:
        results["explain"]["total"] += 1
        text = f"Explain this code:\n{t['code']}\n\n{t['question']}\n\nExplanation:"
        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        passed = t["check"](response)
        if passed:
            results["explain"]["pass"] += 1
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} Response: {response[:150]}...")

    return results


def main():
    model_name = "model_cache/Qwen/Qwen3-0.6B-Base"
    sft_adapter = "checkpoints/qwen-code-sft-lora-v1"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    print(f"加载 tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载 SFT 模型
    print(f"加载 base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    if os.path.exists(sft_adapter):
        print(f"加载 SFT LoRA: {sft_adapter}")
        model = PeftModel.from_pretrained(model, sft_adapter)
    else:
        print("[WARN] SFT adapter not found, using base model")

    model.eval()

    results = test_model(model, tokenizer, device)

    # 计算结果
    pass_at_1 = results["code_gen"]["pass"] / max(results["code_gen"]["total"], 1)
    bug_fix_rate = results["bug_fix"]["pass"] / max(results["bug_fix"]["total"], 1)
    explain_score = results["explain"]["pass"] / max(results["explain"]["total"], 1)

    final = {
        "pass_at_1": pass_at_1,
        "bug_fix_rate": bug_fix_rate,
        "explain_score": explain_score,
        "pass_at_1_detail": results["code_gen"],
        "bug_fix_detail": results["bug_fix"],
        "explain_detail": results["explain"],
    }

    print("\n" + "="*60)
    print("评测结果汇总")
    print("="*60)
    print(f"Pass@1 (代码生成):  {pass_at_1:.0%}")
    print(f"Bug 修复准确率:     {bug_fix_rate:.0%}")
    print(f"代码解释得分:       {explain_score:.2f}")

    # 上线标准
    print("\n--- 上线判定 ---")
    checks = [
        ("Pass@1 > 60%", pass_at_1 > 0.6, f"{pass_at_1:.0%}"),
        ("Bug 修复 > 50%", bug_fix_rate > 0.5, f"{bug_fix_rate:.0%}"),
        ("解释得分 > 3.5/5", explain_score > 0.7, f"{explain_score:.2f}"),
    ]
    passed_checks = 0
    for name, ok, val in checks:
        status = "[PASS]" if ok else "[FAIL]"
        print(f"  {status} {name} (当前: {val})")
        if ok:
            passed_checks += 1

    final["上线"] = passed_checks >= 2
    print(f"\n  Passed {passed_checks}/3 -> {'[SUGGEST DEPLOY]' if passed_checks >= 2 else '[NEED IMPROVEMENT]'}")

    os.makedirs("eval", exist_ok=True)
    with open("eval/results.json", "w") as f:
        json.dump(final, f, indent=2)
    print(f"\n结果已保存到 eval/results.json")


if __name__ == "__main__":
    main()
