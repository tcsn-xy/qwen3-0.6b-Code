"""
全流程一键训练脚本：CPT → SFT → 评测 → 合并
运行: python run_all.py 2>&1 | tee train.log
"""
import subprocess
import sys
import os
import time
import json
from datetime import datetime

STEPS = [
    {
        "name": "CPT 继续预训练",
        "cmd": [sys.executable, "train_cpt.py"],
        "checkpoint": "checkpoints/qwen-code-cpt-lora-v1",
        "check_file": "adapter_model.safetensors",
    },
    {
        "name": "SFT 监督微调",
        "cmd": [sys.executable, "train_sft.py"],
        "checkpoint": "checkpoints/qwen-code-sft-lora-v1",
        "check_file": "adapter_model.safetensors",
    },
    {
        "name": "评测",
        "cmd": [sys.executable, "evaluate.py"],
        "checkpoint": "eval",
        "check_file": "results.json",
    },
    {
        "name": "合并模型",
        "cmd": [sys.executable, "merge_lora.py"],
        "checkpoint": "checkpoints/qwen-code-merged",
        "check_file": "config.json",
    },
]

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)

def run_step(step):
    log(f"{'='*60}")
    log(f"开始: {step['name']}")
    log(f"{'='*60}")
    start = time.time()
    result = subprocess.run(step["cmd"], cwd=os.path.dirname(os.path.abspath(__file__)))
    elapsed = time.time() - start
    if result.returncode != 0:
        log(f"[FAIL] {step['name']} failed! Code: {result.returncode}, Time: {elapsed:.0f}s")
        return False
    # 验证产出
    check_path = os.path.join(step["checkpoint"], step["check_file"])
    if os.path.exists(check_path):
        log(f"[OK] {step['name']} done! Time: {elapsed:.0f}s -> {check_path}")
        return True
    else:
        log(f"[WARN] {step['name']} ran OK but no output: {check_path}")
        return False

def main():
    log("全流程开始")
    log(f"Python: {sys.executable}")
    total_start = time.time()
    for step in STEPS:
        if not run_step(step):
            log(f"流程中断于: {step['name']}")
            sys.exit(1)
    total_elapsed = time.time() - total_start
    log(f"{'='*60}")
    log(f"[DONE] All steps complete! Total: {total_elapsed/60:.1f} min")
    log(f"{'='*60}")

if __name__ == "__main__":
    main()
