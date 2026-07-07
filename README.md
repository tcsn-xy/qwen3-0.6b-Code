# Qwen3-0.6B-Code

基于 [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) 微调的代码增强模型。

## 训练流程

| 阶段 | 方法 | 数据 | Loss |
|------|------|------|------|
| CPT (继续预训练) | LoRA r=16 | 1.1万条代码 (Python/Java/SQL等) | 0.031 |
| SFT (监督微调) | LoRA r=16 | 116条代码问答对 | 1.211 |

## 评测结果

| 维度 | 得分 | 阈值 |
|------|------|------|
| Pass@1 代码生成 | 75% | >60% |
| Bug 修复 | 100% | >50% |
| 代码解释 | 100% | >70% |

> 3/3 通过

## 快速使用

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "checkpoints/qwen-code-merged",
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("checkpoints/qwen-code-merged")
tokenizer.pad_token = tokenizer.eos_token

# 问代码问题
prompt = "Q: 用Python写一个快速排序\nA:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 文件结构

```
├── checkpoints/
│   ├── qwen-code-merged/        # 完整合并模型
│   ├── qwen-code-cpt-lora-v1/   # CPT LoRA 适配器
│   └── qwen-code-sft-lora-v1/   # SFT LoRA 适配器
├── train_cpt.py                 # CPT 训练脚本
├── train_sft.py                 # SFT 训练脚本
├── evaluate.py                  # 评测脚本
├── merge_lora.py                # LoRA 合并脚本
├── inference.py                 # 交互推理
├── prepare_corpus.py            # 语料准备
├── prepare_sft_data.py          # SFT 数据准备
├── data/sft/                    # SFT 训练数据
├── eval/results.json            # 评测结果
└── requirements.txt             # 依赖
```

## 从 LoRA 重建模型

```bash
python merge_lora.py
# 自动加载 CPT + SFT LoRA 并融合为完整模型
```

## 硬件需求

- GPU: 4GB+ 显存（推理）
- CUDA: 12.1+
- 磁盘: ~1.5GB

## License

基于 Qwen3-0.6B-Base，遵循 Apache 2.0 License。
