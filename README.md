# RL Failure Modes Tutorial

一个“从错误中学习 RL”的可复现教程。项目不只展示 DPO/RL 能否跑通，
而是把典型失败模式做成可重复的实验：在固定模型、数据和评测下，故意只
破坏一个变量，并用独立 benchmark 验证模型究竟发生了什么变化。

所有实验常量定义于
[`configs/experiment_constants.yaml`](configs/experiment_constants.yaml)。后续 failure
scenario 只能修改自己的故障注入配置，不能暗中改变基线常量。

## 学习闭环

```text
冻结实验常量 → 原始模型 benchmark → 正常 DPO baseline
      → 注入一个 failure → 观察训练/外部指标 → 解释并修复
```

第一批 failure mode 将覆盖：反转偏好标签、reward/parser hacking、长度偏置、
偏好噪声、低多样性 rollout、截断与过大学习率。

## 当前顺序

1. 冻结模型、数据、随机种子、DPO 超参数、评测子集与解码配置。
2. 对原始 `Qwen2.5-1.5B-Instruct` 运行 UltraFeedback held-out、IFEval、GSM8K 与平均回复长度，保存基线。
3. 实现正常 DPO baseline；其余 failure mode 均与这两个对照比较。

## 固定评测集

| 指标 | 数据 | 固定范围 |
| --- | --- | --- |
| ID preference diagnostic | UltraFeedback test | 全量 1,000 条 |
| instruction following | IFEval | 全量 |
| regression guard | GSM8K main test | seed=42 固定抽样 250 条 |
| generation behavior | 所有生成评测 | 贪心解码、最多 512 tokens |

基线与后续实验均只在启动前确认没有计算进程的单张 GPU 上运行；当前固定使用远程 GPU 2。

## 原始模型基线

已在固定配置下评测 `Qwen2.5-1.5B-Instruct`。这些数值是后续正常 DPO 与
failure mode 的对照点，而非通过训练得到的结果。

| 指标 | 基线结果 |
| --- | ---: |
| UltraFeedback held-out preference accuracy | 57.6%（1,000 对） |
| IFEval strict | 39.93% |
| GSM8K accuracy | 66.4%（固定 250 条） |
| 平均回复长度 | 166.0 词 |

## 环境

实验使用 Python 3.10 与 CUDA GPU。建议为项目创建独立 Conda 环境：

```bash
conda create -y -p /workspace/zzc/envs/rl-failures python=3.10
conda activate /workspace/zzc/envs/rl-failures
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install transformers datasets accelerate pyyaml sentencepiece \
  huggingface_hub absl-py langdetect nltk
```

IFEval 的 strict evaluator 来自 Google Research：

```bash
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/google-research/google-research.git third_party/google-research
git -C third_party/google-research sparse-checkout set instruction_following_eval
python -m nltk.downloader punkt_tab
```

## 模型与数据下载

本项目约定以下远程目录；模型和数据均被 `.gitignore` 排除，不能提交到 GitHub。

```text
/workspace/zzc/
├── models/Qwen2.5-1.5B-Instruct/
├── rl-failures/data/
└── envs/rl-failures/
```

以下命令使用 Hugging Face 镜像站；`HF_HUB_DISABLE_XET=1` 可避免部分环境
在 Xet 下载链路上认证失败。

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

# Model
hf download Qwen/Qwen2.5-1.5B-Instruct \
  --local-dir /workspace/zzc/models/Qwen2.5-1.5B-Instruct

# DPO train set and in-distribution diagnostic
hf download trl-lib/ultrafeedback_binarized --repo-type dataset \
  --local-dir /workspace/zzc/rl-failures/data/ultrafeedback_binarized

# Primary clean evaluation
hf download google/IFEval --repo-type dataset \
  --local-dir /workspace/zzc/rl-failures/data/ifeval

# Regression guard; use the `main` configuration
hf download openai/gsm8k --repo-type dataset \
  --local-dir /workspace/zzc/rl-failures/data/gsm8k
```

## 运行基线

先确认所选 GPU 没有其他计算进程，再运行。示例中的 GPU 2 仅是当前服务器的
空闲卡选择，不是算法超参数。

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/benchmark_base_model.py
python scripts/finalize_base_metrics.py
```

结果会保存到 `results/base_model/`，包括逐题生成、IFEval strict 详情和
`metrics.json`。这些运行产物默认不提交。
