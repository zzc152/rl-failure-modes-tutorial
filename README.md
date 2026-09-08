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

## Quick Start

### 1. 创建环境

```bash
conda create -n rl-failures python=3.10 -y
conda activate rl-failures
pip install -r requirements.txt
```

### 2. 下载模型与数据

项目使用 Hugging Face 镜像。下载脚本会先检查关键文件：完整资产会跳过，
缺失或不完整的资产才会下载或续传。模型和数据只保存在项目目录内，且已被 Git 忽略：

```bash
bash scripts/download_assets.sh
```

下载完成后的目录结构：

```text
rl-failures/
├── models/Qwen2.5-1.5B-Instruct/
└── data/
    ├── ultrafeedback_binarized/
    ├── ifeval/
    └── gsm8k/
```

`models/`、`data/`、`results/`、日志及缓存均不会提交到 GitHub。

## 一键运行当前 benchmark

完成上面的模型和数据下载后，运行：

```bash
bash scripts/run_base_benchmark.sh
```

该命令会按需获取 IFEval 的离线 strict evaluator、下载 NLTK tokenizer，随后生成并
汇总 UltraFeedback held-out preference accuracy、IFEval strict、GSM8K accuracy 与平均回复长度。
默认使用当前可见的第一张 CUDA GPU；如需选择设备，请在命令前设置
`CUDA_VISIBLE_DEVICES`，例如 `CUDA_VISIBLE_DEVICES=1 bash scripts/run_base_benchmark.sh`。
