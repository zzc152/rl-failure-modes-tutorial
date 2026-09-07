# rl-failures

可复现的 RL / DPO 失效模式教程。所有实验常量定义于
[`configs/experiment_constants.yaml`](configs/experiment_constants.yaml)，失败实验只能改变自身的故障注入变量。

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
