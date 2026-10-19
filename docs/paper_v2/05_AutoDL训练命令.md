# Paper v3.0 AutoDL 训练命令

日期：2026-10-19。本次交付只完成代码、配置与有限验收；下文正式训练和最终 Test 命令供后续用户显式执行，本阶段未启动 120 轮训练、未读取 Test。

## 1. 同步分支并设置资源

推荐先在 Windows 手动审查、提交并推送 `paper_v3.0`，然后在服务器执行：

```bash
cd /root/autodl-tmp/DEIMv2-new
git status --short
git fetch origin
git switch paper_v3.0
git pull --ff-only origin paper_v3.0
git branch --show-current
```

若服务器存在未提交源码修改，先核对并保留这些修改再同步；不要执行 `reset --hard`、`clean` 或 `merge beta_select`。`git switch paper_v3.0` 在远端存在唯一同名分支时可建立本地跟踪分支；若没有远端同步条件，按 Windows 说明手动上传本轮全部变更，并确认分支名和源码均正确。无论哪种方式，都保留原 `outputs/`、数据和权重。

激活已有训练 Python 环境后：

```bash
source scripts/paper_v2/autodl/env.sh
unset RESUME
```

`env.sh` 使用用户提供的资源位置：

```bash
export TOMATO_DATA_ROOT="/root/autodl-tmp/dataset/tomato_coco"
export DINOV3_WEIGHTS="/root/autodl-tmp/pretrained/deimv2_l/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
export TOMATO_DETECTOR_CHECKPOINT="/root/autodl-tmp/pretrained/deimv2_l/deimv2_dinov3_l_coco.pth"
export PYTHON="python"
export NPROC_PER_NODE=1
```

数据目录约定：

```text
tomato_coco/
  images/train/
  images/val/
  annotations/instances_train.json
  annotations/instances_val.json
  annotations/visible_masks_train.json
```

mask sidecar 的实际文件名需要服务器确认。若经确认使用其他文件名，通过环境变量指定，无须重命名：

```bash
export TOMATO_VISIBLE_MASK_TRAIN="/root/autodl-tmp/dataset/tomato_coco/annotations/<实际训练mask文件名>.json"
```

将占位内容替换为真实路径再运行。B0 不要求实例 mask；B1 和全部创新/组合模型共用同一 Train sidecar。不要用 Validation/Test mask 作为训练供体。`PYTHON` 可改为服务器训练环境的 Python 绝对路径。

## 2. 九组预检与必要测试

```bash
bash scripts/paper_v2/autodl/preflight.sh all --config-only
"$PYTHON" tools/paper_v2/run.py audit-configs
"$PYTHON" -m pytest tests/paper_v2 -q
bash scripts/paper_v2/autodl/preflight.sh all
```

前三项分别完成配置解析、轻量消融差异审计及合成回归。完整 preflight 检查 Train/Validation 路径、类别 ID、sidecar 和真实组件构造，不读取 Test。审计输出为 `outputs/paper_v2/config_ablation_matrix.csv` 与 `docs/paper_v2/ablation_config_audit.md`；训练协议与分支检查应全部通过后再启动实验。

参考对照使用固定 beta_select commit `f92edb021fd468e354f5767bb81ac745e0347cfe`。若服务器 Git 缺少该对象，可从同一远端 `git fetch origin beta_select` 获取历史，不执行 merge。缺少参考对象时对应测试明确 SKIP（该环境记 NOT RUN），不能把其余测试通过当作参考对照通过。

可先做一个不依赖真实数据的 CPU 冒烟：

```bash
bash scripts/paper_v2/autodl/smoke_one.sh vq_dense_o2o_li_copyblend_dc_sta 42 cpu --synthetic --threads 2 --report outputs/paper_v2_beta_select_algorithms/checks/synthetic_cpu/C4_seed42.json
```

这只检查 128×128 合成输入的一步训练及推理路径，不代表真实 batch 32 显存足够，也不代表真实 LI 供体已接受。

## 3. 真实 Train 有限步验收

以下保持正式有效 batch 32、统一初始化、AMP、EMA、optimizer 和调度，只运行 3 个 Train batch：

```bash
bash scripts/paper_v2/autodl/smoke_train_one.sh baseline_native 42 cuda:0 --smoke-steps 3 --smoke-epoch 0
bash scripts/paper_v2/autodl/smoke_train_one.sh baseline_maskaware 42 cuda:0 --smoke-steps 3 --smoke-epoch 0
bash scripts/paper_v2/autodl/smoke_train_one.sh vq_dense_o2o 42 cuda:0 --smoke-steps 3 --smoke-epoch 10
bash scripts/paper_v2/autodl/smoke_train_one.sh li_copyblend 42 cuda:0 --smoke-steps 3 --smoke-epoch 10
bash scripts/paper_v2/autodl/smoke_train_one.sh dc_sta 42 cuda:0 --smoke-steps 3 --smoke-epoch 10
bash scripts/paper_v2/autodl/smoke_train_one.sh vq_dense_o2o_li_copyblend_dc_sta 42 cuda:0 --smoke-steps 3 --smoke-epoch 10
```

`--smoke-epoch 10` 将公共配方设置到增强活跃区间，并非从旧 epoch 10 权重恢复；每次均重新加载统一预训练。随机分支仍按原概率选择，不能把“活跃区间”解释为每个 batch 都必然执行 LI-CopyBlend。需要结合接受计数检查实际供体行为。

步数允许 1—10。输出目录默认带时间戳：

```text
outputs/paper_v2_beta_select_algorithms/checks/real_train/<model>/seed_42_<timestamp>/
  config_resolved.yml
  train_smoke_preflight.json
  train_smoke.log
  train_smoke_rank_0.json
```

报告包含显存、同步迭代时间、loss、梯度、实际 batch 和 DC optimizer/EMA 检查。不创建 Validation/Test dataloader，不保存正式 checkpoint，也不进入 120-epoch fit。真实数据资源与 GPU 仅在服务器，因此交付时这一项如未执行应记 `NOT RUN`。显存不足时应先统一调整运行资源，不能单独降低某个模型的有效 batch 后将结果混作相同消融协议。

## 4. 五组独立实验：seed 42 完整命令

以下命令会真正启动 120 epochs。先完成上述验收，再按实验安排执行。

```bash
unset RESUME
bash scripts/paper_v2/autodl/train_one.sh baseline_native 42 cuda:0
bash scripts/paper_v2/autodl/train_one.sh baseline_maskaware 42 cuda:0
bash scripts/paper_v2/autodl/train_one.sh vq_dense_o2o 42 cuda:0
bash scripts/paper_v2/autodl/train_one.sh li_copyblend 42 cuda:0
bash scripts/paper_v2/autodl/train_one.sh dc_sta 42 cuda:0
```

也可用现有顺序脚本替代上述五条命令，二者不要重复启动同一运行：

```bash
bash scripts/paper_v2/autodl/train_seed42.sh cuda:0
```

## 5. 四组组合实验：seed 42 完整命令

```bash
unset RESUME
bash scripts/paper_v2/autodl/train_one.sh vq_dense_o2o_li_copyblend 42 cuda:0
bash scripts/paper_v2/autodl/train_one.sh vq_dense_o2o_dc_sta 42 cuda:0
bash scripts/paper_v2/autodl/train_one.sh li_copyblend_dc_sta 42 cuda:0
bash scripts/paper_v2/autodl/train_one.sh vq_dense_o2o_li_copyblend_dc_sta 42 cuda:0
```

对应 C1、C2、C3、C4。CLI 使用完整模型名，不接受旧 `C1` 简写或 `gr_vq`。

## 6. 三 seeds 的统一执行方式

已完成五组独立实验 seed 42 后，补其 seed 0/3407：

```bash
bash scripts/paper_v2/autodl/train_remaining_seeds.sh cuda:0
```

该包装脚本只覆盖五组独立实验。若九组 seed 42 均已完成，且准备统一补齐全部组合，可使用以下循环替代上面的独立组包装脚本：

```bash
unset RESUME
models=(
  baseline_native baseline_maskaware vq_dense_o2o li_copyblend dc_sta
  vq_dense_o2o_li_copyblend vq_dense_o2o_dc_sta
  li_copyblend_dc_sta vq_dense_o2o_li_copyblend_dc_sta
)
for seed in 0 3407; do
  for model in "${models[@]}"; do
    bash scripts/paper_v2/autodl/train_one.sh "$model" "$seed" cuda:0 || exit "$?"
  done
done
```

从零开始完整九组 × 三 seeds 时，将循环 seed 改为 `42 0 3407`，不再单独执行前面的 seed 42 命令。只打印全部 27 条命令可用：

```bash
"$PYTHON" tools/paper_v2/run.py commands
```

包装脚本对已有非空输出会拒绝覆盖。模型之间不共享训练 checkpoint；同一 seed 只共享预训练来源、随机种子约定和公共配方。

## 7. 多 GPU、断点和结果位置

默认单 GPU。若使用 4 张 GPU：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4
bash scripts/paper_v2/autodl/train_one.sh vq_dense_o2o 42 cuda:0
```

入口会使用真实 `torch.distributed.run`，由 `LOCAL_RANK` 选择设备；有效 batch 始终为 32，各进程为 `32 / NPROC_PER_NODE`。进程数须整除 32。不同消融模型应使用相同训练资源设置。切回单卡时重新设置 `NPROC_PER_NODE=1` 和可见设备。

仅恢复本轮同模型、同 seed、同目录的未完成训练，例如：

```bash
export RESUME="/root/autodl-tmp/DEIMv2-new/outputs/paper_v2_beta_select_algorithms/vq_dense_o2o/seed_42/last.pth"
bash scripts/paper_v2/autodl/train_one.sh vq_dense_o2o 42 cuda:0
unset RESUME
```

也可把 `--resume /实际路径/last.pth` 作为 `train_one.sh` 的第四项之后的参数传入。不得用旧 GR-VQ、Residual DC-STA 或旧 Baseline 运行目录继续本轮实验。恢复使用本次运行保存的有效配置，不能换 seed、模块开关或将历史算法权重冒充本轮权重；已完成运行不会重新启动。

训练目录统一为：

```text
outputs/paper_v2_beta_select_algorithms/<model>/seed_<0|42|3407>/
  config_resolved.yml
  run_info.json
  train.log
  last.pth
  best.pth
  selection.json
  completed.json
```

`best.pth` 由 EMA FP32 Mixed Validation AP 选择，精确并列保留较早 epoch。`completed.json` 仅在正式训练正常结束后生成。旧 `outputs/paper_v2/` 的训练和 Test 结果原样保留，不能拼接进新结果表。

## 8. 后续显式评估与效率统计

本阶段不执行本节的真实评估。新训练完成后，可重验 Validation：

```bash
bash scripts/paper_v2/autodl/validate_one.sh vq_dense_o2o 42 cuda:0
```

最终论文 Test 只在用户明确决定执行后运行；它不参与选模：

```bash
# 单模型 Mixed Test 评估
bash scripts/paper_v2/autodl/evaluate_test_one.sh vq_dense_o2o 42 cuda:0

# 九模型 × 三 seeds：Mixed AP 及四光照 AP，统一汇总
bash scripts/paper_v2/autodl/evaluate_and_summarize_ap.sh cuda:0 all

# 仅重新汇总已有 AP JSON，不重新读取图像
bash scripts/paper_v2/autodl/summarize_ap.sh --ap-models baseline_native baseline_maskaware vq_dense_o2o li_copyblend dc_sta vq_dense_o2o_li_copyblend vq_dense_o2o_dc_sta li_copyblend_dc_sta vq_dense_o2o_li_copyblend_dc_sta
```

四光照为自然光、人工光、微弱光和钠黄光；必须使用真实采集标签或明确映射，不能从 Mixed AP 反推。AP 输出位于 `outputs/paper_v2_beta_select_algorithms/ap/test/`，包括 `ap_by_seed.csv`、`ap_summary.csv`、`ap_protocol.json`。不齐全的三 seed 指标保持缺失，不补零。

效率入口沿用同一固定协议，九组正式 checkpoint 都准备好后执行：

```bash
bash scripts/paper_v2/autodl/benchmark_all.sh cuda:0 all
bash scripts/paper_v2/autodl/summarize_efficiency.sh --ap-csv outputs/paper_v2_beta_select_algorithms/ap/test/ap_by_seed.csv
bash scripts/paper_v2/autodl/summarize_results.sh
```

统一 GPU 为 NVIDIA RTX PRO 6000 Blackwell Server Edition，PyTorch eager eval/deploy，FP32，autocast/TF32 关闭，batch 1，GPU 常驻全零 `[1,3,640,640]` 输入，预热 100、正式 500 iterations × 5 repeats/seed。主延迟实测 detector forward + tensor postprocessor；forward-only 和 postprocessor-only 独立测量，不相加替代主延迟。主表 Params 采用 Deploy Params，GFLOPs 只估计 deploy forward，存在不支持算子时保留 PARTIAL 说明。

效率文件写入 `outputs/paper_v2_beta_select_algorithms/efficiency/`。`summarize_results.sh` 汇总运行与已有效率记录，不自动推理 Test，也不代替 Mixed/四光照 AP 计算入口。
