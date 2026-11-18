# study02 · DEIMv2-M 蒸馏目标对照

本轮预算为两次训练，各用 seed 42 从同一原生 M 初始化训练 120 轮。目标是分别估计增强原 MSE 的效果，以及在相同权重下改用 CWD 的效果。旧 M/seed 42 的两条 Validation 记录已足以提供参照。

| 实验 | 蒸馏目标 | 最高 λ | 温度 | Mixed Val AP | 入选 EMA 轮次 |
|---|---|---:|---:|---:|---:|
| study01 · `m_base` | 检测监督 | 0 | — | 77.04655% | 80 |
| study01 · `m_kd` | 原逐元素 MSE | 1 | — | 77.61627% | 98 |
| study02 · `m_mse_l5` | 原逐元素 MSE | 5 | — | 待训练 | — |
| study02 · `m_cwd_l5_t1` | CWD 空间 KL | 5 | 1 | 待训练 | — |

### 实验定义与可比性

两组使用同一个冻结的 VQ-DEIMv2-L 教师 T0。上一轮已在 400 张 Val 图像上复评，Mixed AP 为 79.3063%。M/S 在第 120 轮的实时模型上各测过 8 个 Train 批次，共享参数 KD/检测梯度范数比中位数分别为 0.618% 和 0.545%。这些测量覆盖一个训练阶段；新候选的实际梯度强度要从自己的早期快照测量。

两组共用原检测监督、增强、多尺度输入、优化器、学习率、全局梯度裁剪 0.1、EMA 与验证选模规则。取 HybridEncoder 融合后的 P3、P4、P5，每个尺度有一个独立的 1×1 无偏置学生投影（256→224）。教师以 eval、FP32、无梯度运行 backbone 和 encoder。学生和投影层参与反向传播。有效权重按成功优化器更新次数在 5 个 epoch 的计划步数内线性升至 5，三个尺度等权平均。

对每个尺度，CWD 把特征从 `[B,C,H,W]` 展平为 `[B,C,H×W]`。学生和教师分别沿空间维做 `log_softmax(feature/T)`。逐位置计算 `p_teacher × (log_p_teacher − log_p_student)`，对空间求和，对样本和通道求均值，再乘 `T²`。这里采用已有 CWD 方法在 DEIMv2 编码器上的适配。λ=5 与 T=1 是开发设置；两个目标的实际梯度强度需分别测量。[Shu 等，ICCV 2021，式 (3)–(4)，第 5311–5320 页](https://openaccess.thecvf.com/content/ICCV2021/html/Shu_Channel-Wise_Knowledge_Distillation_for_Dense_Prediction_ICCV_2021_paper.html)

### AutoDL 执行顺序

将本次交付的 `study02.patch` 上传到 `/root/autodl-tmp/study02.patch`。从工程的父目录运行下面的命令。Windows worktree 复制到 AutoDL 后，工程内的 `.git` 文件仍可能指向原来的 D 盘路径；从父目录应用补丁可直接更新源码。若已完整应用过同一补丁，命令会显示“补丁已应用”。若检查失败，先核对报错中列出的文件。

```bash
(
  cd /root/autodl-tmp || exit 1
  patch_file=/root/autodl-tmp/study02.patch
  apply=(git apply --directory=VQ-DenseO2O-thesis-kd --ignore-space-change)
  if "${apply[@]}" --reverse --check "$patch_file" 2>/dev/null; then
    echo '补丁已应用'
  else
    "${apply[@]}" --check "$patch_file" && "${apply[@]}" "$patch_file" && echo '补丁应用成功'
  fi
)
```

以下代码块在同一个 Bash 会话中按阶段执行。沿用 study01 的单卡、Train 全局 batch 32、Val 全局 batch 64。旧运行记录的环境为 PyTorch 2.8.0+cu128、RTX PRO 6000 Blackwell Server Edition。若更换软件、GPU、数据或初始化来源，在比较结果时注明差异。

```bash
cd /root/autodl-tmp/VQ-DenseO2O-thesis-kd
source scripts/thesis_kd/autodl/env.example.sh
export RUN_TAG=study02 NPROC_PER_NODE=1
set -e

for exp in m_mse_l5 m_cwd_l5_t1; do
  bash scripts/thesis_kd/autodl/preflight.sh "$exp" 42 --device cuda:0
done
```

预检加载 M 的 `vittplus_distill.pt`（raw）和 `deimv2_dinov3_m_coco.pth`（model），并检查教师、配方和 Train/Val 标注。教师配置为 `configs/tomato/paper_v2/vq_dense_o2o.yml`，权重为 `/root/autodl-tmp/pretrained/thesis_kd/vq_dense_o2o/seed_3407/best.pth`，分支为 `ema.module`。检查两份预检输出中 `input_identity.student_init_sha256` 相同。旧记录只保存了初始化路径、加载分支和跳过的分类头，无法逐字节追溯旧学生初始化文件。

```bash
for exp in m_mse_l5 m_cwd_l5_t1; do
  bash scripts/thesis_kd/autodl/smoke_one.sh "$exp" 42 --steps 8 --device cuda:0
done
```

短程运行使用正式资源和 batch，检查各尺度损失有限、`optimizer_updates` 为正，并记录 AMP 跳步次数。结果位于 `outputs/thesis_kd_smoke/feature_kd_objective_v1/study02/<实验>/seed_42/`。正式训练使用独立目录，重新从原生预训练初始化。

```bash
bash scripts/thesis_kd/autodl/train_one.sh m_mse_l5 42
bash scripts/thesis_kd/autodl/train_one.sh m_cwd_l5_t1 42
```

中断后，在同一工程和同一环境中设置同样的 `RUN_TAG`，只对中断的实验执行对应行。恢复从最近已提交轮次的 `last_training.pth` 继续，未提交的轮次重跑。

```bash
# m_mse_l5 中断时
bash scripts/thesis_kd/autodl/resume_one.sh m_mse_l5 42
# m_cwd_l5_t1 中断时
bash scripts/thesis_kd/autodl/resume_one.sh m_cwd_l5_t1 42
```

两组均训练到 120 轮后，分别对入选 EMA 在完整 Validation 上做一次部署态 FP32 评估。新协议在发布结果前核对训练记录完整覆盖第 1–120 轮，并核对入选权重的 SHA256。

```bash
bash scripts/thesis_kd/autodl/validate_one.sh m_mse_l5 42 --device cuda:0
bash scripts/thesis_kd/autodl/validate_one.sh m_cwd_l5_t1 42 --device cuda:0
```

### 输出与早期梯度

正式输出根为 `$THESIS_KD_OUTPUT_ROOT/feature_kd_objective_v1/study02/<实验>/seed_42/`。

| 文件 | 用途 |
|---|---|
| `run_info.json`、`config_resolved.yml` | 初始化分支、SHA256、数据和配方摘要、硬件、精度、完整解析配置 |
| `kd_diagnostics.json`、`train.log` | 每轮按 batch 平均的 `native_loss`、`mean_kd`、`weighted_kd`、`kd_P3`、`kd_P4`、`kd_P5`、`effective_lambda`，以及成功更新数和 AMP 跳步数 |
| `epoch_005_training.pth`、`epoch_020_training.pth`、`epoch_060_training.pth`、`epoch_120_training.pth` | 四份独立完整快照，包含实时学生、投影、EMA、优化器、AMP scaler、调度进度、各 rank 随机状态、loader 生成器状态和选择记录；需给每组四份快照及常规权重预留磁盘空间 |
| `validation_history.csv`、`selection.json`、`best_student.pth` | 逐轮 FP32 Mixed Val AP、入选轮次及 SHA256、最佳 EMA 学生权重 |
| `evaluation_val/metrics.json` | 最佳 EMA 的 Mixed AP、AP50/75、目标尺度、四种光照 AP、成熟度类别 AP（`per_class_ap_percent`） |

两组各取第 5 轮快照，在 GPU 空闲时启动独立进程。训练期间已保留的快照可以在训练结束后分析。将 `--epoch 5` 改为 20、60 或 120 即可测量相应阶段。

```bash
for exp in m_mse_l5 m_cwd_l5_t1; do
  "$PYTHON" -m thesis_kd.objective_probe --experiment "$exp" --run-tag "$RUN_TAG" --epoch 5 --batches 8 --device cuda:0
done
```

探针恢复实时学生和同一轮已训练投影，在私有副本上用 8 个新 Train 批次以 FP32 计算梯度。报告记录每个批次的图像 ID、逐尺度损失、有效 λ，以及 backbone、encoder 上加权 KD 与检测损失的裁剪前梯度范数比 R 和方向余弦。探针不做优化器更新，报告在 `objective_probes/epoch_005_8_batches.json`。这是快照采样测量，解读时结合相应轮次的 AMP 训练日志。

### 预先固定的判定

主指标为训练期间最高的 **FP32 EMA Mixed Validation AP**，同值取较早轮次。使用 `selection.json.mixed_ap` 的原始小数比较，不用百分数显示值舍入后比较。设候选为 `a`，须同时满足 `a >= 0.7704654969770546 + 0.003` 和 `a > 0.776162736051232`。0.3 个百分点是继续投入的门槛。四光照、类别、尺度、学习曲线和梯度用于解释结果。

两组均完成并生成 Val 报告后，在上述 Bash 环境中读取结果：

```bash
"$PYTHON" - <<'PY'
import csv
import json
import os
from pathlib import Path

root = Path(os.environ["THESIS_KD_OUTPUT_ROOT"]) / "feature_kd_objective_v1" / "study02"
base = 0.7704654969770546
old_mse = 0.776162736051232
inputs = []
for exp in ("m_mse_l5", "m_cwd_l5_t1"):
    run = root / exp / "seed_42"
    def read(name):
        return json.loads((run / name).read_text(encoding="utf8"))
    info = read("run_info.json")
    selected = read("selection.json")
    val = read("evaluation_val/metrics.json")
    with (run / "validation_history.csv").open(encoding="utf8", newline="") as f:
        history = list(csv.DictReader(f))
    assert [int(row["epoch"]) for row in history] == list(range(1, 121)), exp
    assert info["experiment"] == exp and info["seed"] == 42 and info["run_tag"] == "study02", exp
    assert info["world_size"] == 1 and info["global_batch"] == 32 and info["validation_global_batch"] == 64, exp
    assert val["split"] == "val" and val["status"] == "COMPLETE" and val["selection"] == selected, exp
    inputs.append(info["study02_inputs"])
    assert inputs[0] == inputs[-1], "两组资源摘要不同"
    a = selected["mixed_ap"]
    passed = a >= base + 0.003 and a > old_mse
    print(f"{exp}  epoch={selected['epoch']}  AP={100*a:.5f}%  "
          f"ΔBase={100*(a-base):+.5f}pp  ΔMSE1={100*(a-old_mse):+.5f}pp  过线={passed}")
PY
```

两组都未达门槛就结束本轮。若有候选达标，先冻结其定义，再在 M 上补 seed 0、3407，按种子配对报告 Validation 差值。如果两组都满足，优先选择原始 Mixed Val AP 较高者；若精确相同，保留并列并在追加预算前记录选择理由。S 的容量迁移留到后续。此前研究已观察过 Test，后续报告应注明该历史。

### 交付和本地验收（2026-11-14）

开发开始时，隔离工作树干净，HEAD 为 `515c20fb36d5e2917db19de5f04c71f2bcea07be`。交付时原研究工作树仍干净，HEAD 为 `44bdef481c232d92f78cb5365a9232d9e43cc0bd`。隔离工程的 `engine/`、`train.py`、旧研究配置和依赖文件均与开发起点一致。10 项旧实验的完整解析配置和 `pair_diff` 已逐项核对。旧核心和尺度矩阵仍生成 15 条命令，加入旧权重敏感性组后为 18 条。

| 验收环境 | 结果 | 范围 |
|---|---|---|
| CPU | PASS | CWD 数学性质、空间归一化、无额外 HW 除数、温度、教师/学生梯度、旧 MSE 一致性；两种目标均用原生模型在生成批次上完成两次优化器更新 |
| 本地 CUDA（PyTorch 2.7.1+cu126，RTX 4060 Laptop） | PASS | 两组原生 M 学生和 VQ-L 教师结构以测试初始化运行，CUDA AMP 前向、反向、优化器更新，损失和梯度有限 |
| 训练状态与旧流程 | PASS | 完整快照、中断恢复、独立梯度采样、旧 Base/KD 推理、导出及加载回归 |
| 本地真实数据 | PASS | 原 Train/Val 数据加载与标注核对，分别为 3,200 / 400 张。标注 SHA256 在新协议资源预检和训练入口核对 |
| AutoDL 真实 M 初始化及 T0 联合 smoke | NOT RUN | 本地缺少 M 的两份预训练与 T0 检查点，需在 AutoDL 做上述预检和短程运行；先前教师 Val 复评已有独立记录 |
| 正式训练、两组 Val 及真实快照梯度 | NOT RUN | 依次执行本章命令取得数值。本次开发没有启动完整训练或读取 Test 输入 |

最终回归验收为 **91 通过、1 跳过**。跳过项为真实 T0 装载。Windows 首次运行旧恢复用例时遇到临时文件替换失败，重跑通过；真实标注读取需启用 UTF-8（PowerShell 设置 `$env:PYTHONUTF8='1'`）。单卡 DDP 回归有一条梯度步幅性能提示，该用例通过。本地验收记录为 `outputs/thesis_kd_checks/study02_acceptance_20261114_141656/acceptance_final.xml`。

| 文件（相对于工程根） | 变更 |
|---|---|
| `configs/tomato/thesis_kd/experiments/m_mse_l5.yml`、`m_cwd_l5_t1.yml` | 新增两个实验 |
| `thesis_kd/config.py`、`cli.py` | 新实验解析、资源预检和入口注册 |
| `thesis_kd/objective_losses.py`、`objective_inputs.py`、`objective_probe.py` | 新目标、输入身份校验、独立早期诊断 |
| `thesis_kd/trainer.py`、`checkpoint.py`、`evaluate.py` | 新协议的目标选择、完整快照与恢复、Validation 类别 AP |
| `tests/thesis_kd/test_objective_config.py`、`test_objective_losses.py`、`test_objective_probe.py`、`test_objective_training.py`、`test_objective_validation.py` | 配置、数学、梯度采样、训练与恢复、Val 指标和输入身份回归 |
| `docs/thesis_kd/11_study02_M蒸馏对照.md`、`docs/thesis_kd/README.md` | 操作手册与目录入口 |
