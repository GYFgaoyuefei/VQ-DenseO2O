# 番茄成熟度检测 · paper_v2 配置与实验说明

[配置目录](../../configs/tomato/paper_v2)包含 1 份公共配置和 9 份可运行的实验配置。它们使用同一套 DEIMv2-L 训练配方，比较实例前景质量监督（VQ）、光照兼容实例增强（LI）和空间细节门控（DC）对三类番茄成熟度检测的影响。当前算法套件为 `beta_select_algorithms_v1`；训练和预检入口要求 Git 分支为 `paper_v3.0`。以下命令均从仓库根目录运行。

## 1. 选择实验配置

文件名去掉 `.yml` 就是命令行的 `--model` 值。点击文件名可查看该组相对公共配方的全部覆盖项。表中的“掩膜同步”指训练时随图像变换传播实例掩膜。

| 编号 | 配置文件 | 掩膜同步 | VQ | LI | DC |
| --- | --- | :---: | :---: | :---: | :---: |
| B0 | [baseline_native.yml](../../configs/tomato/paper_v2/baseline_native.yml) | 关 | 关 | 关 | 关 |
| B1 | [baseline_maskaware.yml](../../configs/tomato/paper_v2/baseline_maskaware.yml) | 开 | 关 | 关 | 关 |
| V1 | [vq_dense_o2o.yml](../../configs/tomato/paper_v2/vq_dense_o2o.yml) | 开 | 开 | 关 | 关 |
| L1 | [li_copyblend.yml](../../configs/tomato/paper_v2/li_copyblend.yml) | 开 | 关 | 开 | 关 |
| D1 | [dc_sta.yml](../../configs/tomato/paper_v2/dc_sta.yml) | 开 | 关 | 关 | 开 |
| C1 | [vq_dense_o2o_li_copyblend.yml](../../configs/tomato/paper_v2/combinations/vq_dense_o2o_li_copyblend.yml) | 开 | 开 | 开 | 关 |
| C2 | [vq_dense_o2o_dc_sta.yml](../../configs/tomato/paper_v2/combinations/vq_dense_o2o_dc_sta.yml) | 开 | 开 | 关 | 开 |
| C3 | [li_copyblend_dc_sta.yml](../../configs/tomato/paper_v2/combinations/li_copyblend_dc_sta.yml) | 开 | 关 | 开 | 开 |
| C4 | [vq_dense_o2o_li_copyblend_dc_sta.yml](../../configs/tomato/paper_v2/combinations/vq_dense_o2o_li_copyblend_dc_sta.yml) | 开 | 开 | 开 | 开 |

**怎样选对照组。** B0 与 B1 的差异用于观察掩膜数据管线本身的影响。V1、L1、D1 分别与 B1 比较，可考察各模块的独立增量。B1 至 C4 覆盖三个开关的全部 8 种组合。计算交互时使用同一个 seed，例如固定 DC 关闭时，VQ 与 LI 的交互量为 `AP(C1) − AP(V1) − AP(L1) + AP(B1)`。

## 2. 公共训练配方

9 份实验配置均继承 [common.yml](../../configs/tomato/paper_v2/common.yml)。它在 [DEIMv2-L COCO 配方](../../configs/deimv2/deimv2_dinov3_l_coco.yml)上覆盖类别、数据路径、轮数、增强阶段和匹配器切换时点。运行入口会展开继承并依据开关选择数据集、损失函数和门控组件。每次正式训练保存的 `config_resolved.yml` 可用于核对该次运行的实际参数。[配置解析实现](../../engine/core/innovation_config.py)

| 项目 | 9 组共用的有效设置 |
| --- | --- |
| 检测器 | DINOv3 ViT-S/16，语义特征索引 `[5, 8, 11]`，STA 输出步长 8、16、32；三个尺度各 224 通道，4 层解码器，300 个查询 |
| 输入与类别 | RGB，基准尺寸 640 × 640，训练保留多尺度调度；未熟、半熟、成熟三类，COCO 类别 ID 集合为 `[1, 2, 3]` |
| 训练预算 | 120 个 epoch；种子 `0、42、3407`；全局 batch 为训练 32、验证 64 |
| 优化器 | AdamW，基础学习率 `5e-4`，DINO 参数组 `1.25e-5`，基础权重衰减 `1.25e-4` |
| 学习率与增强调度 | FlatCosine，`warmup_iter=2000`、`flat_epoch=60`、`no_aug_epoch=14`、`lr_gamma=0.5`；增强策略切换点为 `[7, 60, 106]` |
| 实例增强 | 按从 0 开始的 epoch 计，MixUp 在 `[7,60)`，CopyBlend 在 `[7,106)`。两者配置概率均为 0.5；重叠时先抽 MixUp，未触发才抽 CopyBlend |
| 损失与匹配 | MAL 指数 `gamma=1.5`；匹配器切换参数 `matcher_change_epoch=88` |
| 训练与选模 | 训练启用 AMP 和 EMA（`decay=0.9999`、`warmups=1000`）。按 **EMA FP32 Mixed Validation AP** 保存 `best.pth`，选模时关闭 autocast 和 TF32，精确并列取较早 epoch |

## 3. 三个开关实际改变什么

### VQ-DenseO2O · 普通匹配正样本的分类监督

B0、B1 的 MAL 正类软目标为 `IoU^1.5`。开启 VQ 后，对预测框与匹配实例计算覆盖率 `C = 交集面积 / 可见掩膜面积` 和纯度 `P = 交集面积 / 预测框面积`。交集取预测框内属于该实例的可见前景。当前配置固定 `beta=0.10`。

```text
q_fg = √(C × P)
q_vq = stop_gradient(IoU × (0.90 + 0.10 × q_fg))
正类软目标 = q_vq ^ 1.5
```

交集通过 FP32 积分图按连续坐标计算；纯度分母取裁剪前的预测框面积。掩膜为空或标记无效时，该实例回退到原 IoU 目标。VQ 仅作用于 main、aux、pre、encoder 普通匹配正样本的分类损失；DN 分支、匹配、框回归和推理沿用基线。可用 `fallback_ratio` 检查回退占比，用 `target_delta_from_native_mal` 检查目标实际变化。[前景质量计算](../../engine/deim/vq_quality.py) · [损失分支实现](../../engine/deim/vq_criterion.py)

### LI-CopyBlend · 训练期实例增强

LI 在既有 CopyBlend 调用点从当前训练 batch 选择带有效可见掩膜的供体。`light_policy=strict` 要求已知光照的接收图使用同来源光照的供体。接收图光照未知或混合时依靠局部背景判断；正式配置对候选位置开启局部光度检查，阈值为 `0.20`。位置重试上限 `20`，`max_donor_attempts=12`，碰撞容忍度为 `0.0`。

接收图按照 `I_new = I_dst + λ × M × (I_src − I_dst)` 融合，只有二值前景掩膜 `M` 内的像素改变。接受粘贴后同步框、类别、掩膜、面积、实例标识和来源光照。B0、B1、V1、D1 的增强时段相同，使用原生矩形 CopyBlend。比较 LI 实验时应同时看 `attempted_pastes`、`accepted_pastes` 和每图新增实例数，配置概率只说明进入增强分支的机会。[LI 筛选与融合](../../engine/data/li_copyblend.py) · [增强调用与调度](../../engine/data/dataloader.py)

### DC-STA · 训练和推理共用的细节门控

DC 从反归一化 RGB 的 80 × 80 表示提取 8 项统计量，分别为亮度均值、亮度标准差、暗区与亮区软响应、R/G/B 三通道均值，以及相邻像素亮度差分。对 `c2、c3、c4` 三层，通道 MLP 和空间卷积各生成一个 Sigmoid 门，计算 `调节后细节 = STA细节 × 通道门 × 空间门`，再与原语义特征拼接。`gate_init: identity` 对应的门值初始乘积约为 `0.999`。

D1 采用与 B1 一致的掩膜同步管线，门控本身只读取 RGB 和特征。VQ 和 LI 的附加计算都在训练期；含 DC 的模型部署时仍执行门控。因此比较部署参数量和延迟时，应区分是否开启 DC。[DC 门控实现](../../engine/backbone/dc_sta.py) · [与主干的连接](../../engine/backbone/dinov3_adapter.py)

## 4. 数据和预训练权重

运行正式训练前，在当前 shell 配置以下资源。`TOMATO_DATA_ROOT` 下的 `images/train` 和 `images/val` 分别对应 `annotations/instances_train.json`、`annotations/instances_val.json`。最终 Test 评估另读 `images/test` 和 `annotations/instances_test.json`。

| 环境变量 | 用途 |
| --- | --- |
| `TOMATO_DATA_ROOT` | 番茄 COCO 数据根目录，标注中的 `file_name` 须与相应图像目录匹配 |
| `DINOV3_WEIGHTS` | DINOv3 ViT-S/16 骨干预训练权重 |
| `TOMATO_DETECTOR_CHECKPOINT` | 9 组训练共用的 DEIMv2-L detector 初始化权重 |
| `TOMATO_VISIBLE_MASK_TRAIN` | 可选。B1 至 C4 默认读取 `annotations/visible_masks_train.json`，可用此变量指定另一份与 Train 标注匹配的 sidecar |

B0 用原生 `CocoDetection`。其余 8 组用 `VisibleMaskCocoDetection`，加载一次 Train 掩膜，并在裁剪、缩放、Mosaic、MixUp、CopyBlend 和实例过滤后同步几何与实例字段。sidecar 版本为 `visible-mask-sidecar-v1`，加载时校验 Train 标注文件的 SHA256，以及记录的实例 ID、图像 ID 和类别 ID。训练标注一旦变动，应核对相应的 sidecar。验证与测试仅用 RGB 和检测标注。[数据加载与 sidecar 校验](../../engine/data/dataset/visible_coco_dataset.py)

## 5. 从预检到正式训练

**Windows 本地核对。** 在已安装项目依赖的 Python 环境中运行。第一个命令只解析九组配置；第二个打印九模型 × 三种子的 27 条训练命令，供服务器使用。

```powershell
Set-Location 'D:\workspace\python_workspace\study\VQ-DenseO2O'
python -X utf8 tools/paper_v2/run.py preflight --model all --config-only
python -X utf8 tools/paper_v2/run.py commands
```

需要检查模块的确定性行为时，可运行 `python -m pytest tests/paper_v2 -q`。要审计解析后的配置差异并把临时报告留在实验输出目录，可调用现有审计函数，显式指定报告位置。

```shell
python -c "from tools.paper_v2.config_audit import audit; audit(report_path='outputs/paper_v2/config_ablation_audit.md')"
```

**AutoDL 训练。** 在服务器仓库根目录加载现有环境脚本，再核对它设置的数据、权重和 Python 路径。如果机器上的资源位置不同，在当前 shell 中覆盖相应变量。下面以 L1、seed 42 为例，先检查资源，再用增强生效阶段的 3 个真实 Train batch 验证数据和梯度，最后启动该组训练。冒烟检查的输出独立于正式训练目录。

```bash
source scripts/paper_v2/autodl/env.sh
unset RESUME
bash scripts/paper_v2/autodl/preflight.sh all
bash scripts/paper_v2/autodl/smoke_train_one.sh li_copyblend 42 cuda:0 --smoke-steps 3 --smoke-epoch 10
bash scripts/paper_v2/autodl/train_one.sh li_copyblend 42 cuda:0
```

跑全矩阵时，按 `commands` 打印的列表选择尚未完成的组合。`train_seed42.sh` 和 `train_remaining_seeds.sh` 合计只覆盖 B0、B1、V1、L1、D1 的 15 次训练；C1 至 C4 另有 12 次。多卡训练通过 `NPROC_PER_NODE` 设置进程数，总训练 batch 32 须能被进程数整除。[训练入口](../../tools/paper_v2/run.py) · [AutoDL 脚本](../../scripts/paper_v2/autodl)

**结果与续训。** 默认训练目录为 `outputs/paper_v2_beta_select_algorithms/<model>/seed_<seed>/`。`config_resolved.yml` 记录实际配置，`run_info.json` 记录版本和启动参数，`epoch_validation.csv` 与 `selection.json` 记录验证与选模，`train.log` 保存运行日志。`best.pth` 用于选定模型的评估；中断后可在同一模型、同一种子、同一目录使用 `last.pth` 续训。例如恢复上面的 L1 任务：

```bash
RESUME="$PWD/outputs/paper_v2_beta_select_algorithms/li_copyblend/seed_42/last.pth" \
  bash scripts/paper_v2/autodl/train_one.sh li_copyblend 42 cuda:0
```

## 6. Test、效率与三种子汇总

训练选模完成后，使用 [最终评估入口](../../tools/paper_v2/final_test.py)统一处理 9 组 × 3 个 seed。下例沿用上一节的数据环境。先用 `--dry-run` 查看 27 个目录及缺失文件，其退出状态不能代替对缺件清单的检查。

```bash
bash scripts/paper_v2/autodl/evaluate_final_test_all.sh cuda:0 --dry-run
bash scripts/paper_v2/autodl/evaluate_final_test_all.sh cuda:0 --mode all
```

也可按阶段选择 `--mode ap-only`、`efficiency-only` 或 `summarize-only`。其中 `summarize-only` 从已有评估记录重建表格。统一入口默认 `--run-policy existing-checkpoint`，读取各 seed 的 `best.pth` 中的完整 FP32 EMA 权重，优先使用随该次运行保存的推理配置。缺少原配置时会记录仓库配置回退，缺少选模证据时会标记 `UNVERIFIED`。需要核验完整训练记录时可显式使用 `--run-policy strict`。`--runs-dir` 与 `--data-root` 可指定训练根目录和数据位置。Test 仅用于评估已选权重。

| 结果 | 读取方法 |
| --- | --- |
| Mixed 与四光照 AP | 每个 checkpoint 推理一次。Mixed 在全部 Test 图像上计算 COCO bbox AP/AR；自然光、人工光、微弱光、钠黄光分别在各自的图像子集上计算 AP。光照取自采集元数据，必要时传 `--illumination-map` 指定经核实的图像 ID 或文件名映射 |
| 逐 seed 与汇总 | AP 按百分数保存，各列在 0、42、3407 都有有效值时计算均值与样本标准差（`ddof=1`）。缺失或 COCO 未定义的值留空；合法的零值计入统计。表中 `Epoch=120` 表示预算，实际选中轮次查逐 seed 记录 |
| 部署效率 | 统一协议使用 NVIDIA RTX PRO 6000 Blackwell Server Edition，PyTorch eager、FP32、关闭 AMP/TF32，batch 1，GPU 常驻全零 `1×3×640×640` 输入。每 seed 5 轮，每轮每个计时范围预热 100 次、计时 500 次。主延迟直接测量 detector forward 加 tensor postprocessor，FPS 取 `1000 / 平均延迟(ms)`。主表参数量按部署模型计算，GFLOPs 仅估算部署 detector forward，按实际算子覆盖标记 `PARTIAL` |

默认输出均相对于 `outputs/paper_v2_beta_select_algorithms/`：

| 路径 | 内容 |
| --- | --- |
| `ap/test/raw/<model>/seed_<seed>/` | `predictions.json` 与 `metrics.json`，用于逐图核对、光照计数和定性比较 |
| `ap/test/ap_by_seed.csv`、`ap/test/ap_summary.csv` | 逐 seed 及三种子 AP |
| `efficiency/raw/<model>/seed_<seed>/efficiency.json` | 该次运行的设备、计时、参数量和计算量记录 |
| `final_test/test_results_by_seed.csv`、`final_test/test_results_summary.csv` | 可追溯的逐 seed 数值和跨种子统计 |
| `final_test/paper_test_table.csv` | 九模型最终对照表；运行状态和缺失原因见同目录的 `final_test_protocol.json`、`final_test_attempts.json` |

### 可选：用同一批 Test 预测制作 Fig. 8

[定性比较工具](../../tools/paper_v2/qualitative_fig8.py)读取已保存的预测、Test 标注和 RGB 原图。图中六列依次为 GT / Input、B1、VQ、LI、DC、Full，对应原图与标注、B1、V1、L1、D1、C4。默认 seed 42、置信度阈值 0.30，按同类一对一匹配且 IoU ≥ 0.50 记录单图 TP、FP、FN。默认核对 400 张图及预测所属数据集。先生成候选排名与联系表，再根据原图选取两个真实 COCO `image_id`；单图诊断用于解释案例，整体精度取上一节的 COCO 结果。

```bash
root=outputs/paper_v2_beta_select_algorithms
fig_args=(--annotations "$TOMATO_DATA_ROOT/annotations/instances_test.json"
          --images "$TOMATO_DATA_ROOT/images/test" --pred-root "$root/ap/test/raw"
          --seed 42 --score-thr 0.30)
"${PYTHON:-python}" tools/paper_v2/qualitative_fig8.py --mode rank "${fig_args[@]}" --out "$root/fig8/rank"
read -rp '输入两个已复核的 COCO image_id，以空格分隔：' image_a image_b
"${PYTHON:-python}" tools/paper_v2/qualitative_fig8.py --mode render "${fig_args[@]}" \
  --image-ids "$image_a" "$image_b" --out "$root/fig8/cases_${image_a}_${image_b}"
```

正式导出包含 PNG、TIFF、SVG、案例 CSV、审计 JSON 和中英文图注。需要观察局部时可用 `--roi-json` 指定同一行各模型共用的原图区域。
