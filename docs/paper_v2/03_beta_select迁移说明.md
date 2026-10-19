# beta_select 算法迁移说明

日期：2026-10-19。开发分支为用户后续指定的 `paper_v3.0`，从 `paper_v2.0` 建立。算法参考分支 `beta_select` 只读，核对时提交为 `f92edb021fd468e354f5767bb81ac745e0347cfe`。没有 merge 参考分支，也没有整体替换工程。

本轮恢复 Baseline、原 VQ-DenseO2O 和原 Sigmoid DC-STA；保留现有 LI-CopyBlend、统一实验入口、实例字段同步和 Validation 选模管理。配置及脚本目录仍叫 `paper_v2`，不代表可以复用上一轮训练结果。

## 1. 母版来源与迁移位置

| 对象 | beta_select 中的实际来源 | 新分支中的落点与处理 |
|---|---|---|
| Baseline 有效配置 | `configs/tomato/deimv2_l_baseline.yml`，合并其 include 后 DC/VQ 均关闭 | `configs/tomato/paper_v2/common.yml` + `baseline_native.yml`；逐字段核对共同配方 |
| VQ-only β=0.10 | `configs/tomato/vqonly_beta_sweep/vqonly_beta_010.yml` 及其 include 链 | `configs/tomato/paper_v2/vq_dense_o2o.yml`；DC 关闭，mask 同步开启 |
| DC-only | `configs/tomato/deimv2_l_dc_sta.yml` | `configs/tomato/paper_v2/dc_sta.yml`；恢复 Sigmoid 算法，数据路径统一改为 mask-aware |
| DINOv3-S/16、原 STA | `engine/backbone/dinov3_adapter.py`、`dinov3.py` | 保留原语义特征、STA、语义/细节拼接、1×1 投影和 SyncBN；DC 为明确开关 |
| Encoder、Decoder、Matcher | `engine/deim/hybrid_encoder.py`、`deim_decoder.py`、`matcher.py` | 检测算法源码与参考一致 |
| 原 MAL、定位与辅助损失 | `engine/deim/deim_criterion.py` | 继续使用原 DEIMCriterion；普通分支标签与 DN 分支标签仅作监督路由元数据 |
| 原 VQ 质量与监督 | `engine/deim/vq_quality.py`、`vq_criterion.py` | 恢复到同名文件，注册 `VQDEIMCriterion`；增加与当前日志兼容的分离诊断 |
| 原 DC 门控 | `engine/backbone/dc_sta.py` | 同名文件恢复 Sigmoid 计算，附优化器/EMA 检查及轻量 checkpoint 语义标记 |
| 原 CopyBlend、MixUp、Mosaic | `engine/data/dataloader.py`、`transforms/mosaic.py` | 保留原 RGB、bbox、类别、随机选择及调用时间表；保留实例字段同步修复 |
| LI-CopyBlend | 来源为 `paper_v2.0` 当前实现 | `engine/data/li_copyblend.py` 保留；只在明确启用 LI 的配置使用 |

不能直接把 `common_validation.yml` 当作 Baseline：该历史快照自身开启 DC-STA 和 VQ，且 VQ 的 β 为 0.25。实际 Baseline 子配置会覆盖这两个开关。本轮核对和测试使用覆盖后的有效 Baseline 配置。

## 2. 保留的数学语义

VQ 使用连续坐标积分图和 FP32 mask 积分：

\[
C=\frac{|M\cap B_{pred}|}{|M|},\qquad
P=\frac{|M\cap B_{pred}|}{|B_{pred}|},\qquad
q_{fg}=\sqrt{CP}.
\]

\[
q_{vq}=\operatorname{stopgrad}\left(q_{IoU}(0.90+0.10q_{fg})\right),
\qquad t_+=q_{vq}^{1.5}.
\]

积分的交集限制在 mask 画布内；Purity 分母保留预测框未截断的连续面积。缺失有效前景支持时保留 IoU 回退，未改变原边界定义。`q_IoU` 为普通 IoU，正式入口不允许用 GIoU 替代。普通 main/aux/pre/encoder 的正样本 MAL 接受校准；DN、DN-pre、负样本、匹配、query 选择及定位损失保留原路径。没有 GT-referenced 的 Coverage/Purity 比值，也没有推理时 mask 质量计算。

DC 使用原八维 RGB 退化统计、`8 → 16 → C_l` 通道 MLP 和 `10 → 1` 的 3×3 空间卷积：

\[
g_c=\sigma(f_c),\qquad g_s=\sigma(f_s),\qquad S'_l=S_lg_cg_s.
\]

两条门控同时启用时，初始化各门控为约 \(\sqrt{0.999}\)，乘积约 0.999；不是严格 1。没有残差 tanh 或额外 α=0.1。DC 只接受 RGB、语义特征和细节特征，不接受 GT mask。D1 的 mask 同步发生在共同数据管线中。

历史 GR-VQ 文件/结果不是正式模型。正式模型标识与组合使用 `vq_dense_o2o`；旧开关、旧模型名和 Residual-tanh checkpoint 不可作为新实验继续训练。

## 3. 共同训练协议及实际差异

有效 Baseline 配方、此前 `paper_v2.0` 公共配方和本轮公共配方在以下超参数上相同：

| 项目 | 全部九组共同值 |
|---|---|
| 训练 | 120 epochs；seeds 0、42、3407；640×640；3 类 |
| batch | Train 有效总 batch 32；Validation 总 batch 64 |
| Optimizer | AdamW；主学习率 0.0005；DINO 学习率 0.0000125；betas [0.9, 0.999]；weight decay 0.000125，沿用原参数分组 |
| 调度 | flatcosine；warmup_iter 2000；flat_epoch 60；no_aug_epoch 14；lr_gamma 0.5 |
| 增强 | policy epoch [7, 60, 106]；MixUp [7, 60]、概率 0.5；CopyBlend [7, 106]、概率 0.5；Mosaic 分支选择概率 0.5 |
| Matcher | epoch 88 切换；原 cost 权重和 IoU-order 参数 |
| EMA | 初始 decay 0.9999；warmups 1000；start 0 |
| 初始化 | 同一 DINOv3 预训练与 detector 预训练，保留原 head 类别适配逻辑；新增门控重新初始化 |

相对参考分支，实际保留的工程差异如下，全部模型一致应用：

1. **实例同步修复。** 原 Sanitize 没有同步全部 area/iscrowd 等字段，原 MixUp 也未同步全部实例元数据。本轮保留同步机制，不回退这些已确认的问题。过滤依据仍为原 bbox 合法性，mask 无效不删除合法 GT。Mosaic 的 image_id/orig_size 保留主图语义，避免误当作实例字段拼接。合成对照确认 RGB、bbox、类别、原 area 取值和随机数消耗保持一致。
2. **EMA 阶段 decay 生效。** beta_select 的 decay_fn 闭包捕获初始化值，后续修改 `ema.decay` 未必改变实际 EMA 更新。本轮保留 `paper_v2.0` 的 `set_decay` 修复、阶段状态保存及恢复。所有模型采用同一行为。
3. **统一选模。** 使用全程 EMA FP32 Mixed Validation AP；关闭评估 autocast/TF32，相同 AP 保留更早 epoch，`best.pth` 与 `selection.json` 对应。Test 不参与选择。阶段训练恢复与全程选模记录分别保留。
4. **管理字段。** 相对参考配置，print_freq 为 500 → 100、checkpoint_freq 为 5 → 10，另有新输出根与算法 suite 标记；这些不是训练算法因素。
5. **可视化目录保护。** 恢复 beta_select 的既有目录 `FileExistsError`，避免打开 data_vis 时清空历史可视化文件。正式配方仍默认不启用 data_vis。

因此，“Baseline 恢复”指检测结构、初始化规则、损失与基础增强算法得到参考对照；**不声称新实验的完整 120 轮训练轨迹与历史 beta_select 运行完全相同**，也不混用历史 AP。

## 4. 新旧结果隔离

新实验根目录：`outputs/paper_v2_beta_select_algorithms/`。

每个模型/seed 使用 `<model>/seed_<seed>/`；所有正式配置和运行记录带有 `algorithm_suite: beta_select_algorithms_v1`。AP、效率统计也使用此新实验根下的对应目录。已有 `outputs/paper_v2/` 及历史 tomato 实验目录保持原样，不删除、不覆盖、不重命名为新方法。

九组模型都应从共同预训练资源重新训练。不能把旧 GR-VQ 换名为 VQ-DenseO2O，也不能把旧残差门控的 DC-STA 结果当作本轮 Sigmoid DC-STA；Baseline 和 LI 的历史结果也不能拼接进本轮统一配方表。配置审计文件按用户要求单独保留在 `outputs/paper_v2/config_ablation_matrix.csv`，它是配置清单，不是旧结果复用。

## 5. 本节核对证据与边界

执行：

```powershell
python -m pytest tests/paper_v2/test_beta_baseline_reference.py tests/paper_v2/test_data_pipeline.py tests/paper_v2/test_data_sync.py -q --disable-warnings
```

本次结果：**36 PASS，0 FAIL，57.53 秒**。其中新增 15 项参考测试，复用既有 21 项数据路径与同步测试，没有重复实现 B0/B1 已有验收。

- 完整 DEIMv2-L 宽度、层数和 300 queries；双方使用相同的合成 128×128 CPU 输入。所有初始化 state tensor 严格相等；完整检测输出满足 FP32 `rtol=1e-6, atol=1e-7`。
- Matcher 在 epoch 0 和 88 的匹配索引一致；main/aux/pre/encoder 原 MAL、L1/GIoU 损失及关键梯度严格相等。
- 三个 seed 下，MixUp-only、CopyBlend-only、两者联合的 RGB、boxes、labels、area 和随机数状态与 beta_select 一致；Mosaic 合成结果一致。
- 既有真实注册的 640 数据管线在四个增强阶段验证 B0/B1 的图像、bbox、类别、原 MAL 和梯度等价；合成无效 mask 不删除合法 GT，实例字段长度保持一致。

这些测试只读取 Git 参考源码与临时合成样本。**真实 Train、真实 Test、服务器权重、完整训练：NOT RUN。** VQ/DC 专项及九组配置整体结果见 `06_算法正确性测试报告.md`；本节不把 Baseline 对照结果代替其他模块的独立验收。
