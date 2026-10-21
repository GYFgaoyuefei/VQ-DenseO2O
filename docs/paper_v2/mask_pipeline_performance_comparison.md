# paper_v2.0 与 beta_select：Mask 传播静态性能审查

日期：2026-10-19。状态：**源码审查完成；性能测量及正确性执行均 NOT RUN（按用户最新要求）**。

本轮范围已由“真实 Train 短程测试”收窄为“只检查代码”。未连接 AutoDL，未访问 Train/Test 数据，未加载模型、运行合成测试、微基准或训练；未改生产源码、配置、Loss 数学定义或历史实验结果。下文的调用次数是源码推导，历史时长是用户提供的记录，均不冒充本轮测量。

## 1. 比较对象与结论

工程：`D:\workspace\python_workspace\study\DEIMv2-new`。当前审查分支为 `paper_v2.0`。

| 分支 | 固定提交 |
| --- | --- |
| paper_v2.0 | `6da37427fb5627c5b8c8b32ef9905dfe83b1e186` |
| beta_select | `f92edb021fd468e354f5767bb81ac745e0347cfe` |

通过 `git diff`、`git show`、源码调用关系和配置继承检查，确认以下问题：

1. **paper 增加了多个全实例 mask 像素检查点，其中 resize 后与 collate 出口是相同状态的连续重复检查。** 普通样本在多尺度阶段，完整检查调用由 beta 的 2 次变为 paper 的 6 次；最后无增强阶段为 2 对 5。次数不等于性能倍数。
2. **paper 原生 CopyBlend 存在确定无用的完整源 mask clone**：随后被新建的平移 mask 覆盖。MixUp 也有被全部替换的整批 deepcopy；后者在 beta 中已有类似浪费，不能全算作新增成本。
3. **没有发现 paper 新增整图 foreground mask。** beta 的 `instance_masks_only=True` 在构造后才丢弃 foreground 字段；paper 已直接取消构造。也未发现 paper 对同一保留实例新增重复栅格化。
4. **LI=false 跳过增强主算法，但仍执行小型统计与元数据维护。** 不能把关闭状态理解为零附加逻辑，也不能把光照分析/碰撞重试成本算到该状态。
5. **普通分类分支重复构造积分图并非 GR-VQ 新增。** beta VQ 也重复构造，且有更多逐步 GPU 标量读取；GR 还有减少同步和缩小 IoU 矩阵的改动。

以上足以定位冗余代码，但不能确定“耗时最大的操作”或把训练差额分摊为小时。详细 dataset、transform、collate 证据见 [源码审计附件](mask_performance_source_audit.md)。下述源码行号均对应固定提交。

## 2. 数据路径与全 mask 扫描

| 项目 | beta_select | paper_v2.0 | 判断 |
| --- | --- | --- | --- |
| Dataset | 自行准备检测字段及 sidecar mask | 复用原生 CocoDetection 的检测字段，再附加 mask | paper 有第二次 annotation 查询和保留实例遍历；真实耗时未知 |
| instance_masks_only | 默认及 resolver 均为 True | 参数已移除；仅构造实例 mask | 不是 True→False 的回归 |
| foreground_mask | prepare 中构造，load_item 再 pop | 不构造 | paper 删除了此项开销 |
| 原始 mask 缓存 | 无 | 无 | 每次访问重新解码；Mosaic 的增强缓存不是原始解码缓存 |
| Sanitize | 同步 labels/valid/annotation_ids/visibility_group 等少数字段 | 同步全部实例字段，再做完整 mask 检查 | 同步修复应保留，像素扫描可审查合并 |
| batch mask resize | uint8→FP32→nearest→原 dtype | 相同，增加面积分支和完整检查 | 主要转换分配两边共有 |
| LI=false | 无 LI 模块 | 不调用 LI 主算法；继续统计原生 CopyBlend | 有附加逻辑，尚无成本测量 |

`engine/data/visible_geometry.py:52–65` 的 `assert_visible_alignment` 不只是 shape 检查：`((masks != 0) & (masks != 1)).any()` 遍历全部像素并构造布尔中间结果；`masks.flatten(1).bool().any(1)` 再计算非空支持并更新 `visible_mask_valid`。对 uint8 mask，bool 转换也分配张量。

以下计数限定为正常返回的普通非 Mosaic、非 CopyBlend 样本，默认两个 Sanitize，非空实例 mask；不包括逐实例解码时的 nonempty 检查，也不把一次函数调用误称为一次内存扫描：

| 完整 alignment 调用位置 | beta | paper |
| --- | ---: | ---: |
| 原图 load_item 末尾 | 0 | 1 |
| 两次 Sanitize | 0 | 2 |
| dataset __getitem__ 末尾 | 1 | 1 |
| collate resize_masks 内，多尺度开启时 | 0 | 1 |
| collate 出口 | 1 | 1 |
| 合计：epoch 3，以及 epoch 20/80 的上述普通路径 | 2 | 6 |
| 合计：epoch 110，多尺度关闭 | 2 | 5 |

paper 的调用点：`visible_coco_dataset.py:137,143`、`transforms/_transforms.py:66`、`visible_geometry.py:98`、`dataloader.py:397`。成功 CopyBlend 的目标还会在拼接后增加一次检查。Mosaic 正式配置使用缓存，三张 donor 从缓存读取，不能错误地将 dataset 加载检查乘以四。

各检查点的分辨率、实例数不同。第一次在原图上，第一次 Sanitize 可能在 ZoomOut/Mosaic 中间画布上；MixUp 后实例数也可能增加。**6/2 不能解释为三倍耗时。** 可直接确认的冗余是 `resize_masks` 检查之后，到 collate 出口之间没有 mask 修改，却再次检查同一份数据。

## 3. 分配、拷贝和传输

| 操作与证据 | 静态发现 | 优化边界 |
| --- | --- | --- |
| CopyBlend，`visible_geometry.py:105–110` | `instance_fields` 包含 masks，先 clone 一整幅源实例 mask，再用新零图和平移 patch 覆盖 | 可避免被覆盖的 mask clone；须保留其它字段复制、平移、valid 和 donor 隔离 |
| MixUp，`dataloader.py:187–192` | deepcopy 全 targets，然后逐项用从原 targets 拼接的新字典替换 | 整次拷贝结果未使用；保留拼接顺序、权重和 RNG 顺序。beta 也有 mask 深拷贝后替换，属共有浪费 |
| 多尺度 resize，`visible_geometry.py:90–98` | 两边即使选中现有 640 尺寸也执行 FP32 转换与 nearest 插值 | 同尺寸快路径是候选；仍需必要对齐保护、面积和有效性更新 |
| Mosaic，`transforms/mosaic.py` | 两边都有 resize、缓存目标 clone、pad、cat、affine；每 worker 缓存最多 50 张已缩放样本 | 不能删除保护缓存的 clone；不得把缓存增强结果当不可变原始数据 |
| 原生 CopyBlend，`dataloader.py:243–244` | 两边都 clone images/deepcopy targets，再贴图 | 这份副本保护 donor，不同于 MixUp 的废弃副本，不能一并删除 |
| H2D，`solver/det_engine.py:51–52` | 对所有 target tensor 调用 `.to(device)`，包括 B1 原生 criterion 不使用的 masks；未显式传 non_blocking | 两个 mask 路径都有 mask 传输；不能把总时差全归为 CPU；调整传输策略需要另行验证 |

paper `area_from_mask` 对普通 sidecar 实例初值为 False；不能把 resize 中的面积分支描述成“每次重算所有实例 mask 面积”。正式 Crop 是 torchvision v2 `RandomIoUCrop`，备用 `transforms/functional.py` 的 crop/resize 不能加入默认调用数。

LI 开关在 `dataloader.py:207` 阻止调用 `LICopyBlend.__call__`。关闭时仍建立 Counter、整理计数，经 `li_copyblend.py:114–125` 给第一个 target 附加小型统计 tensor，并在训练循环累积。光照筛选、局部光度分析、碰撞重试和 LI 像素混合不执行。额外实例字段会随几何处理、拼接和传输继续传播，其成本未测。

## 4. VQ 与 GR-VQ criterion

paper 已将 `engine/deim/vq_quality.py`、`vq_criterion.py` 替换为 `gr_vq_quality.py`、`gr_vq_criterion.py`，审查覆盖新旧实现。

实际 L 配置 `num_layers: 4`，训练输出 main 1、aux 3、pre 1、encoder 1，对应 **6 个普通 MAL 分支调用**。证据：`configs/deimv2/deimv2_dinov3_l_coco.yml:31`、`deim_decoder.py:481–486,580–583`、`deim_criterion.py:318,332,346,371`。DN/DN-pre 在两种 VQ 实现中均不做 mask 质量计算。

| 项目 | beta VQ | paper GR-VQ |
| --- | --- | --- |
| beta=0 | 检查 mask 前直接走 native loss | 同样提前返回 |
| 积分图复用范围 | 每普通分支、每图重新选取匹配 mask 并构建 | 相同，无跨分支复用 |
| 默认 chunk_size | 32 | 16 |
| 每 chunk 积分图 | 一次二维积分图，两次 cumsum | 相同；GT 参考复用该 prefix |
| 矩形积分查询 | prediction 查询 | prediction + GT 查询，再算相对 Coverage/Purity |
| 非二值/非有限 mask | 抛错 | 标记无效并回退 IoU |
| 诊断记录 | 每分支循环各组，直接提取 Python 标量 | GPU tensor 累积，epoch 边界统一转 CPU |

对非空匹配图像，prefix 构造次数分别为 `Σ分支 Σ图像 ceil(Nmatched/32)` 与 `/16`。这代表静态算子调用规模；不代表处理总像素数翻倍，也不意味着每个 GT 固定处理六次，因为分支匹配及 Dense O2O assignment 可能变化或重复。

GR 有确定的重复算术：`gr_vq_quality.py:83,125` 先计算 pixel FP32 IoU 和 q_grvq；criterion `:78–84` 为保持原生 MAL 的坐标转换 dtype/AMP fallback 语义，又计算 native IoU 并覆盖这些结果。这只是 O(N) 的额外算术，无法据此称为主要瓶颈。公开质量函数仍需返回 IoU，未来不能直接删掉 API 输出或用 pixel FP32 IoU 取代 native fallback。

也必须保留方向相反的证据：

- beta `vq_criterion.py:65–78` 每普通分支六组诊断，每组两次 int 和八次 float 标量转换，当前六分支合计 **360 次 Python 标量提取/step**（VQ 激活时）。CUDA tensor 的标量读取是同步点；`vq_quality.py` 还通过 Python 条件读取 reduction 结果。GR 已把诊断改为 tensor 累积，仅 epoch 汇总转 CPU。此数字是静态调用数，不是同步耗时实测。
- beta 在没有传入质量 `values` 时，对整个 batch 的匹配框构造 `N×N` IoU 矩阵再取对角线；GR 在各图内计算，矩阵规模为 `Σnᵢ²`。配置/分支是否提供 values 必须考虑，不能把该分配宣称为 beta 无条件发生。

因此不能单凭新增 GT 查询和较小 chunk 宣称 GR 一定慢于旧 VQ。B1 没有启用 GR；beta 的 Mask-aware beta=0 也提前返回 native loss。这两个控制路径的差额不应归因于 GR 积分图。

## 5. 增强阶段与计时解释

代码中的 epoch 是 `dataset.epoch`，不额外加一。正式配方保持 `[7,60,106]`，MixUp `[7,60]`，CopyBlend `[7,106]`；batch 32、worker 4 是源配置及用户提供的条件，未与历史 run 的 resolved config 绑定。

| epoch | dataset 重增强 | collate 增强 | batch multiscale |
| --- | --- | --- | --- |
| 3 | Mosaic/Photometric/ZoomOut/IoUCrop 关闭 | MixUp/CopyBlend 关闭 | 开启 |
| 20 | Mosaic 按 0.5 抽样；ZoomOut/IoUCrop 仅非 Mosaic 分支；Photometric 按自身规则 | MixUp 抽样，未选中才抽 CopyBlend | 开启 |
| 80 | 无 Mosaic；Photometric/ZoomOut/IoUCrop 按自身规则 | 无 MixUp；CopyBlend 抽样 | 开启 |
| 110 | 上述重增强关闭；HFlip、固定 Resize、两个 Sanitize 仍执行 | MixUp/CopyBlend 关闭 | 关闭 |

配置证据：两分支正式 tomato 配方、`configs/deimv2/deimv2_dinov3_l_coco.yml`、`configs/base/dataloader.yml`、`transforms/container.py:81–98`。epoch 20 的 MixUp 与 CopyBlend 是 if/elif，不是两个独立增强。

目前的日志不足以区分用户要求的所有阶段：

| 阶段 | 源码能够确认的范围 | 本轮耗时 |
| --- | --- | --- |
| dataset_load / image_transform / mask_transform / sanitize | worker 内读取、解码、RGB 与 mask 几何处理；部分同一个 torchvision 操作同时处理两者 | NOT RUN |
| collate / copyblend / mask_resize | worker 内批拼接和增强，后两项是 collate 的子过程 | NOT RUN |
| data_wait | 主进程等待下一批数据；预取可隐藏部分 worker 时间 | NOT RUN |
| h2d | samples 及每个 target tensor 到 device | NOT RUN |
| forward / criterion / backward / optimizer | 训练迭代内对应调用 | NOT RUN |
| total_step / 整体吞吐 | 必须直接测连续训练步 wall-clock；不能由上述局部时间相加 | NOT RUN |

`engine/misc/logger.py:216–218` 的 `data:` 是主进程取得下一批的等待时间，不等于 dataset+transform+collate 的全部 CPU 时间；`time:` 包括等待与训练执行，未逐阶段 CUDA synchronize。worker 预取与 GPU 计算可重叠，包含关系也使分项不可直接相加。

## 6. 历史时长与硬件证据

| 用户提供的实验记录 | 总时长 | 与指定对照的算术差额 |
| --- | --- | --- |
| paper B0 seed42 | 2:54:53 | — |
| paper B1 seed42 | 11:01:28 | 比 paper B0 多 **8:06:35** |
| paper GR-VQ seed0 | 11:39:18 | 比 paper B1 seed42 多 **0:37:50**；不同 seed |
| beta Mask-aware beta=0 seed42 | 8:00:28 | paper B1 比其多 **3:01:00** |
| beta VQ beta=0.10 seed42 | 8:06:32 | 比 beta Mask-aware 多 **0:06:04** |

以上只是对用户提供总时长做减法。本地未找到可绑定这些 run 的原始日志、resolved config、启动参数或硬件记录，不能独立确认其 GPU、CPU、内存、磁盘、PyTorch/torchvision/CUDA、AMP/TF32 和并发负载相同。部署效率协议中的 RTX PRO 6000 Blackwell Server Edition 不能充当这五次训练的硬件证明；本地旧验收中的 RTX 4060 Laptop 也不属于它们。

`det_solver.py:101,255–257` 的 Training time 包含训练循环内 validation、checkpoint 等活动，不仅是训练 step；循环开始前的初始 validation 不在此计时区间。paper 的 validation 显式启用 FP32/关闭 TF32 的上下文（`det_engine.py:167–192`、`det_solver.py:159`），beta 没有对应的强制协议，这是比较总时长时还需排除的环境/流程因素，尚未证明它贡献了多少时间。

## 7. 正确性：已审查，不宣称测试通过

| 检查项 | 静态结论 | 执行状态 |
| --- | --- | --- |
| RGB、boxes、labels 与合法 GT 保留 | paper 从 native dataset 起步，Sanitize 复用原 bbox predicate；invalid mask 只变更 valid，不作为删除合法 bbox 的条件 | NOT RUN |
| masks / valid / annotation_ids | 通过共同筛选、拼接、平移及对齐检查传播；opaque 与 blend 语义区分 | NOT RUN |
| area / iscrowd | paper 将其纳入统一同步；beta Sanitize 未同步 area/iscrowd，beta MixUp 已拼接 area、但未拼接 iscrowd，不能声明两分支全部字段相同 | NOT RUN |
| source_illumination_id | paper 增加并传播；beta 旧路径未统一提供，不能直接假定字段集合一致 | NOT RUN |
| B0/B1 固定 RNG 等价 | 代码有共享 native 处理及已有测试；未比较真实图片或执行固定随机序列 | NOT RUN |
| Coverage/Purity/IoU fallback/MAL | 查看现有几何参考和 criterion 测试；beta=0、普通分支限定、detach、native AMP fallback 均有覆盖 | NOT RUN |
| 两分支优化前后等价 | 本轮没有实施优化 | NOT RUN |

现有测试源包括 `tests/paper_v2/test_gr_vq.py` 和 `test_gr_vq_criterion.py`：连续像素积分、Coverage/Purity、越界面积、invalid fallback、chunk 等价、beta=0 全 loss/gradient、DN/定位分支不变、matcher 不变、真正 BCE 正目标及 AMP 原生 IoU。beta 有 `test_tomato_innovation.py`、`test_vq_beta_zero_parity.py`、`test_vqonly_validation.py`。**存在测试代码不等于本轮 PASS，也不能用旧结果替代两分支当前执行比较。**

特别注意：beta 未同步 area 时，后续按 area_threshold 筛 donor 可能关联错误实例；paper 的字段同步修复可能纠正这种旧行为。因此不能为了复现旧速度而恢复错位，也不能保证固定 RNG 下两个历史分支的每一步增强结果完全一致。

## 8. 候选优化与八项问题的回答

以下按冗余确定性排序，不是耗时排行榜，均**未实施**：

1. 避免 CopyBlend 立即废弃的源 mask clone，删除 MixUp 被完全替换的 deepcopy。保持 RNG 消耗、拼接顺序和 donor 隔离。
2. 合并 resize 与 collate 出口相邻的重复全像素检查。保留实例字段长度、mask shape、finite bbox、二值支持与 empty→invalid 保护；不得把所有断言直接删除。
3. 为同尺寸 mask resize 增加快路径，但保留输出 dtype、必要面积/valid 更新。验证职责可分成轻量结构保护与边界处完整像素检查；中间检查移除需要另行正确性验证。
4. 若未来缓存，只缓存不可变的原始解码支持，并给增强独立可写副本。跨 loss 分支复用 prefix 需按 batch 内唯一实例组织、处理重复匹配与不同索引，控制显存；不得缓存预测相关质量值或跨 batch 复用。此项比删除废弃拷贝风险更高。
5. GR 的重复 IoU 可以研究内部接口复用 native IoU，必须保留公开 API、AMP fallback 和原生 MAL 目标，不得改数学定义。

| 原问题 | 本轮回答 |
| --- | --- |
| 1. 两分支 Mask 源码差异？ | 已定位：新增完整扫描与实例同步、废弃 clone、统计路径；删除 foreground 构造；共有 resize/Mosaic/积分图工作详见上文 |
| 2. 哪个操作额外耗时最大？ | 未测，无法排名。全像素扫描和废弃 mask clone 是确定的额外工作，不是已测最大瓶颈 |
| 3. B1 为何比 B0 慢约 8 小时？ | B1 增加实例解码、dense mask 几何/复制/检查/传输；B1 无 GR。不能将 8:06:35 全部定量归给其中一项 |
| 4. paper 为何比 beta 慢约 3 小时？ | paper 的额外扫描和 CopyBlend 源 mask clone 是源码差异；同时有删除 foreground 等减负变化。无法据源码证明 3:01:00 的真实来源 |
| 5. Mask 与 GR 分别多少？ | 未测。paper GR 与 B1 的 0:37:50 来自不同 seed 的总体时长，不是 GR criterion 的独立耗时 |
| 6. 哪些优化可保持语义？ | 删除废弃副本、合并同状态重复校验、同尺寸 resize 快路径是候选；必要同步、valid 和 native MAL 语义必须保留 |
| 7. 优化前后单步提升？ | NOT RUN；本轮未改生产代码，无可报告的 ms 或百分比 |
| 8. 所有正确性测试通过？ | 本轮 NOT RUN；仅审查现有实现与测试覆盖，不能标 PASS |

输出 `outputs/paper_v2/mask_performance_profile.csv` 保留 A 原生、B beta mask、C paper mask、D paper mask+GR 与四个 epoch 的 16 条条件，状态统一 `NOT_RUN_PER_USER`，各 ms 和吞吐列留空、采样数为 0。它是明确未执行的条件清单，**不是性能结果表，不能用于论文速度结论**。
