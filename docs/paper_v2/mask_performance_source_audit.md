# Mask 传播源码与历史日志证据审计

日期：2026-10-19。按本轮最新要求，仅检查代码及本地已有日志，不执行真实或合成性能测试，不访问 Test，不启动训练。本文件是主报告的源码证据附件；所有耗时归因和优化收益均未实测。

固定比较对象：

- `paper_v2.0`：`6da37427fb5627c5b8c8b32ef9905dfe83b1e186`
- `beta_select`：`f92edb021fd468e354f5767bb81ac745e0347cfe`

下面行号对应上述提交，不随后续文件修改更新。使用 `git diff`、`git show <SHA>:<path>` 和源码检索获得证据。未改动历史提交或训练结果。

## 1. 已确认的主要差异

| 检查项 | beta_select | paper_v2.0 | 源码确认的影响；不能替代实测 |
| --- | --- | --- | --- |
| Dataset 基类 | 自行准备检测字段和 mask | 继承原生 CocoDetection，原生 bbox/category 准备后添加 mask | paper 再做一次 COCO annotation 查询及 Python bbox 保留规则遍历；更明确保持原生检测路径 |
| instance_masks_only | 构造参数默认 True；加载后 pop foreground 两字段 | 删除该参数，根本不构造 foreground 两字段 | 不能说 paper 因默认 False 传播整图前景 mask；没有此变化 |
| foreground_mask | prepare 中先构造，随后才被 instance_masks_only 丢弃 | 不存在该构造逻辑 | paper 反而删除这部分分配/归并工作 |
| Sidecar identity 校验 | 每次 prepare 检查 image_id/category_id | 数据集初始化 read_sidecar 遍历校验 | paper 将这部分固定校验移出热路径 |
| 有效 mask 解码 | 有 segmentation 即先解码；valid=False 时另建零 mask | 仅 valid=True 时解码；否则零 mask；无效编码走 fallback | paper 没有新增同一实例重复栅格化；两边都没有原始解码结果缓存 |
| Sanitize | SynchronizedSanitizeBoundingBoxes 同步少数 mask 字段 | SanitizeBoundingBoxes 同步全部实例字段，SanitizeVisible 再执行全 mask 检查 | paper 新增两个 Sanitize 后的全像素检查，同时修复 area/iscrowd 等同步缺口 |
| assert_visible_alignment | 同样包含 binary 检查、bool 转换、nonempty 归约 | 操作类似，但新增多个调用点 | 调用次数差异是明确的重复工作来源，具体毫秒及瓶颈排名未测 |
| 普通 CopyBlend | 创建目标位置全零 mask 并填充 patch | 除上述工作，先 clone 源实例整幅 mask，随即覆盖 clone；新增元数据与检查 | 一幅完整源 mask 的 clone 被直接丢弃，可明确定位 |
| MixUp | deepcopy targets 后拼接 boxes/labels/area/mask 及少数字段 | deepcopy targets 后，用新拼接字典逐项替换全部拷贝 | paper 的 deepcopy 整体结果不再被使用；两版本的 mask 深拷贝再拼接均存在重复工作 |
| LI 开关 false | 无 LI 模块 | 跳过 LI 光照/碰撞/光度算法，但仍统计原生 CopyBlend | 不满足“完全零附加逻辑”；不能把 LI 主算法成本归到关闭状态 |
| mask batch resize | uint8→FP32 nearest resize→原 dtype | 同样转换；多一轮对齐检查和 area_from_mask 分支 | FP32 resize 分配是共有成本，不是 paper 独有公式变化 |

## 2. Dataset 与整图前景 mask

beta `engine/data/dataset/visible_coco_dataset.py:33` 的构造参数明确是 `instance_masks_only=True`，`load_item:109` 才 pop `foreground_mask`、`foreground_pixel_valid`。此前 `ConvertCocoPolysToMask:244` 已对每个有 segmentation 的实例解码；`:265` 根据 `foreground_complete` 构造前景并集及有效像素图，或两个 `[1,H,W]` 零图。因而 beta 的“实例 mask only”并未避免其生成成本。

paper 同路径 `:71` 不再接收 `instance_masks_only`，`:75` 固定让原生 converter `return_masks=False`。`:88` 的 `load_item` 在原生 bbox/category 后添加 sidecar 实例 mask；`:109–126` 每个保留实例最多解码一次并 stack；没有 foreground 两字段。无效 mask 不决定 bbox 保留，`:113–118` 对可处理的解码异常设置零 mask 和 valid=False。`:119` 有一次解码后的 nonempty 检查，`:137` 还对 stack 后的全部 mask 再检查。

paper `:92–100` 再次获取 annotation 并比对检测实例数。这个额外 Python 工作存在，但相对全 mask 的内存流量大小不能由源码确定。

## 3. Sanitize 与全像素重复扫描

beta `engine/data/transforms/_transforms.py:37–64` 的 labels getter 仅列出 `labels/visible_mask_valid/annotation_ids/visibility_group`。boxes 和 tv Mask 由 torchvision 的 Sanitize 一并筛选。它没有显式将 `area`、`iscrowd` 等普通 tensor 加入同一个筛选结果。

paper 同文件 `:38–54` 对 `instance_fields` 返回的所有实例 tensor 统一筛选，排除由 torchvision 自行处理的 boxes/masks；`:58–66` 在共享 bbox 筛选之后执行 `assert_visible_alignment`。因此两者使用原 bbox 有效性规则，但 paper 修复的是字段同步，不应为性能退回 beta 的未同步 area/iscrowd 行为。无效或空 mask 只更改 valid，不删除合法 bbox。

paper `engine/data/visible_geometry.py:52–65` 每次检查执行：

- 实例字段长度、bbox finite、mask shape/gradient 检查；这部分成本主要与实例数相关。
- `((masks != 0) & (masks != 1)).any()`：全 mask 两次比较、布尔中间结果及归约。
- `masks.flatten(1).bool().any(1)`：uint8 mask 的 bool 转换及每实例 nonempty 归约。

beta 同文件 `:17–32` 也做相同级别的像素检查；差异主要是调用位置增多，不是 beta 从未检查。

以下是正式配置调用路径的静态计数，不是 profiler 采样。前提是非空 mask 图像、普通 dataset 获取、Mosaic 使用配置中的 `use_cache=True`、不执行 LI 主算法。配置包含两个 Sanitize，且 `base_size_repeat=3` 令 batch multiscale 启用；是否抽到 640 不影响当前 resize 函数被调用。

| 完整 alignment 检查位置 | beta 每个目标图像 | paper 每个目标图像 |
| --- | ---: | ---: |
| load_item 原图 | 0 | 1 |
| 两个 Sanitize | 0 | 2 |
| dataset __getitem__ 最后 | 1 | 1 |
| 普通 CopyBlend 有成功 paste 的目标 | 0 | 额外 1 |
| collate 的 resize_masks 内，epoch<106 | 0 | 1 |
| collate 返回前 | 1 | 1 |

因此不含成功 CopyBlend 时，epoch 3/20/80 为 beta 2 次、paper 6 次；epoch 110 为 beta 2 次、paper 5 次。这里各次 mask 的尺寸和实例数并不相同，不能用 6/2 推断耗时变为三倍。尤其首次检查在原始分辨率，MixUp/Mosaic 后实例数可能增加。

可直接确认的同状态重复：`resize_masks:98` 检查后返回 collate，`dataloader.py:396–397` 在没有任何 mask 修改的情况下再次检查相同目标、相同尺寸。保留最终出口检查即可避免此处重复扫描，但本轮未修改代码、未验证收益。

`engine/data/transforms/functional.py:crop` 也新增检查，但正式 Compose 中使用的是 `torchvision.transforms.v2.RandomIoUCrop/Resize`，并非该备用函数。不能将备用函数调用算入上述默认热路径。

## 4. Mosaic、Crop、Resize、MixUp、CopyBlend 的分配

- Mosaic：两个版本均先 resize tile、缓存最多 50 张、取四份目标 clone、逐实例 pad 至 mosaic 大小、torch.cat、RandomAffine，再经过正式 640 Resize。paper 改为 `concatenate_instances`，保留图像级字段并拼接全部实例字段；没有新增一轮实例 mask rasterization。`mosaic.py:41` 为 Mask 显式使用 fill=0；实际正式 RGB fill 本来也为 0，不能据此认为修改增强概率。
- Crop：正式 RandomIoUCrop 对 tv Mask 进行几何同步；两个版本下游 Sanitize 均会依据 bbox keep 选择 mask 行。paper 扩大同步字段范围，不改 bbox keep 规则。torchvision 内部具体 dtype/内存行为取决于训练时版本，历史环境缺失时不能断言它完全一致。
- Resize：两个版本 `visible_geometry.resize_masks` 都调用 `F.interpolate(masks[:,None].float(), mode='nearest')` 再回转 storage dtype；至少产生 FP32 输入和 resize 输出。paper 额外计算 `area_from_mask` 选中实例面积，普通 sidecar 实例初值全 False；不能把所有普通实例面积重算当作默认行为。
- MixUp：paper `dataloader.py:188` 深拷贝整个 targets，`:192` 用 `concatenate_instances([original, shifted])` 逐项替换；新字典从原始 targets 构造，完全未使用这次 deepcopy 内容。这是明确的无用内存复制。beta 本来也深拷贝，随后将 mask/boxes 等替换为拼接 tensor，因此不能将整项成本认作 paper 新增。
- CopyBlend：paper `visible_geometry.py:105–106` comprehensions 包含 masks 的 `clone()`；`:108–110` 另建全零目标 mask，填 patch 后覆盖 `record['masks']`，第一份 mask clone 不再使用。beta `copyblend_mask:66–71` 直接建全零目标 mask，只 clone 少量向量字段，没有该丢弃的 mask clone。
- 原生 CopyBlend 在两版本都复制目标 images/targets，以便在输出中原地贴图且不污染 donor。不能一概删除这项 deepcopy；MixUp 的可删拷贝与此不同。

## 5. LI-CopyBlend 关闭时的真实行为

paper `dataloader.py:207` 有明确 enabled 条件，false 时不会调用 `LICopyBlend.__call__`，所以不执行光照候选筛选、局部光度比较、碰撞重试或整批 RGB 反归一化。

但 `:175` 仍构造 Counter，普通 CopyBlend 记录 attempted/accepted，`:376` 每批调用 `record_stats`。`engine/data/li_copyblend.py:114–124` 会整理统计、更新 Counter，并向第一个 target 放入小型 `li_copyblend_stats` tensor，包括未发生增强的批次。`engine/solver/det_engine.py:129–130` 仍将此统计累加。它是附加逻辑，但没有全 mask 扫描，具体成本未测。

关闭 LI 后，dataset 仍生成 illumination/source/train/blend_opacity/area_from_mask 字段，它们需要随实例拼接/筛选，并参与 target 到 GPU 的迁移；这主要是额外小 tensor 管理。实例 mask 本身仍是大项传输候选。

## 6. beta=0 的对照含义与时间表

beta `configs/tomato/vq_ablation/mask_control_beta000.yml` 明确设置 `vq_dense_o2o.enabled=true, beta=0`；`configs/tomato/vqonly_beta_sweep/vqonly_beta_000.yml` 相同。resolver 因 enabled=true 打开 VisibleMaskCocoDetection 和 SynchronizedSanitize，`VQDEIMCriterion.get_loss:29` 因 `self.beta == 0` 提前调用原生 loss。因此 beta=0 是“传播 mask，但不执行 VQ 积分图”的控制路径。具体 8:00:28 日志用了哪一个 YAML 尚未找到，不能凭名称完成 run 绑定。

已检查两边正式配方的共有增强阶段：

| epoch | Mosaic | ZoomOut/IoUCrop、Photometric | MixUp | 普通 CopyBlend | batch multiscale |
| --- | --- | --- | --- | --- | --- |
| 3 | 关闭 | 关闭 | 关闭 | 关闭 | 开启 |
| 20 | 按 mosaic_prob=0.5 抽样 | Photometric 按概率；ZoomOut/IoUCrop 仅非 Mosaic 分支 | 按概率抽样 | MixUp 未选中时按概率抽样 | 开启 |
| 80 | 关闭 | 按各操作规则 | 关闭 | 按概率抽样 | 开启 |
| 110 | 关闭 | 关闭 | 关闭 | 关闭 | 关闭 |

RandomHorizontalFlip/固定 Resize/Sanitize 不在关闭的 policy_ops 内，后期仍执行。以上 epoch 为代码实际 `dataset.epoch`，不是人为从 1 开始重新编号。增强几何与概率本轮未改。

## 7. 日志、硬件和归因边界

检索范围为本地 `outputs/`、`docs/` 及两个比较提交的受控文件路径。未找到用户给出的五次真实训练日志、各 run 的 resolved config、启动命令或硬件 manifest；不能独立复核 2:54:53、11:01:28、11:39:18、8:00:28、8:06:32。

- 本地 `docs/paper_v2/08_训练前最终检查报告.md` 明确记载当地 RTX 4060 Laptop GPU，真实正式 batch32 Train 测试 NOT RUN。它不证明 AutoDL 五个 run 的硬件。
- `outputs/paper_v2/efficiency/efficiency_protocol.json` 与效率文档指定 RTX PRO 6000 Blackwell Server Edition，是部署评测协议，不能当成五次训练的实际硬件记录。
- `engine/solver/det_solver.py:255–257` 的 Training time 是训练循环整体 wall-clock；其中包含 validation、checkpoint 等工作。不能从这个总时长直接推出 DataLoader 或 criterion 占比。
- GPU 型号、CPU/内存配额、存储介质/负载、PyTorch/torchvision/CUDA、AMP/TF32、验证方式、checkpoint 开销均未与这五条时长绑定。

所以当前证据能确认“哪些重复扫描/拷贝在代码中发生”，不能确认“哪个操作耗时最多”、8 小时和 3 小时差额的真实分项、GR-VQ 占比或优化前后毫秒。用户最新要求仅看代码，本轮上述性能指标和两分支固定随机状态的实际正确性比较均为 **NOT RUN**，没有以旧测试或本地合成结果替代。

## 8. 可保持训练语义的候选优化边界

优先级是静态重复工作明显程度，不是实测耗时排名：

1. 移除 `translated_record` 中立即被覆盖的源 mask clone，保留新目标 mask、向量元数据 clone、位置平移、invalid fallback 和最终对齐保护。
2. 移除 MixUp 被全量替换的 deepcopy，保持拼接次序、image metadata、mixup 权重及 RNG 调用顺序；不要连带去掉原生 CopyBlend 对 donor 保护所需的 deepcopy。
3. 让 batch resize 之后的最终出口承担一次完整像素检查，避免在无变更的相邻调用点重复；仍保留 shape/实例字段与非空有效性保护。对原图或中间几何状态移除检查需另行验证，不能笼统删除所有检查。
4. 原始 sidecar 解码缓存如将来引入，只存不可变原始结果；每次增强取得独立可写副本，不把增强结果放入原始缓存。本轮不实施缓存，也不声称缓存一定加速。
5. 积分图跨普通 loss 分支复用需要与 criterion 审计一起确定边界；不能改 Coverage/Purity/IoU fallback 或 MAL 目标数学定义。

本附件未修改生产源码、训练配置、数学定义或历史结果。
