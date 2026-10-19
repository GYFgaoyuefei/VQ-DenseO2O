# paper_v3.0：Mask 数据路径冗余优化

日期：2026-10-19。实施分支：`paper_v3.0`；修改前提交：`fa5dc028a5df4f7cf69b3c0fa209243a2f9bf717`。

本轮依据 [paper_v2.0 与 beta_select 静态审查](mask_pipeline_performance_comparison.md)，先在 v3 重新确认活动代码，再优化确定无用的复制和相邻重复检查。按用户要求不运行训练、真实数据测试、合成测试或性能基准；未访问 Train/Test，未修改历史实验结果。

## 已实施的四项优化

| 位置 | 修改前 | 修改后 | 保留的行为 |
| --- | --- | --- | --- |
| `engine/data/visible_geometry.py` / `translated_record` | clone 完整来源 mask，随后立即被平移 mask 覆盖 | 构造 record 时跳过 mask clone；直接创建独立的目标 mask 并复制相同 patch | 平移位置、二值支持、opaque/blend 行为、其它实例字段复制、valid 更新、合成 annotation ID 与 keypoints |
| `engine/data/dataloader.py` / MixUp | deepcopy 全批 targets，逐项被新拼接字典替换 | 用列表收集原有 `concatenate_instances` 结果 | RGB 运算、实例顺序、所有实例字段、mixup 权重、图像级元数据引用及 RNG 调用次序 |
| `engine/data/visible_geometry.py` / `resize_masks` | 抽到同尺寸也做 FP32 转换、nearest 插值和 dtype 回转 | 三维 uint8/bool mask 且目标空间尺寸相同、为正整数时，复制为独立连续普通 Tensor | 数值、dtype、输出独立存储、面积选择性更新、完整 alignment 检查；浮点和其它 dtype 继续走原 FP32 路径 |
| `engine/data/dataloader.py` / collate 出口 | `resize_masks` 已完整检查，出口立即重复检查相同状态 | 用局部 `alignment_checked` 标记跳过这一轮重复检查 | 每次 resize 仍完整检查所有目标；无 resize 或无 mask 的路径继续执行出口检查 |

同尺寸快路径仍复制一份 mask，未返回来源 tensor 的别名。这样避免后续原地修改污染 donor 或 Mosaic 缓存，同时去掉不必要的 FP32 中间张量和插值。零实例仍走原有 `new_zeros`；非法尺寸不进入快路径。该函数的 `area_from_mask` 更新和二值/shape/实例数/finite bbox/empty→invalid 检查均位于快路径之后，未被提前返回绕过。

多尺度分支中，`resize_masks` 返回之后到 collate 出口之间没有任何目标修改，因此合并的是同一状态的检查；标记是单次调用内的局部变量，不是跨 batch 的“已验证”缓存。未删除 dataset、Sanitize 或 CopyBlend 几何变化后的检查。

## 理论与正确性边界

- 模型、优化器、学习率、随机种子、增强概率及 epoch 时间表没有修改。
- bbox 合法性与 GT 保留规则、实例字段同步、mask 支持与混合透明度的区分没有修改。
- 原生 CopyBlend 的 image clone 和 targets deepcopy 仍保留，用于隔离 donor；只删除确认会被覆盖的额外复制。
- LI=false 的小型统计仍保留，因为它负责原生 CopyBlend 的新增实例诊断与零尝试批次分母；没有启用 LI 光照/碰撞算法。
- v3 正式使用 `VQDEIMCriterion`，历史 `gr_vq*` 不注册为正式模块。本轮未修改任何 VQ/GR-VQ、MAL、匹配、Coverage/Purity、IoU fallback 或 DC-STA 代码。
- 未引入原始解码缓存或跨分支积分图缓存。后者需要处理不同匹配、重复 GT、target 副本和每步失效，且增加 FP32 常驻显存，不属于本轮无需运行验证的最小修改。

## 能确认的计算量变化

| 操作 | 静态变化；不是耗时测量 |
| --- | --- |
| 普通非 Mosaic/非 CopyBlend 样本，默认两个 Sanitize，多尺度开启 | 完整 alignment 调用从 v3 修改前每图 6 次减至 5 次 |
| epoch 110，多尺度关闭 | 仍为每图 5 次；没有删除后期必要检查 |
| 每次成功原生 CopyBlend paste | 少一次会被覆盖的来源实例整幅 mask clone |
| 每个 MixUp batch | 少一次结果完全未使用的 targets deepcopy |
| 同尺寸 uint8/bool mask resize | 从 FP32 转换/插值/回转，改为原存储 dtype 的独立复制 |

上述调用数沿用审查报告的路径限定，各检查点分辨率和实例数不同，不能据此计算吞吐提升比例。没有实测数据支持将历史约 3 小时差额归因给这些修改，也不承诺具体训练时长。

## 验证记录

本轮只做静态验证，不导入或执行项目、torch 或测试模块。

| 项目 | 状态 | 依据 |
| --- | --- | --- |
| 两个生产文件的独立源码复核 | PASS（静态） | 核对 donor alias、empty/invalid、dtype、字段同步、RNG 与每条 collate 检查路径 |
| VQ 正式路径与历史 GR 边界 | PASS（静态） | 检查 resolver、模块注册及 criterion 调用；相关源码无修改 |
| 两个生产文件与新增测试源码的 Python 语法检查 | PASS（静态） | 仅内存 AST/编译检查，不执行模块 |
| 随机调用与 diff 检查 | PASS（静态） | AST 比较确认 dataloader 的 random 调用表达式及源码顺序不变；`git diff --check` 通过。此项不替代固定 RNG 运行比较 |
| 固定 RNG 的 RGB/GT/MAL 等价性 | NOT RUN | 复用现有 `test_data_pipeline.py` 与 `test_li_copyblend.py` 中的等价性用例，未重写或执行 |
| 同尺寸 mask 快路径回归 | NOT RUN | 在 `test_data_sync.py` 增补独立存储、valid/area 更新及异常保护用例源码 |
| 真实 Train、GPU 性能、优化前后单步比较 | NOT RUN | 遵守仅代码检查要求 |

旧测试 XML/JSON、旧性能 CSV 均未重新生成，不能把旧 PASS 当作本次修改的运行验证。历史审查报告保留原比较提交和当时结论，本文件单独记录 v3 后续实施。
