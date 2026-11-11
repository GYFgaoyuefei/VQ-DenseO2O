# Fig. 8：真实 Test 预测的可追溯定性比较

脚本：`tools/paper_v2/qualitative_fig8.py`。本地项目现名为 `VQ-DenseO2O`；服务器仍使用 `/root/autodl-tmp/DEIMv2-new`。脚本只读取 COCO 标注、RGB 图片及已保存的 detection JSON，不读取 pth，不调用训练或推理。不会改动正式 AP、原始预测及 Fig6。

## 文件与环境

同步以下代码到服务器，保持相对目录：

- `tools/paper_v2/qualitative_fig8.py`：命令行、绘图及审计。
- `tools/paper_v2/qualitative_fig8_data.py`：COCO ID/类别/来源检查、逐图匹配与候选排序。
- `tests/paper_v2/test_qualitative_fig8.py`：独立合成 fixture 回归测试；不用于论文。
- 本中文说明。

依赖 Python 3.10+、Matplotlib 和 Pillow，测试另需 pytest。使用项目已有环境；不依赖 PyTorch、CUDA 或 COCO evaluator。工程原 AP/数据集模块的光照词表和冲突检查规则用于这里的轻量解析；不导入其训练相关依赖。原项目未提供适用于这五组模型的独立绘图工具，所以图层使用 Matplotlib。

固定列顺序：GT / Input、B1、VQ、LI、DC、Full。其中 B1 为 `baseline_maskaware`，Full 为 `vq_dense_o2o_li_copyblend_dc_sta`。默认 seed=42、统一 score≥0.30；GT 没有置信度。

类别按 COCO `categories` 的名称关联 ID，再映射为 Immature（绿）、Half-ripe（橙）、Ripe（红）；绝不按类别 ID 大小或数组位置认定类别。遇到不能识别的名称直接报错，需核实实际名称后补充明确别名。

真实 Test 中的 `half` 已明确映射为 `Half-ripe`（橙色），支持大小写与首尾空格规范化；审计仍保留原始类别名 `original_name`。无需修改 COCO 标注、类别 ID 或预测 JSON。

## 1. 检查真实数据来源

每组读取 `{pred_root}/{model}/seed_42/predictions.json`。默认利用同目录 `metrics.json` 的 `dataset.annotation_sha256`、`split=test`、`model`、`seed`，核实它们绑定相同的 Test 标注。全部预测的 image_id/category_id 必须存在于 COCO；ID 重复、未知引用和来源不符不能绕过。允许某张图没有预测；不会因缺少检测而漏掉该图的 FN。

若导出时没有保留 `metrics.json`，优先从原评估目录取回。确实只有独立预测 JSON 时，可提供经人工核实的 `--dataset-manifest PATH`：键为模型完整目录名或 B1/VQ/LI/DC/Full，值包含该次预测所用 Test 标注的 `annotation_sha256`（可附 `split`、`model`、`seed`）。五组都必须有证据。不要为了让检查通过而给来源未知的预测填入当前文件摘要；仅 image_id 集合相同不能证明数据相同。

光照读取 images 元数据的 `illumination`、`illumination_label`、`source_illumination_id` 或 `light_condition`，支持自然光、人工光、微弱光、钠黄光以及项目明确的英文/整数标签。缺少时使用：

```bash
--illumination-map /root/autodl-tmp/dataset/tomato_coco/annotations/test_illumination_map.json
```

映射 JSON 的键是实际 image_id 字符串或完整 file_name，值是光照名称。未知标签/冲突不会被猜测；没有标签的图标记 unknown，不作为钠黄光局限性候选。不能根据 ID、文件顺序或图像亮度推断采集光照。

## 2. AutoDL：先生成候选排名与全景联系表

在现有 Python 环境执行：

```bash
cd /root/autodl-tmp/DEIMv2-new
python tools/paper_v2/qualitative_fig8.py \
  --mode rank \
  --annotations /root/autodl-tmp/dataset/tomato_coco/annotations/instances_test.json \
  --images /root/autodl-tmp/dataset/tomato_coco/images/test \
  --pred-root /root/autodl-tmp/DEIMv2-new/outputs/paper_v2_beta_select_algorithms/ap/test/raw \
  --seed 42 --score-thr 0.30 \
  --out /root/autodl-tmp/DEIMv2-new/outputs/paper_v2/fig8/seed42_rank
```

默认检查 400 张图。`--contact-count 12` 决定联系表展示数量，CSV 始终覆盖全部图片；增大该值可查看更多候选。只有确切使用其他数据规模或合成测试时才调整 `--expected-images`。

辅助匹配：在统一阈值下，按置信度降序逐个检测，与尚未匹配的同类 GT 取最大 IoU，IoU≥0.50 才成为 TP；其余为 FP，剩余 GT 为 FN。重复框不能匹配同一 GT。再将剩余检测与剩余异类 GT 一对一关联，记录疑似类别错误；这些错误仍属于 FP/FN，不再计一次 TP。crowd/ignore GT 不参加此诊断；未匹配常规 GT、但与同类 ignored GT 的 IoU≥0.50 的检测不计入 FP，相关数量记录在匹配结果中。

**这是用于解释单图的匹配诊断，不是 COCO AP。** 定位改善只比较双方共同匹配的同一个 GT，不能拿不同 GT 集合的平均 IoU 作改善证据。A 候选关注 VQ 相对 B1 的漏检、类别或定位改善；B 候选要求明确钠黄光，关注 Full 相对 B1/VQ 的退化。即使诊断提供支持，也只能称候选，不能自动确认科研结论。

输出：

```text
seed42_rank/
  fig8_candidate_ranking.csv          # 全部图片；五模型诊断、推荐原因、A/B支持及排序分数
  fig8_candidate_contact_sheet.png    # 全景，六列对齐，优先兼顾A/B候选
  Fig8_rank_audit.json                # 来源摘要、非法框、联系表实际ID及绘图参数
```

## 3. 人工确定两个真实案例

1. 打开 CSV 与联系表，核实推荐原因和真实目标；核对漏检、重复检测、类别错误以及 GT 是否合理。
2. A 选能说明 VQ 相对 B1 改善的实际图片；B 选钠黄光下 Full 的实际局限。若不存在支持证据，不强行选“成功/失败”案例。
3. 必须记录 COCO 的 `image_id`，不能用联系表中的行号或文件排序位置。
4. 优先先看全景；如需 ROI，以 GT 和需要解释的目标确定原图坐标。六列共享同一 ROI，不可为个别模型隐藏错误。
5. 检查正式导出的全景联系表与局部图，以及审计 JSON 中的匹配条目，再决定论文是否采用。

## 4. AutoDL：正式输出

下面两个 read 命令由研究者输入实际选定的 COCO ID，没有预设任何案例：

```bash
cd /root/autodl-tmp/DEIMv2-new
read -rp '输入人工复核后的 A 图 COCO image_id: ' IMAGE_A
read -rp '输入人工复核后的 B 图 COCO image_id: ' IMAGE_B
python tools/paper_v2/qualitative_fig8.py \
  --mode render \
  --annotations /root/autodl-tmp/dataset/tomato_coco/annotations/instances_test.json \
  --images /root/autodl-tmp/dataset/tomato_coco/images/test \
  --pred-root /root/autodl-tmp/DEIMv2-new/outputs/paper_v2_beta_select_algorithms/ap/test/raw \
  --seed 42 --image-ids "$IMAGE_A" "$IMAGE_B" --score-thr 0.30 --dpi 300 \
  --out "/root/autodl-tmp/DEIMv2-new/outputs/paper_v2/fig8/seed42_A${IMAGE_A}_B${IMAGE_B}"
```

需要局部图时，将实际 ID 和 `[x1,y1,x2,y2]` 写入 ROI JSON，然后在上面的命令增加 `--roi-json /绝对路径/fig8_roi.json`，并使用新的 `--out`。ROI 必须满足 `0≤x1<x2≤原图宽`、`0≤y1<y2≤原图高`；未指定的图使用全景。不同图片可以不同 ROI，同一行六列完全一致，图像始终保持原比例。

GT 虚线框，预测实线框和真实 score；统一白背景、DejaVu Sans 字体，图宽 7.2 英寸。PNG/TIFF ≥300 dpi，SVG 嵌入 RGB 栅格且保留检测框、标题和图例为矢量；不生成 PDF。

显示时只裁切视窗，不修改原预测框坐标。与 ROI 相交的框可以显示；图中“改善/局限性”标签的证据只取完全位于 ROI 内的框，避免用被裁掉的目标支持解释。正式标签还要求比较方 TP/FP/FN/类别错误没有相反方向的退化，共同匹配 GT 的平均 IoU 没有≥0.05 的退化；指标互有升降的情况在 CSV 保留推荐理由和 tradeoffs，标题使用中性的 Selected comparison。程序提供的是保守的标题保护，最终仍需人工复核。

输出：

```text
seed42_A<实际ID>_B<实际ID>/
  Fig8_Qualitative_Comparison.png
  Fig8_Qualitative_Comparison.tif
  Fig8_Qualitative_Comparison.svg
  Fig8_full_image_contact_sheet.png   # 同两个案例的原图全景六列比较
  Fig8_selected_cases.csv             # 两个案例×五模型，10行详细审计
  Fig8_render_audit.json              # 每个面板、来源hash、ROI、GT/匹配、非法框、输出像素
  Fig8_caption_en.txt
  Fig8_caption_zh.txt
```

非法框（非有限坐标、宽高≤0）及非法 score 不绘制、不参加诊断，但保留原记录位置和原因的审计。不会 abs、交换端点、修复坐标或重写预测 JSON。审计包含整份预测的非法记录，而逐案例 CSV 包含该图全部非法记录数（不限于阈值以上）。论文原 AP 仍来自已有正式评估。

已有同名 Fig8 文件会拒绝覆盖；换一个输出目录重做。rank 与 render 文件名不同，可放同一目录。不要将输出路径指向原图或预测目录。

## 本地测试

```powershell
cd D:\workspace\python_workspace\study\VQ-DenseO2O
& D:/ProgramData/anaconda3/envs/pytorch310/python.exe -m pytest tests/paper_v2/test_qualitative_fig8.py -q
```

测试临时生成带 SYNTHETIC TEST 标记的 RGB fixture、COCO GT 和保存预测，仅验证程序，不作为正式论文图。覆盖非连续 image_id、打乱类别 ID、xywh/IoU/一对一匹配、统一 ROI/阈值、非法框审计、数据集绑定、输出尺寸与 dpi、SVG 图层、图注证据以及禁止覆盖。

2026-10-23 本地结果：**25 passed in 8.79s**。Matplotlib 3.10.0、Pillow 11.1.0、pytest 8.3.5。合成导出验证为 2160×1003 像素（实际正式图高度随 ROI 宽高比确定）、300 dpi，SVG 含嵌入 RGB 和矢量图层。测试发现本机 LZW 编码器异常，因此 TIFF 使用无压缩 raw 格式；同时修复了边缘框标签被截断的问题。CLI `--help`、语法与差异空白检查通过，确认导入不加载 torch/engine。

同日补充真实类别别名 `half` 后：**28 passed in 11.91s**。rank/render 合成 fixture 已使用 `half` 并刻意设置为 ID=28，验证按名称映射为橙色 Half-ripe、SVG 标签为 `Half-ripe 0.90`、审计保留 `half`；另覆盖标准拼写、大小写及首尾空格。

真实 Test 筛选与正式论文图：尚未在服务器执行。以上数字只反映程序的合成测试，不是模型性能，也不是正式案例确认结果。
