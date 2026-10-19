> 历史文档（paper_v2.0）。本轮 paper_v3.0 已恢复原 VQ-DenseO2O 与 Sigmoid DC-STA；请从 [当前文档入口](README.md) 阅读新说明。以下旧算法、旧命令和旧测试结论仅作追溯，不能用于本轮新实验。

# Test Mixed AP与四光照AP统一计算

更新日期：2026-10-19。按用户确认，默认用于最终Test结果。脚本只评估已完成训练、由EMA FP32 Mixed Validation AP选出的`best.pth`，不训练、不重新选模、不用Test选择checkpoint。Validation与Test使用不同报告目录。

## 一条命令完成五组×三个seed

把本次代码同步到AutoDL后，在服务器已有环境执行：

```bash
cd /root/autodl-tmp/DEIMv2-new
source scripts/paper_v2/autodl/env.sh
bash scripts/paper_v2/autodl/evaluate_and_summarize_ap.sh cuda:0 independent
```

默认读取`outputs/paper_v2/<model>/seed_<seed>/`，模型为B0/B1/G1/L1/D1（baseline_native、baseline_maskaware、gr_vq、li_copyblend、dc_sta），seed为0、42、3407。Test数据根目录取`TOMATO_DATA_ROOT`，已配置为`/root/autodl-tmp/dataset/tomato_coco`；使用`images/test`和`annotations/instances_test.json`，无需mask sidecar或原始预训练权重。

每个模型/seed只推理一次，保存同一批COCO检测结果，再分别对全量Test和四个真实光照子集运行COCO bbox evaluator。Mixed AP在全量Test上直接计算，不能由四光照AP平均得到。推理使用原生预处理及postprocessor、FP32、autocast=false、TF32=false；COCO AP采用IoU=0.50:0.05:0.95、maxDets=[1,10,100]。

## 输出

默认目录：`outputs/paper_v2/ap/test/`。

|文件|内容|
|---|---|
|ap_by_seed.csv|每模型/seed的Mixed COCO指标及Natural、Artificial、Low-light、Sodium-light AP，均为百分数|
|ap_summary.csv|每模型三个seed的均值、样本标准差（ddof=1）及完整性状态|
|ap_protocol.json|评估口径、数据来源与汇总完整性|
|evaluation_attempts.json|本次各模型/seed的执行、复用、缺失或失败原因|
|raw/&lt;model&gt;/seed_&lt;seed&gt;/metrics.json|该模型/seed的原始指标、光照图像数量、checkpoint与数据标识|
|raw/&lt;model&gt;/seed_&lt;seed&gt;/predictions.json|该模型/seed一次推理得到的COCO检测框|

缺少训练完成标记、best.pth或运行元数据时标记NOT RUN。缺少某seed或某光照组时，已有单seed值保留；相应指标未凑齐三个有效seed则不输出均值/标准差，不补0。合法的AP=0会正常参与统计，COCO未定义指标保持空值。

重跑相同命令会核对并复用已有记录；checkpoint、数据标注或显式光照映射发生变化时拒绝混用，使用新的`--ap-dir`开始另一批评估。失配/执行失败会记录原因，以非零状态退出，但先导出其余成功记录的部分CSV。失败或来源未核验通过的缓存不会进入CSV，离线汇总也遵循这一排除；单seed重试不会重新引入尚未修复的其他失败缓存。单模型/单seed补算后的自动汇总会保留同一目录中其他模型的已有原始结果，不会把共享总表缩成单模型表。

四光照列保持原论文顺序：

|CSV列名|对应光照|
|---|---|
|Natural AP (%)|自然光|
|Artificial AP (%)|人工光|
|Low-light AP (%)|微弱光|
|Sodium-light AP (%)|钠黄光|

`ap_protocol.json`的`illumination_columns`也保存这一中英文对应。run为COMPLETE时，Mixed AP和上述四项AP必须全部非空。

## 负宽高报错与部分失败后的恢复

2026-10-19用户提供的服务器日志显示10个run成功、5个run被旧脚本的预测框校验中断。旧版在任一失败后直接退出，导致成功记录也没有导出CSV。可先在仓库根目录运行`bash scripts/paper_v2/autodl/summarize_ap.sh`导出已有10组结果；只有B0三个seed齐全，其余模型的三seed均值暂时为空。

修复涉及`tools/paper_v2/ap_metrics.py`、`ap_workflow.py`、`ap_summary.py`。将这三个文件同步到AutoDL后，保留现有`raw/`记录，重新执行：

```bash
cd /root/autodl-tmp/DEIMv2-new
source scripts/paper_v2/autodl/env.sh
bash scripts/paper_v2/autodl/evaluate_and_summarize_ap.sh cuda:0 independent
```

来源一致的10个成功run会复用，5个失败run重新推理。此次日志未保存出错框的具体值，不能从通用报错判断其幅度；源码和合成复现已确认DEIM的有符号距离解码可产生有限负宽高，原生COCO评估原样接收这些预测。新脚本与原生保持一致，不裁剪、不交换端点、不取绝对值、不丢弃检测。非有限坐标、非法分数或未知ID仍拒绝。

新生成的`raw/<model>/seed_<seed>/metrics.json`增加`prediction_diagnostics`，记录预测总数、负宽/负高/负尺寸数量、零尺寸数量、负尺寸最大分数和最多5条实例，方便判断其实际影响。原有成功记录无需重新推理；其预测本就通过旧版非负宽高校验。

## 其他用法

```bash
# 后续C1—C4完成后：九组×三个seed（已完成且匹配的五组结果会复用）
bash scripts/paper_v2/autodl/evaluate_and_summarize_ap.sh cuda:0 all

# 指定保存训练结果的根目录、数据根目录和新的AP输出目录
bash scripts/paper_v2/autodl/evaluate_and_summarize_ap.sh cuda:0 independent \
  --runs-dir /root/autodl-tmp/DEIMv2-new/outputs/paper_v2 \
  --data-root /root/autodl-tmp/dataset/tomato_coco \
  --ap-dir outputs/paper_v2/ap/test_campaign_2

# 只补算G1的seed42；汇总均值仍要求0、42、3407全部有效
python tools/paper_v2/run.py evaluate-ap --model gr_vq --ap-seeds 42 --device cuda:0

# 仅重新汇总已有五组记录：不需要GPU，不读取图像或checkpoint
bash scripts/paper_v2/autodl/summarize_ap.sh

# 明确评估Validation，输出到ap/validation，始终与Test分开
bash scripts/paper_v2/autodl/evaluate_and_summarize_ap.sh cuda:0 independent --ap-split validation

# 将Test AP接入原论文效率主表，保留原有列顺序
bash scripts/paper_v2/autodl/summarize_efficiency.sh \
  --ap-csv outputs/paper_v2/ap/test/ap_by_seed.csv
```

九组离线汇总时，在`summarize_ap.sh`后追加`--ap-models baseline_native baseline_maskaware gr_vq li_copyblend dc_sta C1 C2 C3 C4`。指定新AP目录时，离线汇总和主表合并也要指向该目录。AP表中的百分数无需再乘100。

## 光照来源

只使用COCO image metadata中的`illumination`、`illumination_label`、`source_illumination_id`、`light_condition`，支持natural/artificial/low_light/sodium_light及自然光/人工光/微弱光/钠黄光、对应整数0/1/2/3。未知或互相冲突的标签会报错，不能从图像亮度或文件名猜测。

确有经过核实的映射时可增加`--illumination-map /真实路径/test_illumination_map.json`。文件内容为image_id字符串或完整file_name映射到光照标签，例如`{"1001":"natural","camera2/example.jpg":"钠黄光"}`。映射与已有有效元数据必须一致。

旧评估的`metrics.json`仅含Mixed指标，无法还原四光照AP；首次使用本脚本需执行上述推理。数据和权重只在AutoDL，本地验证只使用临时合成COCO数据；真实Test AP仍须服务器执行后生成。

## 已执行验证

2026-10-19全量回归427项通过（原有328项、新增99项），包含真实COCO合成评估、EMA推理、三个seed统计和效率主表导入；两个Bash脚本语法及离线shell→CLI冒烟通过。结果见`ap_regression_test_results.xml`及[08_训练前最终检查报告.md](08_训练前最终检查报告.md)。未读取真实Test，也未启动训练。

同日根据服务器失败反馈修复后，全量回归**435项全部通过**，耗时107.64秒，证据为`ap_failure_recovery_test_results.xml`。覆盖15组中5失败的部分CSV导出、失败缓存隔离、负宽高与原生COCO的12项指标一致性，以及原生后处理后的完整预测保存。修复后服务器补跑尚未执行，未在本地读取实际Test。
