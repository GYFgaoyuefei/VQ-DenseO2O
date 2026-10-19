# 新算法配置消融审计

状态：**PASS**。开发分支：paper_v3.0。

B1—C4组成完整2³矩阵；全部启用同一mask管线，VQ beta固定0.10。
以下比较实际解析后的配置，输出目录/实验名/seed单列为管理字段。没有读取数据或checkpoint。

## baseline_native → baseline_maskaware

唯一允许新增因素：`mask`。

|实际字段|原值|新值|分类|
|---|---|---|---|
|innovation.instance_mask_sync|false|true|允许差异|
|train_dataloader.dataset.split|"<absent>"|"train"|允许差异|
|train_dataloader.dataset.transforms.ops[4].type|"SanitizeBoundingBoxes"|"SanitizeVisibleBoundingBoxes"|允许差异|
|train_dataloader.dataset.transforms.ops[7].type|"SanitizeBoundingBoxes"|"SanitizeVisibleBoundingBoxes"|允许差异|
|train_dataloader.dataset.type|"CocoDetection"|"VisibleMaskCocoDetection"|允许差异|
|train_dataloader.dataset.visible_mask_required|"<absent>"|true|允许差异|
|train_dataloader.dataset.visible_mask_sidecar|"<absent>"|"${TOMATO_DATA_ROOT}/annotations/visible_masks_train.json"|允许差异|

## baseline_maskaware → vq_dense_o2o

唯一允许新增因素：`vq_dense_o2o`。

|实际字段|原值|新值|分类|
|---|---|---|---|
|VQDEIMCriterion.alpha|"<absent>"|0.75|允许差异|
|VQDEIMCriterion.gamma|"<absent>"|1.5|允许差异|
|VQDEIMCriterion.losses[0]|"<absent>"|"mal"|允许差异|
|VQDEIMCriterion.losses[1]|"<absent>"|"boxes"|允许差异|
|VQDEIMCriterion.losses[2]|"<absent>"|"local"|允许差异|
|VQDEIMCriterion.matcher.alpha|"<absent>"|0.25|允许差异|
|VQDEIMCriterion.matcher.change_matcher|"<absent>"|true|允许差异|
|VQDEIMCriterion.matcher.gamma|"<absent>"|2.0|允许差异|
|VQDEIMCriterion.matcher.iou_order_alpha|"<absent>"|4.0|允许差异|
|VQDEIMCriterion.matcher.matcher_change_epoch|"<absent>"|88|允许差异|
|VQDEIMCriterion.matcher.type|"<absent>"|"HungarianMatcher"|允许差异|
|VQDEIMCriterion.matcher.weight_dict.cost_bbox|"<absent>"|5|允许差异|
|VQDEIMCriterion.matcher.weight_dict.cost_class|"<absent>"|2|允许差异|
|VQDEIMCriterion.matcher.weight_dict.cost_giou|"<absent>"|2|允许差异|
|VQDEIMCriterion.reg_max|"<absent>"|32|允许差异|
|VQDEIMCriterion.vq_dense_o2o.apply_to.branches[0]|"<absent>"|"main"|允许差异|
|VQDEIMCriterion.vq_dense_o2o.apply_to.branches[1]|"<absent>"|"aux"|允许差异|
|VQDEIMCriterion.vq_dense_o2o.apply_to.branches[2]|"<absent>"|"pre"|允许差异|
|VQDEIMCriterion.vq_dense_o2o.apply_to.branches[3]|"<absent>"|"encoder"|允许差异|
|VQDEIMCriterion.vq_dense_o2o.apply_to.dense_o2o_positive_only|"<absent>"|true|允许差异|
|VQDEIMCriterion.vq_dense_o2o.beta|"<absent>"|0.1|允许差异|
|VQDEIMCriterion.vq_dense_o2o.copyblend.visibility_aware|"<absent>"|false|允许差异|
|VQDEIMCriterion.vq_dense_o2o.enabled|"<absent>"|true|允许差异|
|VQDEIMCriterion.vq_dense_o2o.invalid_mask_policy|"<absent>"|"iou_fallback"|允许差异|
|VQDEIMCriterion.vq_dense_o2o.modify_bbox_target|"<absent>"|false|允许差异|
|VQDEIMCriterion.vq_dense_o2o.modify_class_target|"<absent>"|false|允许差异|
|VQDEIMCriterion.vq_dense_o2o.modify_hungarian_matcher|"<absent>"|false|允许差异|
|VQDEIMCriterion.vq_dense_o2o.quality.combination|"<absent>"|"geometric_mean"|允许差异|
|VQDEIMCriterion.vq_dense_o2o.quality.use_coverage|"<absent>"|true|允许差异|
|VQDEIMCriterion.vq_dense_o2o.quality.use_purity|"<absent>"|true|允许差异|
|VQDEIMCriterion.weight_dict.loss_bbox|"<absent>"|5|允许差异|
|VQDEIMCriterion.weight_dict.loss_ddf|"<absent>"|1.5|允许差异|
|VQDEIMCriterion.weight_dict.loss_fgl|"<absent>"|0.15|允许差异|
|VQDEIMCriterion.weight_dict.loss_giou|"<absent>"|2|允许差异|
|VQDEIMCriterion.weight_dict.loss_mal|"<absent>"|1|允许差异|
|criterion|"DEIMCriterion"|"VQDEIMCriterion"|允许差异|
|innovation.vq_dense_o2o.enabled|false|true|允许差异|

## baseline_maskaware → li_copyblend

唯一允许新增因素：`li_copyblend`。

|实际字段|原值|新值|分类|
|---|---|---|---|
|innovation.li_copyblend.enabled|false|true|允许差异|
|train_dataloader.collate_fn.li_copyblend.enabled|false|true|允许差异|

## baseline_maskaware → dc_sta

唯一允许新增因素：`dc_sta`。

|实际字段|原值|新值|分类|
|---|---|---|---|
|DINOv3STAs.dc_sta.enabled|false|true|允许差异|
|innovation.dc_sta.enabled|false|true|允许差异|

## vq_dense_o2o → vq_dense_o2o_li_copyblend

唯一允许新增因素：`li_copyblend`。

|实际字段|原值|新值|分类|
|---|---|---|---|
|innovation.li_copyblend.enabled|false|true|允许差异|
|train_dataloader.collate_fn.li_copyblend.enabled|false|true|允许差异|

## vq_dense_o2o → vq_dense_o2o_dc_sta

唯一允许新增因素：`dc_sta`。

|实际字段|原值|新值|分类|
|---|---|---|---|
|DINOv3STAs.dc_sta.enabled|false|true|允许差异|
|innovation.dc_sta.enabled|false|true|允许差异|

## li_copyblend → li_copyblend_dc_sta

唯一允许新增因素：`dc_sta`。

|实际字段|原值|新值|分类|
|---|---|---|---|
|DINOv3STAs.dc_sta.enabled|false|true|允许差异|
|innovation.dc_sta.enabled|false|true|允许差异|

## vq_dense_o2o_li_copyblend → vq_dense_o2o_li_copyblend_dc_sta

唯一允许新增因素：`dc_sta`。

|实际字段|原值|新值|分类|
|---|---|---|---|
|DINOv3STAs.dc_sta.enabled|false|true|允许差异|
|innovation.dc_sta.enabled|false|true|允许差异|

意外差异：0项。

共同协议：120e；640²；3类；train batch32/val64；AdamW lr5e-4、DINO lr1.25e-5；
FlatCosine warmup2000/flat60/no_aug14；EMA FP32 Mixed Validation AP选模，精确并列保留较早epoch。
beta_select Baseline与当前共同配方数值一致；保留paper管理/同步修复，详见03_beta_select迁移说明.md。
