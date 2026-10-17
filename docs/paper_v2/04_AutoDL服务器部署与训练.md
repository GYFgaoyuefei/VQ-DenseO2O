# AutoDL 服务器部署与训练

本轮没有连接AutoDL、没有上传文件、没有正式训练。下列命令对应本轮真实脚本。

用户已确认正式数据、mask sidecar及预训练权重仅在AutoDL。本地候选数据检查仅作为兼容性证据；真实batch32验收以服务器有限步结果为准，当前为NOT RUN。

用户已提供以下服务器资源路径，已写入`env.sh`和`env.example.sh`。这些是用户确认的路径，尚未在服务器检查文件存在性或加载权重：

|变量|服务器路径|
|---|---|
|TOMATO_DATA_ROOT|/root/autodl-tmp/dataset/tomato_coco|
|DINOV3_WEIGHTS|/root/autodl-tmp/pretrained/deimv2_l/dinov3_vits16_pretrain_lvd1689m-08c60483.pth|
|TOMATO_DETECTOR_CHECKPOINT|/root/autodl-tmp/pretrained/deimv2_l/deimv2_dinov3_l_coco.pth|

## 1. 上传与环境

把paper_v2.0工作区的源码、configs、tools/paper_v2、scripts/paper_v2、docs/paper_v2上传到 /root/autodl-tmp/DEIMv2-new；推荐在本地检查diff并自行提交后用Git同步该分支，避免漏传未跟踪新文件。服务器已有改动时不要reset。数据独立位于 /root/autodl-tmp/dataset/tomato_coco。

数据布局：
```text
tomato_coco/
  images/train/
  images/val/
  images/test/                    # 只由独立Test命令读取
  annotations/instances_train.json
  annotations/visible_masks_train.json
  annotations/instances_val.json
  annotations/instances_test.json # 只由独立Test命令读取
```

激活服务器已有合适环境，核对torch/torchvision/faster-coco-eval等。参考仓库requirements.txt，不要在未知环境直接大量升级。
```bash
cd /root/autodl-tmp/DEIMv2-new
git branch --show-current
python -c 'import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())'
# 已提供配置好的env.sh；缺少时才从示例创建，保留服务器已有自定义配置
test -f scripts/paper_v2/autodl/env.sh || cp scripts/paper_v2/autodl/env.example.sh scripts/paper_v2/autodl/env.sh
source scripts/paper_v2/autodl/env.sh
bash scripts/paper_v2/autodl/preflight.sh
python -m pytest tests/paper_v2 -q
```

五组共用上述DINO初始化和detector --tuning。mask sidecar文件名尚未确认，当前默认`annotations/visible_masks_train.json`；若服务器采用其他名称，在`env.sh`设置`TOMATO_VISIBLE_MASK_TRAIN`为实际文件。不要仅因本地存在vqcr文件就假定服务器也使用该文件。

## 2. 真实数据烟雾测试与正式训练

```bash
for model in baseline_native baseline_maskaware gr_vq li_copyblend dc_sta; do
  bash scripts/paper_v2/autodl/smoke_one.sh "$model" 42 cuda:0
done
# 首先单独启动一组
bash scripts/paper_v2/autodl/train_one.sh gr_vq 42 cuda:0
# 或顺序执行五组seed42（不要重复执行已经创建结果的组）
bash scripts/paper_v2/autodl/train_seed42.sh cuda:0
# 首批检查结束后，执行其余两个正式种子
bash scripts/paper_v2/autodl/train_remaining_seeds.sh cuda:0
```

真实smoke使用640输入、batch2和实际预训练资源；它不能保证全局batch32能装入一张卡。正式batch32保持不变。需要4卡时：
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4
bash scripts/paper_v2/autodl/train_one.sh dc_sta 42 cuda:0
```
wrapper实际生成 python -m torch.distributed.run --standalone --nproc_per_node 4 train.py ...，每进程train batch8。单卡默认NPROC_PER_NODE=1，每卡batch32；应事先确认显存，不能将batch2 smoke当作batch32容量证明。

本轮服务器验收只需执行以下有限步循环；上面的正式训练命令留待验收完成后的实验阶段。有限步入口检查实际训练batch，保留全部训练配方：
```bash
for model in baseline_native baseline_maskaware gr_vq li_copyblend dc_sta; do
  bash scripts/paper_v2/autodl/smoke_train_one.sh "$model" 42 cuda:0 \
    --smoke-steps 3 --smoke-epoch 10 || break
done
```
该循环使用实际Train、统一预训练、AMP、EMA、原生optimizer/调度；总batch固定32，每组seed42、3步，任一组失败即停止后续组。入口最多允许10步，不进入120-epoch fit、不执行Validation/Test、不保存checkpoint。每rank输出loss、梯度、显存和迭代时间。默认结果位于outputs/paper_v2_checks/real_train/，不会覆盖正式run。epoch10只指定增强/调度阶段，并非先运行10轮训练。sidecar文件名不同可在env.sh中设置TOMATO_VISIBLE_MASK_TRAIN。当前本地验收范围与NOT RUN项见08_训练前最终检查报告.md。

恢复同一run：
```bash
export RESUME=/root/autodl-tmp/DEIMv2-new/outputs/paper_v2/gr_vq/seed_42/last.pth
bash scripts/paper_v2/autodl/train_one.sh gr_vq 42 cuda:0
unset RESUME
```
非空目录不允许新训练覆盖，完成标志存在时拒绝恢复。不要用一个run的checkpoint冒充另一模型/seed恢复。

## 3. Validation、独立Test、效率和汇总

```bash
bash scripts/paper_v2/autodl/validate_one.sh gr_vq 42 cuda:0
# 固定方法/超参数和Validation选模后，才执行下面的独立Test命令
bash scripts/paper_v2/autodl/evaluate_test_one.sh gr_vq 42 cuda:0
bash scripts/paper_v2/autodl/benchmark_one.sh gr_vq 42 cuda:0
bash scripts/paper_v2/autodl/summarize_results.sh
```

Validation/Test默认使用run内best.pth的EMA权重，可在评估脚本末尾加 --checkpoint /真实/路径.pth。评估写入新的带时间戳目录，不覆盖训练日志；Test命令不由训练脚本调用。

正式benchmark只接受完成120 epoch训练的同一run内best.pth，并核对EMA Mixed Validation选模记录。所有B0–D1及C1–C4共用同一程序：指定RTX PRO 6000 Blackwell Server Edition、PyTorch eager eval/deploy、FP32、禁用autocast/TF32、batch1、GPU驻留全零[1,3,640,640]输入。每seed独立5轮，每轮100次预热、500次正式计时，每次调用前后CUDA同步并用Python wall-clock计时。主延迟独立测量detector + tensor postprocessor；另独立测量forward-only及postprocessor-only。不接受自定义计时次数，不用两段相加代替主延迟。完整协议、FLOPs覆盖说明及论文表格见08_统一效率统计协议.md。

```bash
# 全部五组×三个种子；先完成对应训练，逐组顺序执行
bash scripts/paper_v2/autodl/benchmark_all.sh cuda:0 independent
# 后续四组组合也完成后，可在全新结果目录执行九组
bash scripts/paper_v2/autodl/benchmark_all.sh cuda:0 all --efficiency-dir outputs/paper_v2/efficiency_all
# 汇总已测记录，不加载GPU或checkpoint，不触发Test
bash scripts/paper_v2/autodl/summarize_efficiency.sh
```

## 4. 输出与选模

每组路径 outputs/paper_v2/<model>/seed_<seed>/，保存train.log、config_resolved.yml、run_info.json、epoch_validation.csv、best.pth、last.pth、selection.json、原生log.txt及stage检查点。成功跑完写completed.json。run_info记录源码commit/dirty状态、命令、资源路径、seed及最佳Validation epoch。

统一最终best.pth按每个epoch评估时的EMA Mixed Validation AP严格提高更新；相同AP保留较早epoch。best_stg1/best_stg2是原生stage控制产物，不作为跨组最终选择依据。epoch编号从0开始。恢复时继续同一run，不应根据Test改变训练设置。

epoch106进入最终阶段时按原机制重载best_stg1模型/优化器/EMA，并设置有效decay=0.9999；后续原stage判据触发回退时保留阶段比较状态、应用decay递减。best.pth捕获评估后、回退前的选中EMA；last.pth保存回退后的续训状态与实际循环epoch。中断恢复持久化阶段决策状态，避免恢复时重新以当前AP初始化阈值。该行为已做CPU状态机对照，真实DDP恢复仍需服务器验证。
