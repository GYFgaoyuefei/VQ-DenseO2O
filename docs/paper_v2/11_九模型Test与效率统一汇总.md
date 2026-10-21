# 九个模型 × 三个 seed：最终 Test 与效率汇总

日期：2026-10-21。代码位于 `paper_v3.0`，默认评估 `outputs/paper_v2_beta_select_algorithms` 中已有的 checkpoint，不按训练分支、suite 标签或 DC 版本号限制加载。脚本只在显式运行时读取 Test；不会训练，也不会根据 Test 重新选 checkpoint。

## 一条命令

先将本次更新的以下文件一起同步到服务器同一仓库（不能只更新 Bash 入口）：

- `scripts/paper_v2/autodl/evaluate_final_test_all.sh`
- `tools/paper_v2/final_test.py`
- `tools/paper_v2/final_test_report.py`
- `tools/paper_v2/checkpoint_binding.py`
- `tools/paper_v2/evaluation_state.py`
- `tools/paper_v2/run.py`
- `tools/paper_v2/ap_evaluate.py`
- `tools/paper_v2/ap_workflow.py`
- `tools/paper_v2/ap_summary.py`
- `tools/paper_v2/efficiency_summary.py`

使用已有训练 Python 环境执行：

```bash
cd /root/autodl-tmp/DEIMv2-new
bash scripts/paper_v2/autodl/evaluate_final_test_all.sh cuda:0 \
  --runs-dir /root/autodl-tmp/DEIMv2-new/outputs/paper_v2_beta_select_algorithms \
  --data-root /root/autodl-tmp/dataset/tomato_coco
```

默认即为上述目录，也可简写为：

```bash
bash scripts/paper_v2/autodl/evaluate_final_test_all.sh cuda:0
```

若 Python 不在当前 PATH，可设置 `export PYTHON=/你的环境/bin/python`。无需 Train mask sidecar 或最初的 DINO/detector 预训练权重；评估加载各 run 的 `best.pth` 中的 EMA 参数。

模型覆盖：`baseline_native`、`baseline_maskaware`、`vq_dense_o2o`、`li_copyblend`、`dc_sta`、`vq_dense_o2o_li_copyblend`、`vq_dense_o2o_dc_sta`、`li_copyblend_dc_sta`、`vq_dense_o2o_li_copyblend_dc_sta`，每个模型均执行 `seed_0`、`seed_42`、`seed_3407`，共 27 组。

## 数据与 checkpoint

Test 使用：

```text
/root/autodl-tmp/dataset/tomato_coco/images/test/
/root/autodl-tmp/dataset/tomato_coco/annotations/instances_test.json
```

默认 `--run-policy existing-checkpoint`，每个 seed 目录只要求：

```text
best.pth
```

优先使用本次训练保存的 `config_resolved.yml`（也支持 `config.yml`、`config.yaml`），保留其推理结构和参数。没有配置时，使用模型目录名对应的仓库配置，明确打印 `CONFIG FALLBACK` 并记录配置来源；所有实际参数均须严格加载成功。DC 的门控编号只属历史记录，不阻止评估；有原始配置时按其中定义的 sigmoid 或残差 tanh 公式计算，兼容旧 `DCDINOv3STAs` 名称。若 DC 也缺少原配置，使用目录对应的默认 sigmoid 公式并显式记录这一假设；同形权重本身不能证明门控公式。

`run_info.json`、`selection.json`、`completed.json` 都是可选来源信息。不会补造这些文件，不会因旧 `algorithm_suite` 标签失败。没有充分记录时，选模历史显示 `UNVERIFIED`；真实 epoch 读取 `best.pth:last_epoch`。仍要求 checkpoint 中存在完整 FP32 `ema.module`，不会擅自改用 last.pth、非 EMA 参数或丢弃不匹配权重。`Training DINOv3 from scratch...` 是模型构造时不读取预训练源的旧日志，随后会明确打印已加载的 EMA checkpoint；此脚本不执行训练。

旧的严格训练记录审计仅作为显式选项 `--run-policy strict` 保留，默认命令无需此选项。

仅检查 27 个目录的必需文件并打印命令（不读取 Test、加载权重或测量）：

```bash
bash scripts/paper_v2/autodl/evaluate_final_test_all.sh cuda:0 --dry-run
```

## 四光照与 Mixed

每个 checkpoint 只进行一次 Test 推理，同一批检测结果分别用于完整 Mixed Test 和四个光照子集的 COCO bbox 评估。Mixed AP 直接在全体 Test 上计算，不能把四个 AP 平均当 Mixed AP。

| 输出列 | 光照 | 标准字符串 / 整数标签 |
| --- | --- | --- |
| Natural AP (%) | 自然光 | `natural` / `0` |
| Artificial AP (%) | 人工光 | `artificial` / `1` |
| Low-light AP (%) | 微弱光 | `low_light` / `2` |
| Sodium-light AP (%) | 钠黄光 | `sodium_light` / `3` |

读取 COCO `images` 元数据中的 `illumination`、`illumination_label`、`source_illumination_id`、`light_condition`。也支持对应中文标签。脚本在推理前检查全部图像标签一致且四组均有图像，打印每组数量。未知或冲突标签报错，不从亮度或文件名猜测。

如果标注 JSON 未包含光照字段，使用已经核实的映射文件，例如：

```json
{
  "1001": "自然光",
  "1002": "人工光",
  "camera1/frame003.jpg": "微弱光",
  "camera2/frame004.jpg": "钠黄光"
}
```

key 是图像 ID 的字符串或 COCO 中完整 `file_name`，需覆盖所有缺少光照标签的 Test 图片。追加：

```bash
bash scripts/paper_v2/autodl/evaluate_final_test_all.sh cuda:0 \
  --illumination-map /root/autodl-tmp/dataset/tomato_coco/annotations/test_illumination_map.json
```

四组 image count、原始预测、各项 AP 来源保存在协议与每个 run 的原始 JSON 中。原生有限负宽高预测按原 COCO 路径处理，不为消除报错而裁剪、交换端点或删除。

## 输出文件与表格顺序

```text
outputs/paper_v2_beta_select_algorithms/
  final_test/
    test_results_by_seed.csv       # 27行，每个模型/seed的实测值及状态
    test_results_summary.csv       # 9行，数值均值 + 独立样本标准差列
    paper_test_table.csv           # 9行，严格23列，适合整理进论文
    final_test_protocol.json       # 协议、数据来源、光照数量、覆盖状态与错误
    final_test_attempts.json       # 本次执行/复用/失败记录
    logs/
  ap/test/
    ap_by_seed.csv
    ap_summary.csv
    ap_protocol.json
    evaluation_attempts.json
    raw/<model>/seed_<seed>/metrics.json
    raw/<model>/seed_<seed>/predictions.json
  efficiency/
    efficiency_by_seed.csv
    efficiency_summary.csv
    efficiency_protocol.json
    paper_efficiency_table.csv
    raw/<model>/seed_<seed>/efficiency.json
```

最终使用 **`final_test/paper_test_table.csv`**，列严格如下：

| Model | Epoch | Params(M) | GFLOPs | Latency(ms) | FPS | Hardware | Mixed AP (%) | Mixed AP50 (%) | Mixed AP75 (%) | Mixed AP-S (%) | Mixed AP-M (%) | Mixed AP-L (%) | Mixed AR1 (%) | Mixed AR10 (%) | Mixed AR100 (%) | Mixed AR-S (%) | Mixed AR-M (%) | Mixed AR-L (%) | Natural AP (%) | Artificial AP (%) | Low-light AP (%) | Sodium-light AP (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

`Model` 使用原文件夹名，便于对应实验。`Epoch=120` 表示原论文中的预期训练预算，不表示已核实训练完成；各 seed 选中的实际轮次另存为 `Selected epoch`（从 1 开始）和 `selected_epoch_zero_based`。逐 seed 表还记录 `Configuration source` 与 `Selection verification`，完整来源保存在协议 JSON。主表 `Params(M)` 为 **Deploy Params(M)**，不是 Full Params。

`paper_test_table.csv` 的 AP/AR 与 Latency 使用 `均值 ± 样本标准差`，保留三位小数；`test_results_summary.csv` 的均值和标准差分列存储，便于后续画图。每项指标必须有三个有效 seed 才汇总；缺失、失败或 COCO 未定义指标留空，合法的 0 正常参与统计。

## 效率口径

直接复用已有统一效率程序，所有 27 组都使用相同协议：

- NVIDIA RTX PRO 6000 Blackwell Server Edition；PyTorch eager eval/deploy；FP32，autocast=false、TF32=false。
- batch=1，GPU 常驻全零 `[1,3,640,640]` FP32 输入。
- 每 seed 5 轮；每轮每个计时范围 100 次预热、500 次正式计时，逐次 CUDA 同步的 Python wall-clock。
- 主 Latency 为 detector forward + tensor postprocessor 实际合并测量；forward-only、postprocessor-only 独立测量，额外记录 P50/P95 和峰值显存。
- 每 seed 先取 5 轮均值，再对三个 seed 求均值及样本标准差。FPS 始终为 `1000 / 对应平均Latency(ms)`，不平均各 seed 的 FPS。
- GFLOPs 仅统计 deploy detector forward。记录 calflops 版本与覆盖说明，主表明确显示 `PARTIAL`；不能宣称完整精确 FLOPs。

IO、预处理、dataloader、传输、criterion、VQ/LI 训练计算和 COCO evaluator 不进入效率计时。若当前 GPU 不符合既定硬件，效率阶段明确失败，不把其他 GPU 的速度冒充标准结果；AP 已成功的结果仍导出。

## 分开执行与中断恢复

先只算 Test AP（不要求协议指定 GPU，也不做效率测试）：

```bash
bash scripts/paper_v2/autodl/evaluate_final_test_all.sh cuda:0 --mode ap-only
```

之后在符合协议的 GPU 上补效率：

```bash
bash scripts/paper_v2/autodl/evaluate_final_test_all.sh cuda:0 --mode efficiency-only
```

只从已有原始 JSON 重建汇总，不读取图像、标注或 checkpoint：

```bash
bash scripts/paper_v2/autodl/evaluate_final_test_all.sh cuda:0 --mode summarize-only
```

此前出现 `Missing completed run files` 或 `old runs cannot be reused` 时，同步本次代码后直接重跑原命令，无需删除任何结果或补写训练记录。重新运行默认命令会校验并复用已有成功 AP/效率记录，补算失败项。suite 标签或补充训练说明不会让缓存失效；实际推理配置、checkpoint、数据或协议发生变化时才拒绝复用。未完成或尚未核验的缓存不会混入最终表；个别 run 失败会继续处理其它 run，并导出已有成功结果。完整阶段失败或结果不齐时返回非零退出码，具体原因见日志与 JSON，不能将部分表误认为完整论文结果。

AP 和效率还会对照相同 model/seed 的 checkpoint 身份和 selected epoch，拒绝混合不同重训的权重。若 checkpoint、Test 标注/光照映射或效率运行环境改变，应使用新的 `--ap-dir` / `--efficiency-dir` / `--report-dir`，保留历史结果，不删除旧记录强行复用。

本次交付的 Python AST/内存编译检查、Bash `-n`、CLI `--help` 与 `--dry-run` 均通过；dry-run 确认 9 模型、3 seeds、27 条效率命令。新增汇总和续跑回归用例源码，但未执行测试。未连接 AutoDL、未读取真实 Test、未生成任何虚构指标。运行结果须以服务器实际执行输出为准。
