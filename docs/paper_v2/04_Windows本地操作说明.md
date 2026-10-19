# Paper v3.0 Windows 本地操作说明

日期：2026-10-19。工程路径为 `D:\workspace\python_workspace\study\DEIMv2-new`，开发分支为 `paper_v3.0`。脚本继续放在 `tools/paper_v2/` 和 `scripts/paper_v2/`，但正式实验使用恢复后的 beta_select 算法及全新的输出根目录。

本地没有 AutoDL 的真实数据与预训练资源。下面的默认本地检查只解析配置、运行合成测试及 CPU 冒烟，不读取 Test，不启动完整训练。

## 1. 进入工程并检查工作区

在 PowerShell 中执行：

```powershell
Set-Location 'D:\workspace\python_workspace\study\DEIMv2-new'
$Py = 'D:\ProgramData\anaconda3\envs\pytorch310\python.exe'
$env:PYTHONUTF8 = '1'
git status --short
git branch --show-current
git branch --list paper_v3.0 paper_v2.0 beta_select
& $Py -X utf8 tools/paper_v2/run.py --help
```

本轮代码已在 `paper_v3.0` 上修改。若显示其他分支，先核对本地修改范围再切换；不要使用 `reset --hard`、`clean` 或整工程覆盖来处理未提交修改。`beta_select` 只作读取和数值对照，不能执行 `git merge beta_select`。

## 2. 九组配置解析及消融差异审计

现有 PowerShell 预检脚本的真实参数为 `-Python`、`-Model` 和 `-ConfigOnly`：

```powershell
.\scripts\paper_v2\windows\preflight.ps1 -Python $Py -Model all -ConfigOnly
& $Py -X utf8 tools/paper_v2/run.py audit-configs
```

`-ConfigOnly` 不展开资源环境变量，也不构建真实 Dataset 或读取权重。`audit-configs` 检查九组有效配置、完整 `2^3` 因素矩阵、B1-C4 的 mask 同步及允许差异，生成：

- `outputs/paper_v2/config_ablation_matrix.csv`
- `docs/paper_v2/ablation_config_audit.md`

这里的 `outputs/paper_v2/` 仅保留用户指定的配置审计文件位置；新训练、AP 与效率结果不会写入旧实验目录。

## 3. 合成测试与有限 CPU 冒烟

```powershell
& $Py -X utf8 -m pytest tests/paper_v2 -q

& $Py -X utf8 tools/paper_v2/run.py smoke --model baseline_native --seed 42 --device cpu --synthetic --threads 2 --report outputs/paper_v2_beta_select_algorithms/checks/synthetic_cpu/baseline_native_seed42.json

& $Py -X utf8 tools/paper_v2/run.py smoke --model vq_dense_o2o_li_copyblend_dc_sta --seed 42 --device cpu --synthetic --threads 2 --report outputs/paper_v2_beta_select_algorithms/checks/synthetic_cpu/C4_seed42.json
```

也可使用现有测试包装脚本：

```powershell
.\scripts\paper_v2\windows\run_tests.ps1 -Python $Py
```

`smoke --synthetic` 使用两张 128×128 合成图，检查真实模型/criterion 前向、loss、backward、optimizer step 和推理 shape，不下载预训练权重。它不经过真实增强供体选择，也不等价于正式 batch 32 的显存验收；LI 的具体像素融合、光照条件、碰撞和字段同步由单元测试覆盖，真实数据增强由 AutoDL 有限 Train 冒烟补充。

实际 PASS/FAIL 和尚未执行项以 `06_算法正确性测试报告.md` 为准，不能用合成 PASS 代替真实 AP 或收敛结论。

Baseline/VQ 参考对照固定读取本次核实的 beta_select commit `f92edb021fd468e354f5767bb81ac745e0347cfe`，避免参考分支以后移动导致对照漂移。若手动上传或浅克隆缺少该 Git 对象，对照项会明确 SKIP，必须视为该环境 NOT RUN；其他合成测试继续运行。本机已存在该对象，实际对照已执行。需要复现时从可信的同一远端获取 beta_select 历史（`git fetch origin beta_select`），不合并或切换参考算法分支。

## 4. 生成全部正式训练命令

```powershell
.\scripts\paper_v2\windows\generate_commands.ps1 -Python $Py
```

该命令只打印九模型 × 三 seeds 的 27 条 AutoDL 启动命令，不执行训练。正式模型标识为：

```text
baseline_native
baseline_maskaware
vq_dense_o2o
li_copyblend
dc_sta
vq_dense_o2o_li_copyblend
vq_dense_o2o_dc_sta
li_copyblend_dc_sta
vq_dense_o2o_li_copyblend_dc_sta
```

不要继续使用 `gr_vq`、`dc_sta_residual` 或旧 `C1` 简写作为 CLI 模型值；组合使用上述完整名称。

## 5. 将本轮源码同步至 AutoDL

推荐使用完整 Git 分支同步，避免只传部分文件造成新配置配上旧算法。先由用户检查变更、选择应提交的文件并在本地完成 commit，然后执行：

```powershell
git push -u origin paper_v3.0
```

本次工程修改不会替用户自动 commit/push。提交时应包含本轮新增及修改源码、配置、脚本、测试、文档，并保留明确的文件删除记录；不要提交真实数据、预训练权重或旧实验结果。

服务器端在确认工作区改动已妥善保留后，从同一远端 fetch 并切换 `paper_v3.0`，详见 `05_AutoDL训练命令.md`。不要把旧训练的 `best.pth`、`last.pth`、`selection.json` 拷入新实验目录。

若只能手动上传，以最终交付文件列表为准，上传实际变更涉及的 `engine/`、`configs/`、`tools/paper_v2/`、`scripts/paper_v2/`、`tests/paper_v2/`、`docs/paper_v2/` 及根目录入口文件（例如有变更的 `train.py`）。核对服务器本地修改后再覆盖对应源码，不要只上传新 YAML；同时处理列表中已删除的正式旧配置。保留服务器原 `outputs/`、数据集和权重，禁止以删除输出目录的方式“重新部署”。

手动上传后服务器仍须处于名为 `paper_v3.0` 的 Git 分支，入口会检查分支名；若服务器没有该分支，在确认当前工作区范围后由用户从现有分支创建，再应用本轮源码。仅创建同名分支并不代表代码已同步，必须重新执行配置审计和测试。

交付同时提供本轮完整源码快照包 `outputs/paper_v2_beta_select_algorithms/delivery/paper_v3_source.zip`（仓库跟踪源码加本轮新增文件，不包含`.git`、数据、权重或`outputs/`）。可先上传到服务器临时目录并解压检查，再按仓库相对路径同步；不要用“删除整个工程再解压”的方式处理。包内[PAPER_V3_FILES_CHANGED.txt](PAPER_V3_FILES_CHANGED.txt)列出本轮新增、修改与删除项。若采用文件覆盖方式，核对后单独移除其中7个退役配置/测试路径，保留所有旧训练结果；Git分支同步会自然应用这些源码删除。`.gitattributes`保证AutoDL包装脚本使用LF换行，也应一并同步。包不创建Git分支，手动同步仍需确认服务器分支和参考提交可用性。

## 6. 真实数据与运行边界

真实 Train batch、两份统一预训练权重及正式 GPU 均在 AutoDL；本地对应检查标为 `NOT RUN`。服务器先执行预检与 `smoke_train_one.sh` 的有限步验收，通过后再由用户启动正式训练。有限 Train 冒烟不创建 Validation/Test dataloader，不保存正式模型 checkpoint。

正式新结果写入：

```text
outputs/paper_v2_beta_select_algorithms/<model>/seed_<seed>/
```

之前 `outputs/paper_v2/` 的 Baseline、GR-VQ、Residual DC-STA 等结果保留原样。即使模型名相同，旧结果也不能当作本轮结果复用。断点恢复只允许本轮新算法、同模型、同 seed、同一运行目录的 checkpoint；不能从旧 GR-VQ/Residual DC-STA 接着训练。
