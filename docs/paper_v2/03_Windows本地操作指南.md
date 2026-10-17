# Windows 本地操作指南

所有命令在PowerShell执行。本轮使用已有环境 D:\ProgramData\anaconda3\envs\pytorch310\python.exe；默认base Python的torch DLL初始化失败，因此没有使用它，也没有安装依赖。

```powershell
Set-Location D:\workspace\python_workspace\study\DEIMv2-new
$py = "D:\ProgramData\anaconda3\envs\pytorch310\python.exe"
git status --short
git branch --show-current
& $py -c "import torch,torchvision; print(torch.__version__,torchvision.__version__)"
& .\scripts\paper_v2\windows\preflight.ps1 -Python $py -ConfigOnly
& .\scripts\paper_v2\windows\run_tests.ps1 -Python $py
& .\scripts\paper_v2\windows\generate_commands.ps1 -Python $py
& $py tools/paper_v2/run.py generate-combinations
```

合成完整模型单步前向、反向、optimizer step及推理，不读取真实数据/权重：
```powershell
& $py tools/paper_v2/run.py smoke --model gr_vq --seed 42 --device cpu --synthetic --report outputs/paper_v2_checks/gr_vq_smoke.json
```
synthetic 使用batch2、128×128随机输入及随机初始化真实模型，保留真实网络层数与损失。它不能替代640真实数据和预训练权重的烟雾测试。各模块数学/增强测试由tests/paper_v2独立覆盖。

需要真实本地preflight时，填写真实路径后运行：
```powershell
$env:TOMATO_DATA_ROOT = "D:\你的真实数据目录\tomato_coco"
$env:DINOV3_WEIGHTS = "D:\你的真实权重目录\实际DINO文件.pth"
$env:TOMATO_DETECTOR_CHECKPOINT = "D:\你的真实权重目录\实际Detector文件.pth"
& .\scripts\paper_v2\windows\preflight.ps1 -Python $py
```
这些示例路径不是已经核实的资源。无真实资源时config-only可以运行，完整preflight应报缺失，而不能标PASS。

常见问题：
- WinError1114/c10.dll：选择已有可用PyTorch环境，不要在base环境盲目升级。
- ModuleNotFoundError：先检查选中的环境；calflops缺失只跳过FLOPs，核心依赖不可用需在服务器按requirements单独配置。
- sidecar SHA mismatch：使用对应Train标注生成/提供正确sidecar，不编辑校验字段来绕过。
- LI accepted_pastes长期为0：检查真实光照metadata、有效mask比率、背景兼容阈值及碰撞统计，不凭Test指标放宽。
- 输出目录已存在：使用相同run的last.pth恢复；不要删除原结果。
