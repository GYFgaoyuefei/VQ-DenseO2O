# Windows 本地操作

在开发工作树打开 PowerShell。建立独立环境时复制已验证复现环境的版本；本项目不会调用 pip、conda install 或联网拉取权重。本次检查使用已有 `pytorch310` 解释器，未改依赖。该机实际为 PyTorch 2.7.1+cu126 / torchvision 0.22.1，仓库 requirements 指定 2.5.1 / 0.20.1。移到正式环境应重新跑验收。

```powershell
Set-Location 'D:/workspace/python_workspace/study/VQ-DenseO2O-thesis-kd'
. ./scripts/thesis_kd/windows/env.example.ps1
$env:PYTHON = 'D:/ProgramData/anaconda3/envs/pytorch310/python.exe'
& ./scripts/thesis_kd/windows/preflight.ps1 -Experiment m_base
& ./scripts/thesis_kd/windows/run_tests.ps1
& ./scripts/thesis_kd/windows/generate_commands.ps1 -RunTag study01
```

本机已核对 `D:/workspace/python_workspace/study/tomato_data/data_new/tomato_coco` 下的 Train/Val。替换下面两条权重示例路径后，再执行资源核对和短 smoke。成功完成后，最后一条会启动正式 120 轮训练，应由研究者按计划单独运行。

```powershell
$env:THESIS_KD_M_BACKBONE_WEIGHTS = 'D:/weights/m/vittplus_distill.pt'
$env:THESIS_KD_M_DETECTOR_CHECKPOINT = 'D:/weights/m/deimv2_dinov3_m_coco.pth'
& ./scripts/thesis_kd/windows/preflight.ps1 -Experiment m_base -WithResources
& $env:PYTHON -m thesis_kd.cli smoke --experiment m_base --seed 42 --steps 2 --run-tag memcheck --device cuda
```

```powershell
& $env:PYTHON -m thesis_kd.cli train --experiment m_base --seed 42 --run-tag study01 --device cuda
```

`-WithResources` 检查当前实验的图像路径及标注文件，并试装初始化权重。Base 不需要 T0。静态 preflight 和 resolve-config 只需 Python、PyYAML 及当前源码。已有 run 目录会令新训练报错；resume 使用原 tag，新实验使用新 tag。每个节点在本地路径安全检查通过后才会创建结果目录。
