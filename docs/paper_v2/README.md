# paper_v3.0 当前文档入口

本轮恢复 beta_select Baseline、VQ-DenseO2O beta=0.10 和原 Sigmoid DC-STA，保留 LI-CopyBlend 及统一实验管理。配置/脚本目录沿用 `paper_v2`；分支为 `paper_v3.0`，所有新结果写入 `outputs/paper_v2_beta_select_algorithms/`。

1. [算法实现说明](01_算法实现说明.md)
2. [实验配置矩阵](02_实验配置矩阵.md)
3. [beta_select迁移说明](03_beta_select迁移说明.md)
4. [Windows本地操作说明](04_Windows本地操作说明.md)
5. [AutoDL训练命令](05_AutoDL训练命令.md)
6. [算法正确性测试报告](06_算法正确性测试报告.md)
7. [后续组合消融与统计说明](07_后续组合消融与统计说明.md)

实际配置差异见 [ablation_config_audit.md](ablation_config_audit.md)。其他旧编号文档、旧测试 XML/JSON 保留作历史，不能将其中 GR-VQ、残差门控或旧输出目录命令用于本轮实验。
