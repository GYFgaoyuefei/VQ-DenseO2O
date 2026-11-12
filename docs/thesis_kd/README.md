# 番茄 DEIMv2-M/S 特征蒸馏

从独立开发目录执行下文命令。协议 `feature_kd_ms_v1` 用同一 VQ-DEIMv2-L 教师比较 M、S 学生在三个编码器尺度上的特征蒸馏。训练得到原生学生检测器，部署时直接加载选中的 EMA 参数。

| 任务 | 操作说明 |
|---|---|
| 核对工作树、保护范围和 SHA256 | [01 隔离开发](01_旧研究保护与隔离开发.md) |
| 核对实验、初始化与特征位置 | [02 配置](02_实验矩阵及配置分层.md) · [03 权重](03_教师学生与权重准备.md) · [04 公式](04_特征蒸馏公式与代码对应.md) |
| 设置环境、生成命令、运行 smoke | [05 Windows](05_Windows操作指南.md) · [06 AutoDL](06_AutoDL部署与训练.md) |
| 从中断恢复、导出、冻结 Test | [07 恢复和导出](07_恢复训练与学生导出.md) · [08 评价](08_验证测试与效率统计.md) |
| 复核实测、论文表及限制 | [09 验收记录](09_正确性和回归测试报告.md) · [10 论文表](10_论文结果表与已知限制.md) |

```sh
python -m thesis_kd.cli preflight --experiment m_base --static
python -m thesis_kd.cli resolve-config --experiment s_kd
python -m thesis_kd.cli export-commands --run-tag study01
```

本次交付状态以第 09 章为准。本地已确认可读的 Train/Val 路径见第 05 章。启动正式训练前，需要相应学生的两份初始化文件，并用目标硬件按正式 batch 做 smoke。
