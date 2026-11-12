# AutoDL 部署与训练

在服务器准备同一基线的独立仓库副本，将交付包解压到该副本根目录。建议目录为 `/root/autodl-tmp/VQ-DenseO2O-thesis-kd`。普通 git worktree 的 .git 文件依赖本机路径，跨机器请用实际 checkout 加 KD 源文件，或用含 KD 文件的版本。保持 Bash 脚本 LF 换行，可以直接运行 `bash 脚本`。

```bash
cd /root/autodl-tmp/VQ-DenseO2O-thesis-kd
source scripts/thesis_kd/autodl/env.example.sh
export RUN_TAG=study01
# 在此将环境变量中的示例替换为实际资源，或给单个 CLI 传 --paths。
python -m thesis_kd.cli resolve-config --experiment m_kd
bash scripts/thesis_kd/autodl/preflight.sh m_base 42
bash scripts/thesis_kd/autodl/smoke_one.sh m_base 42 --steps 2
bash scripts/thesis_kd/autodl/smoke_one.sh m_kd 42 --steps 2
```

训练按实验串行运行。下面的单组命令与核心矩阵脚本是两种启动方式；同一 tag 下的同一实验/种子只启动一次。

```bash
bash scripts/thesis_kd/autodl/train_one.sh m_base 42
```

```bash
bash scripts/thesis_kd/autodl/train_core_seed42.sh
bash scripts/thesis_kd/autodl/train_remaining_seeds.sh
bash scripts/thesis_kd/autodl/train_m_scale_ablation.sh
python -m thesis_kd.cli export-commands --include-sensitivity --run-tag study01
```

多卡先在同一独立训练环境设置 `NPROC_PER_NODE` 和可见 GPU，再执行同一训练脚本。脚本使用 `torch.distributed.run --standalone`，由启动器设置 LOCAL_RANK、RANK、WORLD_SIZE。各 rank 读取自己的一份训练批次及本 rank 的冻结教师。32 和 64 都须能被进程数整除。例如四卡分别为每卡训练 8、验证 16；日志记录实际值。数据目录可用 DATA_ROOT，输出根可用 OUTPUT_ROOT，项目根可用 PROJECT_ROOT 覆盖。输出必须在当前 checkout 的授权树内。显存不足时先在全部可比组统一修订方案并标记协议版本；batch 和梯度累积对 BN、实例级增强的影响需要单独验证。
