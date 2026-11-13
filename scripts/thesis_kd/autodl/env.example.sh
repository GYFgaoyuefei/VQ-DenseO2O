#!/usr/bin/env bash
# 在独立副本中 source 此文件，然后填入实际的本地资源。
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)}"
export TOMATO_DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/dataset/tomato_coco}"
export THESIS_KD_OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/thesis_kd}"
export RUN_TAG="${RUN_TAG:-study01}"
export PYTHON="${PYTHON:-python}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
# AutoDL 上的学生预训练权重和教师检查点
export THESIS_KD_M_BACKBONE_WEIGHTS=/root/autodl-tmp/pretrained/thesis_kd/deimv2_m/vittplus_distill.pt
export THESIS_KD_M_DETECTOR_CHECKPOINT=/root/autodl-tmp/pretrained/thesis_kd/deimv2_m/deimv2_dinov3_m_coco.pth
export THESIS_KD_S_BACKBONE_WEIGHTS=/root/autodl-tmp/pretrained/thesis_kd/deimv2_s/vitt_distill.pt
export THESIS_KD_S_DETECTOR_CHECKPOINT=/root/autodl-tmp/pretrained/thesis_kd/deimv2_s/deimv2_dinov3_s_coco.pth
export THESIS_KD_TEACHER_CONFIG=/root/autodl-tmp/VQ-DenseO2O-thesis-kd/configs/tomato/paper_v2/vq_dense_o2o.yml
export THESIS_KD_TEACHER_CHECKPOINT=/root/autodl-tmp/pretrained/thesis_kd/vq_dense_o2o/seed_3407/best.pth
# 可选：通过 THESIS_KD_TEACHER_VALIDATION 记录选择该检查点的依据。
