#!/usr/bin/env bash
# 在独立副本中 source 此文件，然后填入实际的本地资源。
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)}"
export TOMATO_DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/dataset/tomato_coco}"
export THESIS_KD_OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/thesis_kd}"
export RUN_TAG="${RUN_TAG:-study01}"
export PYTHON="${PYTHON:-python}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
# 示例路径；请核对实际文件
export THESIS_KD_M_BACKBONE_WEIGHTS=/root/autodl-tmp/pretrained/m/vittplus_distill.pt
export THESIS_KD_M_DETECTOR_CHECKPOINT=/root/autodl-tmp/pretrained/m/deimv2_dinov3_m_coco.pth
export THESIS_KD_S_BACKBONE_WEIGHTS=/root/autodl-tmp/pretrained/s/vitt_distill.pt
export THESIS_KD_S_DETECTOR_CHECKPOINT=/root/autodl-tmp/pretrained/s/deimv2_dinov3_s_coco.pth
# 进行 KD 前还须 export THESIS_KD_TEACHER_CONFIG、THESIS_KD_TEACHER_CHECKPOINT、
# THESIS_KD_TEACHER_VALIDATION，指向指定 T0 的真实配置、权重并写明 Validation 依据。
