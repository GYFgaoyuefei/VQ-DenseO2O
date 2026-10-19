#!/usr/bin/env bash
# Server paths supplied by the user on 2026-10-17; existence is checked on AutoDL.
export TOMATO_DATA_ROOT="/root/autodl-tmp/dataset/tomato_coco"
export DINOV3_WEIGHTS="/root/autodl-tmp/pretrained/deimv2_l/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
export TOMATO_DETECTOR_CHECKPOINT="/root/autodl-tmp/pretrained/deimv2_l/deimv2_dinov3_l_coco.pth"
# Optional when the verified sidecar uses another filename:
# Default: ${TOMATO_DATA_ROOT}/annotations/visible_masks_train.json
# export TOMATO_VISIBLE_MASK_TRAIN="${TOMATO_DATA_ROOT}/annotations/visible_masks_train_vqcr.json"
export PYTHON="python"
# Total train batch = 32 / NPROC_PER_NODE per process. Check VRAM with smoke first.
export NPROC_PER_NODE=1
# Optional: export CUDA_VISIBLE_DEVICES=0,1,2,3
# Optional resume for train_one.sh only:
# export RESUME="/root/autodl-tmp/DEIMv2-new/outputs/paper_v2_beta_select_algorithms/vq_dense_o2o/seed_42/last.pth"
