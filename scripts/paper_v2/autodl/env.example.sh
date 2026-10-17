#!/usr/bin/env bash
# Copy to env.sh, fill actual local weight paths, then source it.
export TOMATO_DATA_ROOT="/root/autodl-tmp/dataset/tomato_coco"
export DINOV3_WEIGHTS="/ABSOLUTE/PATH/TO/YOUR/DINOV3_WEIGHTS.pth"
export TOMATO_DETECTOR_CHECKPOINT="/ABSOLUTE/PATH/TO/YOUR/DETECTOR_CHECKPOINT.pth"
# Optional when the verified sidecar uses another filename:
# export TOMATO_VISIBLE_MASK_TRAIN="/ABSOLUTE/PATH/TO/visible_masks_train_vqcr.json"
export PYTHON="python"
# Total train batch = 32 / NPROC_PER_NODE per process. Check VRAM with smoke first.
export NPROC_PER_NODE=1
# Optional: export CUDA_VISIBLE_DEVICES=0,1,2,3
# Optional resume for train_one.sh only:
# export RESUME="/root/autodl-tmp/DEIMv2-new/outputs/paper_v2/gr_vq/seed_42/last.pth"
