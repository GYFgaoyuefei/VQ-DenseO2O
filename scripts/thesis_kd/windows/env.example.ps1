$env:PROJECT_ROOT = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$env:TOMATO_DATA_ROOT = 'D:/workspace/python_workspace/study/tomato_data/data_new/tomato_coco'
$env:THESIS_KD_OUTPUT_ROOT = Join-Path $env:PROJECT_ROOT 'outputs/thesis_kd'
$env:PYTHONDONTWRITEBYTECODE = '1'
# 按实际路径设置 M/S 各自的 THESIS_KD_{M,S}_{BACKBONE_WEIGHTS,DETECTOR_CHECKPOINT}
# KD 还需要 THESIS_KD_TEACHER_CONFIG、THESIS_KD_TEACHER_CHECKPOINT、THESIS_KD_TEACHER_VALIDATION。
