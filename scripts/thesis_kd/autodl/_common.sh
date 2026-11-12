#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$script_dir/../../.." && pwd -P)}"
cd -- "$PROJECT_ROOT"
export TOMATO_DATA_ROOT="${TOMATO_DATA_ROOT:-${DATA_ROOT:-}}"
export THESIS_KD_OUTPUT_ROOT="${THESIS_KD_OUTPUT_ROOT:-${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/thesis_kd}}"
export PYTHONDONTWRITEBYTECODE=1
python_bin="${PYTHON:-python}"
tag="${RUN_TAG:-study01}"
kd() { "$python_bin" -m thesis_kd.cli "$@"; }
training() {
    if (( ${NPROC_PER_NODE:-1} > 1 )); then
        "$python_bin" -m torch.distributed.run --standalone --nproc-per-node "$NPROC_PER_NODE" -m thesis_kd.cli "$@"
    else
        kd "$@"
    fi
}
