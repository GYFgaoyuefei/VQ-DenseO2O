#!/usr/bin/env bash
# 9 models x 3 seeds: explicit Test inference, fixed efficiency, final paper tables.
# Usage: bash scripts/paper_v2/autodl/evaluate_final_test_all.sh [cuda:0] [options...]
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
device="${1:-cuda:0}"
if [[ $# -gt 0 ]]; then shift; fi
"${PYTHON:-python}" tools/paper_v2/final_test.py --device "$device" "$@"
