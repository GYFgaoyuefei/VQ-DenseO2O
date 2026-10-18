#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# Usage: bash evaluate_and_summarize_ap.sh [cuda:0] [independent|all] [extra CLI args...]
# Explicit final-Test evaluation of the EMA Validation-selected best.pth; no training.
device="${1:-cuda:0}"
group="${2:-independent}"
models=(baseline_native baseline_maskaware gr_vq li_copyblend dc_sta)
case "$group" in
  independent) ;;
  all) models+=(C1 C2 C3 C4) ;;
  *) echo "Expected group independent or all" >&2; exit 2 ;;
esac
paper evaluate-ap --device "$device" --ap-split test --ap-models "${models[@]}" "${@:3}"
