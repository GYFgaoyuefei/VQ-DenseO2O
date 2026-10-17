#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# Usage: bash benchmark_all.sh [cuda:0] [independent|all] [extra CLI args...]
# Each formal checkpoint must already exist; failures stop this serial campaign.
device="${1:-cuda:0}"
group="${2:-independent}"
models=(baseline_native baseline_maskaware gr_vq li_copyblend dc_sta)
case "$group" in
  independent) ;;
  all) models+=(C1 C2 C3 C4) ;;
  *) echo "Expected group independent or all" >&2; exit 2 ;;
esac
for model in "${models[@]}"; do
  for seed in 0 42 3407; do
    paper benchmark --model "$model" --seed "$seed" --device "$device" "${@:3}"
  done
done
