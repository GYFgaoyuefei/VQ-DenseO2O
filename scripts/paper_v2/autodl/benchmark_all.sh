#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# Usage: bash benchmark_all.sh [cuda:0] [independent|all] [extra CLI args...]
# Each formal checkpoint must already exist; failures stop this serial campaign.
device="${1:-cuda:0}"
group="${2:-independent}"
models=(baseline_native baseline_maskaware vq_dense_o2o li_copyblend dc_sta)
case "$group" in
  independent) ;;
  all) models+=(vq_dense_o2o_li_copyblend vq_dense_o2o_dc_sta li_copyblend_dc_sta vq_dense_o2o_li_copyblend_dc_sta) ;;
  *) echo "Expected group independent or all" >&2; exit 2 ;;
esac
for model in "${models[@]}"; do
  for seed in 0 42 3407; do
    paper benchmark --model "$model" --seed "$seed" --device "$device" "${@:3}"
  done
done
