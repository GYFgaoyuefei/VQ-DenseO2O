#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
if [[ -n "${RESUME:-}" ]]; then echo "Unset RESUME for a batch launch; resume runs with train_one.sh." >&2; exit 2; fi
for model in baseline_native baseline_maskaware gr_vq li_copyblend dc_sta; do
  bash scripts/paper_v2/autodl/train_one.sh "$model" 42 "${1:-cuda:0}"
done
