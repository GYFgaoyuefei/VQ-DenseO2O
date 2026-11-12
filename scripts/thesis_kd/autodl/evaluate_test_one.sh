#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
experiment="${1:?Usage: bash evaluate_test_one.sh EXPERIMENT [SEED] [CLI OPTIONS...]}"
shift
seed=42
if [[ ${1:-} =~ ^[0-9]+$ ]]; then seed="$1"; shift; fi
kd evaluate-test --experiment "$experiment" --seed "$seed" --run-tag "$tag" "$@"
