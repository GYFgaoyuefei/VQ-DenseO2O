#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
extra=()
if [[ -n "${RESUME:-}" ]]; then extra+=(--resume "$RESUME"); fi
paper train --model "${1:?model required}" --seed "${2:?seed required}" --device "${3:-cuda:0}" --nproc-per-node "${NPROC_PER_NODE:-1}" "${extra[@]}" "${@:4}"
