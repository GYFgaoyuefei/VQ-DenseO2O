#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# Real Train only; formal batch32 (32/NPROC_PER_NODE per rank), three steps.
paper smoke-train --model "${1:?model required}" --seed "${2:-42}" \
  --device "${3:-cuda:0}" --nproc-per-node "${NPROC_PER_NODE:-1}" "${@:4}"
