#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
paper validate --model "${1:?model required}" --seed "${2:-42}" --device "${3:-cuda:0}" "${@:4}"
