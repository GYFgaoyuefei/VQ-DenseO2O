#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
paper preflight --model "${1:-all}" "${@:2}"
