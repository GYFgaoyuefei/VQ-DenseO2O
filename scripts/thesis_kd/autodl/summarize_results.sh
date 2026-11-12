#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
kd summarize --run-tag "$tag" "$@"
