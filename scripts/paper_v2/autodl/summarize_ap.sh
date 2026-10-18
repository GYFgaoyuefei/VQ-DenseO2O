#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# Offline: read AP records only; no GPU, images, checkpoint or COCO evaluation.
paper summarize-ap "$@"
