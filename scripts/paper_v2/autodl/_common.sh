#!/usr/bin/env bash
set -euo pipefail
PAPER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PAPER_ROOT"
paper() { "${PYTHON:-python}" tools/paper_v2/run.py "$@"; }
