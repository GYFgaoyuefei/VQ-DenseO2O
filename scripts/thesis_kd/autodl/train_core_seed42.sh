#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
for seed in 42; do
    for experiment in m_base m_kd s_base s_kd; do
        training train --experiment "$experiment" --seed "$seed" --run-tag "$tag" "$@"
    done
done
