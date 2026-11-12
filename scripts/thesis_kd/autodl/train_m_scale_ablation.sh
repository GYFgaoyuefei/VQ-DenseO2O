#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
for seed in 42; do
    for experiment in m_kd_p3 m_kd_p4 m_kd_p5; do
        training train --experiment "$experiment" --seed "$seed" --run-tag "$tag" "$@"
    done
done
