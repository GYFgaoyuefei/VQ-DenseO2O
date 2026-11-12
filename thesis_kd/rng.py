"""Preserve all parent-process RNG streams across auxiliary construction."""
import random
from contextlib import contextmanager

import numpy as np
import torch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def capture_rng():
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


@contextmanager
def isolated_rng():
    state = capture_rng()
    try:
        yield
    finally:
        restore_rng(state)
