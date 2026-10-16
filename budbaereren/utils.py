"""Small shared helpers."""

from __future__ import annotations

import random

import numpy as np


def seed_everything(seed: int) -> np.random.Generator:
    """Seed Python's and NumPy's global RNGs and return a fresh Generator.

    All library code takes an explicit ``np.random.Generator``; the global seeds
    are set only as a safety net for anything that slips through.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - deliberate global safety net
    return np.random.default_rng(seed)
