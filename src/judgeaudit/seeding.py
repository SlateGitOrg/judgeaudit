"""Stable, order-independent random streams derived from string keys."""

from __future__ import annotations

import hashlib

import numpy as np


def stable_rng(*parts: object) -> np.random.Generator:
    """A generator seeded from the SHA-256 of the parts.

    Unlike Python's ``hash``, this is identical across processes and platforms, so a draw keyed by
    (seed, pair id, order) never depends on the order in which pairs are processed.
    """
    key = "|".join(str(p) for p in parts).encode()
    seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    return np.random.default_rng(seed)
