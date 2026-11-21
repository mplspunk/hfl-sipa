"""Cryptographic primitives and fixed-point quantization.

The quantization is load-bearing for correctness: pair masks are drawn as
int64 vectors, and mask cancellation in the cluster head's sum is bit-exact
only in integer arithmetic. Using floats here would introduce roundoff
indistinguishable to the cluster head from a faulty contribution.
"""

from __future__ import annotations
import hashlib

import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


QUANT_BITS = 16
QUANT_SCALE = 1 << QUANT_BITS


def quantize(delta: np.ndarray) -> np.ndarray:
    """Float update -> int64 fixed-point at 2^QUANT_BITS scale."""
    return np.round(delta * QUANT_SCALE).astype(np.int64)


def dequantize(agg_q: np.ndarray) -> np.ndarray:
    """Inverse of quantize; only quantization roundoff, no cancellation error."""
    return agg_q.astype(np.float64) / QUANT_SCALE


def derive_mask_seed(shared_secret: bytes, round_id: int) -> bytes:
    """Per-round HKDF expansion of a long-term DH secret.

    Per-round derivation is critical: reusing a mask across rounds would
    reveal the *difference* of two rounds' updates, breaking privacy even
    though each round in isolation looks masked.
    """
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=f"hfl-sipa-secagg-round-{round_id}".encode(),
    ).derive(shared_secret)


def prg(seed: bytes, length: int) -> np.ndarray:
    """SHAKE-256 stream expansion into an int64 mask vector.

    Values are drawn from int32 range and widened to int64 so accumulation
    of N masks across the plane cannot overflow. int32 range (~2e9) is far
    larger than quantized gradients (~2^20 after clipping), so the masked
    vector is statistically indistinguishable from uniform to the cluster
    head.
    """
    raw = hashlib.shake_256(seed).digest(length * 4)
    return np.frombuffer(raw, dtype=np.int32).astype(np.int64)
