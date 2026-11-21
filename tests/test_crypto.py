"""Tests for crypto primitives."""

from __future__ import annotations

import numpy as np
import pytest

from hfl_sipa.crypto import (
    QUANT_BITS, QUANT_SCALE,
    derive_mask_seed, prg, quantize, dequantize,
)


class TestQuantization:
    def test_roundtrip_bit_exact_for_representable_values(self):
        # Values on the quantization grid roundtrip exactly.
        grid = np.array([i / QUANT_SCALE for i in range(-10, 11)])
        assert np.array_equal(dequantize(quantize(grid)), grid)

    def test_roundtrip_error_bounded_by_half_lsb(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(-1, 1, size=10_000)
        err = np.max(np.abs(dequantize(quantize(x)) - x))
        assert err <= 0.5 / QUANT_SCALE + 1e-15

    def test_output_dtype_is_int64(self):
        # int64 is load-bearing: mask cancellation only exact in integer math.
        assert quantize(np.array([0.1, 0.2])).dtype == np.int64


class TestPRG:
    def test_deterministic_from_seed(self):
        s = b"\x00" * 32
        assert np.array_equal(prg(s, 100), prg(s, 100))

    def test_different_seeds_diverge(self):
        a = prg(b"\x00" * 32, 100)
        b = prg(b"\x01" + b"\x00" * 31, 100)
        assert not np.array_equal(a, b)

    def test_output_length(self):
        assert prg(b"seed" + b"\x00" * 28, 42).shape == (42,)

    def test_output_dtype_is_int64(self):
        assert prg(b"seed" + b"\x00" * 28, 10).dtype == np.int64


class TestMaskSeedDerivation:
    def test_deterministic_per_round(self):
        secret = b"shared" + b"\x00" * 26
        assert derive_mask_seed(secret, 5) == derive_mask_seed(secret, 5)

    def test_different_rounds_produce_different_seeds(self):
        # Cross-round mask reuse would leak the delta of two rounds' updates.
        secret = b"shared" + b"\x00" * 26
        assert derive_mask_seed(secret, 0) != derive_mask_seed(secret, 1)

    def test_different_shared_secrets_produce_different_seeds(self):
        assert derive_mask_seed(b"a" * 32, 0) != derive_mask_seed(b"b" * 32, 0)
