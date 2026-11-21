"""End-to-end tests for the SecAgg ring: correctness and mask hiding."""

from __future__ import annotations

import numpy as np
import pytest

from hfl_sipa import begin_round, run_round, setup_plane
from hfl_sipa.crypto import QUANT_SCALE


class TestCorrectness:
    @pytest.mark.parametrize("plane_size", [3, 4, 8, 16])
    def test_aggregate_matches_true_sum(self, plane_size):
        rng = np.random.default_rng(42)
        D = 1000
        sats = setup_plane(plane_size, D, threshold=max(2, plane_size // 2))
        begin_round(sats, round_id=0)
        deltas = {i: rng.standard_normal(D) * 0.01 for i in range(plane_size)}
        agg = run_round(sats, deltas, round_id=0)
        expected = sum(deltas.values())
        # Only quantization roundoff -- cancellation is bit-exact in int64.
        # Bound: N independently quantized floats accumulate up to N * 0.5 LSB
        # per element in the worst case.
        assert np.max(np.abs(agg - expected)) < plane_size / QUANT_SCALE

    def test_multiple_rounds_use_fresh_masks(self):
        rng = np.random.default_rng(1)
        N, D = 6, 500
        sats = setup_plane(N, D, threshold=3)
        deltas = {i: rng.standard_normal(D) * 0.01 for i in range(N)}
        expected = sum(deltas.values())

        for r in range(3):
            begin_round(sats, round_id=r)
            agg = run_round(sats, deltas, round_id=r)
            assert np.max(np.abs(agg - expected)) < N / QUANT_SCALE


class TestMaskHiding:
    def test_masked_vector_does_not_reveal_delta(self):
        # Masked vector should be statistically indistinguishable from noise
        # to anyone without the pair-mask cancellations. Concretely, the
        # magnitudes should be dominated by the mask, not the gradient.
        rng = np.random.default_rng(0)
        N, D = 8, 5_000
        sats = setup_plane(N, D, threshold=4)
        begin_round(sats, round_id=0)
        delta = rng.standard_normal(D) * 0.01
        masked = sats[0].mask(delta)

        # Gradient std ~ 0.01 * QUANT_SCALE ~ 650 after quantization.
        # Mask std ~ int32 std ~ 1.2e9. Ratio should be enormous.
        assert masked.std() > 1e6 * (0.01 * QUANT_SCALE)

    def test_masks_change_across_rounds(self):
        # Same delta, different rounds -> different masked vectors, or
        # cross-round mask reuse would leak the delta directly.
        N, D = 5, 500
        sats = setup_plane(N, D, threshold=3)
        delta = np.zeros(D)  # deterministic input isolates the mask
        begin_round(sats, round_id=0)
        m0 = sats[0].mask(delta)
        begin_round(sats, round_id=1)
        m1 = sats[0].mask(delta)
        assert not np.array_equal(m0, m1)


class TestValidation:
    def test_invalid_plane_size_rejected(self):
        with pytest.raises(ValueError, match="ring needs at least 3"):
            setup_plane(2, 100, threshold=1)

    def test_invalid_threshold_rejected(self):
        with pytest.raises(ValueError, match="threshold"):
            setup_plane(5, 100, threshold=5)  # must be <= N - 1
        with pytest.raises(ValueError, match="threshold"):
            setup_plane(5, 100, threshold=0)

    def test_wrong_shape_delta_rejected(self):
        sats = setup_plane(4, 100, threshold=2)
        begin_round(sats, round_id=0)
        with pytest.raises(ValueError, match="shape"):
            sats[0].mask(np.zeros(50))

    def test_mask_before_begin_round_rejected(self):
        sats = setup_plane(4, 100, threshold=2)
        with pytest.raises(RuntimeError, match="begin_round"):
            sats[0].mask(np.zeros(100))
