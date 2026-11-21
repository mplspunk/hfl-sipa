"""Tests for dropout recovery and the asymmetric-disclosure security rule."""

from __future__ import annotations

import numpy as np
import pytest

from hfl_sipa import ClusterHead, begin_round, run_round, setup_plane
from hfl_sipa.crypto import QUANT_SCALE


class TestDroppedSats:
    def test_single_dropout_recovers_survivor_sum(self):
        rng = np.random.default_rng(0)
        N, D, T = 8, 500, 5
        sats = setup_plane(N, D, T)
        begin_round(sats, round_id=0)
        deltas = {i: rng.standard_normal(D) * 0.01 for i in range(N)}
        agg = run_round(sats, deltas, round_id=0, dropped={3})
        expected = sum(d for i, d in deltas.items() if i != 3)
        assert np.max(np.abs(agg - expected)) < N / QUANT_SCALE

    def test_two_nonadjacent_dropouts(self):
        rng = np.random.default_rng(1)
        N, D, T = 8, 500, 5
        sats = setup_plane(N, D, T)
        begin_round(sats, round_id=0)
        deltas = {i: rng.standard_normal(D) * 0.01 for i in range(N)}
        agg = run_round(sats, deltas, round_id=0, dropped={2, 6})
        expected = sum(d for i, d in deltas.items() if i not in {2, 6})
        assert np.max(np.abs(agg - expected)) < N / QUANT_SCALE

    def test_two_adjacent_dropouts(self):
        # Adjacent dropouts create one shared pair edge between two dropped
        # sats -- neither contributes, so nothing to disclose.
        rng = np.random.default_rng(2)
        N, D, T = 8, 500, 5
        sats = setup_plane(N, D, T)
        begin_round(sats, round_id=0)
        deltas = {i: rng.standard_normal(D) * 0.01 for i in range(N)}
        agg = run_round(sats, deltas, round_id=0, dropped={2, 3})
        expected = sum(d for i, d in deltas.items() if i not in {2, 3})
        assert np.max(np.abs(agg - expected)) < N / QUANT_SCALE

    def test_wraparound_dropout(self):
        # Dropped sat 0 has neighbors N-1 and 1 -- test the modular boundary.
        rng = np.random.default_rng(3)
        N, D, T = 6, 500, 3
        sats = setup_plane(N, D, T)
        begin_round(sats, round_id=0)
        deltas = {i: rng.standard_normal(D) * 0.01 for i in range(N)}
        agg = run_round(sats, deltas, round_id=0, dropped={0})
        expected = sum(d for i, d in deltas.items() if i != 0)
        assert np.max(np.abs(agg - expected)) < N / QUANT_SCALE


class TestSilentSats:
    def test_silent_sat_recovered_via_shamir(self):
        # "Silent" = sat sent masked update but didn't disclose b_i. All N
        # deltas should still be summed after Shamir reconstruction.
        rng = np.random.default_rng(0)
        N, D, T = 8, 500, 5
        sats = setup_plane(N, D, T)
        begin_round(sats, round_id=0)
        deltas = {i: rng.standard_normal(D) * 0.01 for i in range(N)}
        agg = run_round(sats, deltas, round_id=0, silent={3})
        expected = sum(deltas.values())
        assert np.max(np.abs(agg - expected)) < N / QUANT_SCALE

    def test_multiple_silent_sats(self):
        rng = np.random.default_rng(1)
        N, D, T = 8, 500, 3
        sats = setup_plane(N, D, T)
        begin_round(sats, round_id=0)
        deltas = {i: rng.standard_normal(D) * 0.01 for i in range(N)}
        agg = run_round(sats, deltas, round_id=0, silent={1, 4, 7})
        expected = sum(deltas.values())
        assert np.max(np.abs(agg - expected)) < N / QUANT_SCALE

    def test_mixed_dropped_and_silent(self):
        rng = np.random.default_rng(2)
        N, D, T = 8, 500, 4
        sats = setup_plane(N, D, T)
        begin_round(sats, round_id=0)
        deltas = {i: rng.standard_normal(D) * 0.01 for i in range(N)}
        agg = run_round(sats, deltas, round_id=0, dropped={2}, silent={5})
        expected = sum(d for i, d in deltas.items() if i != 2)
        assert np.max(np.abs(agg - expected)) < N / QUANT_SCALE


class TestAsymmetricDisclosureRule:
    """The security-critical enforcement: honest peers refuse disclosures
    that would enable an honest-but-curious cluster head to unmask a
    late-arriving update."""

    def test_share_disclosure_refused_for_dropped_sat(self):
        N, D, T = 6, 100, 3
        sats = setup_plane(N, D, T)
        begin_round(sats, round_id=0)
        alive = {0, 1, 3, 4, 5}  # sat 2 dropped
        with pytest.raises(PermissionError, match="dropped set"):
            sats[0].disclose_share(of_sat_id=2, alive_set=alive)

    def test_pair_mask_disclosure_refused_for_alive_neighbor(self):
        N, D, T = 6, 100, 3
        sats = setup_plane(N, D, T)
        begin_round(sats, round_id=0)
        alive = set(range(N))  # all alive
        with pytest.raises(PermissionError, match="alive"):
            sats[1].disclose_pair_mask(dropped_neighbor_id=0, alive_set=alive)

    def test_pair_mask_disclosure_requires_actual_ring_neighbor(self):
        N, D, T = 6, 100, 3
        sats = setup_plane(N, D, T)
        begin_round(sats, round_id=0)
        alive = {0, 1, 2, 3, 4}  # sat 5 dropped (not adjacent to sat 2)
        with pytest.raises(ValueError, match="no ring pair"):
            sats[2].disclose_pair_mask(dropped_neighbor_id=5, alive_set=alive)


class TestAbortConditions:
    def test_too_few_alive_for_silent_recovery_aborts(self):
        N, D, T = 8, 200, 5
        sats = setup_plane(N, D, T)
        begin_round(sats, round_id=0)
        rng = np.random.default_rng(0)
        deltas = {i: rng.standard_normal(D) * 0.01 for i in range(N)}
        # 4 alive, 1 of them silent -> can't Shamir-reconstruct (need T=5).
        with pytest.raises(RuntimeError, match="threshold"):
            run_round(sats, deltas, round_id=0, dropped={0, 1, 2, 3}, silent={4})

    def test_low_alive_count_ok_if_no_silent(self):
        # If everyone alive disclosed their own b_i, Shamir isn't needed.
        N, D, T = 8, 200, 5
        sats = setup_plane(N, D, T)
        begin_round(sats, round_id=0)
        rng = np.random.default_rng(0)
        deltas = {i: rng.standard_normal(D) * 0.01 for i in range(N)}
        agg = run_round(sats, deltas, round_id=0, dropped={0, 1, 2, 3})
        expected = sum(deltas[i] for i in {4, 5, 6, 7})
        assert np.max(np.abs(agg - expected)) < N / QUANT_SCALE

    def test_dropped_and_silent_cannot_overlap(self):
        N, D, T = 6, 100, 3
        sats = setup_plane(N, D, T)
        begin_round(sats, round_id=0)
        rng = np.random.default_rng(0)
        deltas = {i: rng.standard_normal(D) * 0.01 for i in range(N)}
        with pytest.raises(ValueError, match="both dropped and silent"):
            run_round(sats, deltas, round_id=0, dropped={2}, silent={2})
