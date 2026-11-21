"""Tests for Shamir secret sharing."""

from __future__ import annotations
import secrets

import pytest

from hfl_sipa.shamir import shamir_reconstruct, shamir_split


class TestShamirRoundtrip:
    def test_reconstruct_with_exact_threshold_shares(self):
        secret = secrets.token_bytes(32)
        shares = shamir_split(secret, threshold=3, x_coords=[1, 2, 3, 4, 5])
        for chosen in [shares[:3], shares[1:4], shares[2:5], [shares[0], shares[2], shares[4]]]:
            assert shamir_reconstruct(chosen) == secret

    def test_reconstruct_with_more_than_threshold_shares(self):
        secret = secrets.token_bytes(32)
        shares = shamir_split(secret, threshold=3, x_coords=[1, 2, 3, 4, 5])
        assert shamir_reconstruct(shares) == secret

    def test_different_secrets_produce_different_shares(self):
        # Same threshold + x_coords + different secrets -> different y values.
        a = shamir_split(b"a" * 32, 2, [1, 2, 3])
        b = shamir_split(b"b" * 32, 2, [1, 2, 3])
        assert [y for _, y in a] != [y for _, y in b]


class TestShamirValidation:
    def test_rejects_zero_threshold(self):
        with pytest.raises(ValueError, match="threshold"):
            shamir_split(b"\x00" * 32, threshold=0, x_coords=[1, 2])

    def test_rejects_too_few_share_slots(self):
        with pytest.raises(ValueError, match="threshold"):
            shamir_split(b"\x00" * 32, threshold=3, x_coords=[1, 2])

    def test_rejects_duplicate_x_coords(self):
        with pytest.raises(ValueError, match="distinct"):
            shamir_split(b"\x00" * 32, threshold=2, x_coords=[1, 1, 2])

    def test_rejects_zero_x_coord(self):
        # x=0 is the secret slot; using it as a share slot would leak the secret directly.
        with pytest.raises(ValueError, match="nonzero"):
            shamir_split(b"\x00" * 32, threshold=2, x_coords=[0, 1, 2])

    def test_reconstruct_rejects_duplicate_share_x(self):
        with pytest.raises(ValueError, match="duplicate"):
            shamir_reconstruct([(1, 100), (1, 200)])

    def test_reconstruct_rejects_empty(self):
        with pytest.raises(ValueError):
            shamir_reconstruct([])


class TestShamirRandomization:
    def test_two_splits_of_same_secret_produce_different_shares(self):
        # Coefficients are random, so the y values differ round-to-round
        # even though both sets reconstruct to the same secret. This is
        # what makes per-round splitting safe.
        secret = b"same" + b"\x00" * 28
        a = shamir_split(secret, 3, [1, 2, 3, 4])
        b = shamir_split(secret, 3, [1, 2, 3, 4])
        assert [y for _, y in a] != [y for _, y in b]
        assert shamir_reconstruct(a[:3]) == secret
        assert shamir_reconstruct(b[:3]) == secret
