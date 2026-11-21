"""Shamir secret sharing over the 13th Mersenne prime.

Used to split each satellite's per-round self-mask secret b_i across its
plane peers, so the cluster head can recover b_i for any satellite that
was alive during aggregation but became silent before disclosing b_i
directly. See satellite.py and cluster_head.py for how the shares are
used at aggregation time.
"""

from __future__ import annotations
import secrets


SHAMIR_PRIME = 2**521 - 1  # 13th Mersenne prime; comfortably > 2^256


def shamir_split(
    secret: bytes, threshold: int, x_coords: list[int],
) -> list[tuple[int, int]]:
    """Split ``secret`` into ``len(x_coords)`` shares of a threshold scheme.

    Any ``threshold`` shares recover the secret; fewer reveal nothing about
    it (information-theoretically). ``x_coords`` must be distinct and
    nonzero -- we use sat_id + 1 for each so reconstruction unambiguously
    identifies whose share is whose.

    Raises ``ValueError`` on malformed inputs so callers fail fast instead
    of silently generating unrecoverable shares.
    """
    if threshold < 1:
        raise ValueError(f"threshold must be >= 1, got {threshold}")
    if len(x_coords) < threshold:
        raise ValueError(
            f"need at least threshold={threshold} share slots, got {len(x_coords)}")
    if len(set(x_coords)) != len(x_coords):
        raise ValueError("x_coords must be distinct")
    if 0 in x_coords:
        raise ValueError("x_coords must be nonzero (0 is the secret slot)")

    s_int = int.from_bytes(secret, "big")
    if s_int >= SHAMIR_PRIME:
        raise ValueError("secret is too large for the field")

    coefs = [s_int] + [secrets.randbelow(SHAMIR_PRIME) for _ in range(threshold - 1)]
    return [
        (
            x,
            sum(c * pow(x, i, SHAMIR_PRIME) for i, c in enumerate(coefs)) % SHAMIR_PRIME,
        )
        for x in x_coords
    ]


def shamir_reconstruct(shares: list[tuple[int, int]], byte_length: int = 32) -> bytes:
    """Lagrange-interpolate the constant term at x=0.

    Caller is responsible for supplying at least ``threshold`` shares
    (typically enforced by the cluster head via alive-set membership).
    """
    if not shares:
        raise ValueError("need at least one share")
    if len({x for x, _ in shares}) != len(shares):
        raise ValueError("duplicate x-coordinates in shares")

    secret = 0
    for i, (xi, yi) in enumerate(shares):
        num, den = 1, 1
        for j, (xj, _) in enumerate(shares):
            if i == j:
                continue
            num = (num * (-xj)) % SHAMIR_PRIME
            den = (den * (xi - xj)) % SHAMIR_PRIME
        lagrange = (num * pow(den, -1, SHAMIR_PRIME)) % SHAMIR_PRIME
        secret = (secret + yi * lagrange) % SHAMIR_PRIME
    return secret.to_bytes(byte_length, "big")
