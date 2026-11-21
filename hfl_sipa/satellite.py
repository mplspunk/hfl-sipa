"""Ring satellite: per-round self-mask, pair masks with two neighbors,
and enforced asymmetric disclosure."""

from __future__ import annotations
import secrets
from dataclasses import dataclass, field

import numpy as np
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey, X25519PublicKey,
)

from hfl_sipa.crypto import derive_mask_seed, prg, quantize
from hfl_sipa.shamir import shamir_split


@dataclass
class Satellite:
    """A satellite in one orbital plane's SecAgg ring.

    Long-term state (set once at provisioning): an X25519 keypair and
    ECDH-derived shared secrets with the two ring neighbors.

    Per-round state (refreshed by ``begin_round``): a fresh 32-byte
    self-secret ``b_i``, and Shamir shares of every other sat's ``b_j``.
    """

    sat_id: int
    plane_size: int
    update_dim: int
    threshold: int
    private_key: X25519PrivateKey = field(default_factory=X25519PrivateKey.generate)

    left_secret: bytes = b""
    right_secret: bytes = b""

    _round_id: int = -1
    _self_secret: bytes = b""
    _shares_held: dict[int, tuple[int, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 <= self.sat_id < self.plane_size:
            raise ValueError(f"sat_id {self.sat_id} outside [0, {self.plane_size})")
        if self.plane_size < 3:
            raise ValueError("ring needs at least 3 sats")
        if not 1 <= self.threshold <= self.plane_size - 1:
            raise ValueError(
                f"threshold must be in [1, {self.plane_size - 1}], got {self.threshold}")

    @property
    def public_key(self) -> X25519PublicKey:
        return self.private_key.public_key()

    @property
    def left_id(self) -> int:
        return (self.sat_id - 1) % self.plane_size

    @property
    def right_id(self) -> int:
        return (self.sat_id + 1) % self.plane_size

    @property
    def shamir_x(self) -> int:
        """Shamir x-coordinate for this sat's shares (nonzero)."""
        return self.sat_id + 1

    def establish_secrets(
        self, left_pub: X25519PublicKey, right_pub: X25519PublicKey,
    ) -> None:
        """One-time ECDH with both ring neighbors."""
        self.left_secret = self.private_key.exchange(left_pub)
        self.right_secret = self.private_key.exchange(right_pub)

    def _pair_sign(self, neighbor_id: int) -> int:
        """Sign convention: lower id adds, higher id subtracts. Both endpoints
        of a pair derive the same mask from their shared secret, so the pair
        contributes +M and -M to the plane sum and cancels."""
        return +1 if self.sat_id < neighbor_id else -1

    def begin_round(self, round_id: int) -> dict[int, tuple[int, int]]:
        """Generate a fresh self-secret and Shamir-split it. Returns
        ``{recipient_sat_id: share}`` for distribution over ISL."""
        self._round_id = round_id
        self._self_secret = secrets.token_bytes(32)
        self._shares_held = {}

        recipients = [j for j in range(self.plane_size) if j != self.sat_id]
        x_coords = [j + 1 for j in recipients]  # matches recipients' shamir_x
        shares = shamir_split(self._self_secret, self.threshold, x_coords)
        return dict(zip(recipients, shares))

    def receive_share(self, from_sat_id: int, share: tuple[int, int]) -> None:
        self._shares_held[from_sat_id] = share

    def mask(self, delta: np.ndarray) -> np.ndarray:
        """Quantize the local update; add pair masks and the self-mask.

        Returns an int64 vector. The cluster head cannot distinguish this
        from a uniformly random int64 vector without the pair-mask cancellations
        (from the plane sum) and the self-mask disclosure (direct or Shamir).
        """
        if delta.shape != (self.update_dim,):
            raise ValueError(
                f"expected shape ({self.update_dim},), got {delta.shape}")
        if self._round_id < 0:
            raise RuntimeError("call begin_round before mask")

        delta_q = quantize(delta)
        right = prg(derive_mask_seed(self.right_secret, self._round_id), self.update_dim)
        left = prg(derive_mask_seed(self.left_secret, self._round_id), self.update_dim)
        self_m = prg(derive_mask_seed(self._self_secret, self._round_id), self.update_dim)
        return (
            delta_q
            + self._pair_sign(self.right_id) * right
            + self._pair_sign(self.left_id) * left
            + self_m
        )

    # ---- Enforced asymmetric disclosure ---- #
    #
    # A satellite is in exactly one state per round: alive (its masked update
    # was received) or dropped. The disclosure methods enforce this
    # partitioning to defeat late-arrival unmasking by an honest-but-curious
    # cluster head.

    def disclose_self_secret(self) -> bytes:
        """Alive sat reveals its own b_i so the cluster head can strip its
        self-mask from the aggregate."""
        return self._self_secret

    def disclose_share(
        self, of_sat_id: int, alive_set: set[int],
    ) -> tuple[int, int]:
        """Reveal a Shamir share of another sat's b_j -- ONLY if that sat is
        declared alive. If j is dropped, releasing shares of b_j would let
        the cluster head reconstruct b_j and unmask any late update from j."""
        if of_sat_id not in alive_set:
            raise PermissionError(
                f"sat {of_sat_id} is in the dropped set; refusing to disclose "
                f"share of its b_i (would enable late-arrival unmasking)")
        if of_sat_id not in self._shares_held:
            raise KeyError(f"no share of sat {of_sat_id}'s b_i held")
        return self._shares_held[of_sat_id]

    def disclose_pair_mask(
        self, dropped_neighbor_id: int, alive_set: set[int],
    ) -> np.ndarray:
        """Reveal the pair-mask vector shared with a dropped neighbor.
        Refused if the neighbor is alive: pair masks for alive sats cancel
        naturally in the plane sum, and disclosing them would only help an
        attacker."""
        if dropped_neighbor_id in alive_set:
            raise PermissionError(
                f"neighbor {dropped_neighbor_id} is alive; refusing to disclose")
        if dropped_neighbor_id == self.right_id:
            secret = self.right_secret
        elif dropped_neighbor_id == self.left_id:
            secret = self.left_secret
        else:
            raise ValueError(
                f"sat {self.sat_id} has no ring pair with {dropped_neighbor_id}")
        return prg(derive_mask_seed(secret, self._round_id), self.update_dim)
