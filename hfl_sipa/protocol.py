"""Protocol-level helpers: plane provisioning and per-round setup.

In a real deployment these correspond to (a) constellation provisioning,
where public keys are signed and pair secrets established once, and (b) the
per-round bootstrap where each sat generates and distributes fresh Shamir
shares of its self-secret over the ISL fabric.
"""

from __future__ import annotations

import numpy as np

from hfl_sipa.cluster_head import ClusterHead
from hfl_sipa.satellite import Satellite


def setup_plane(plane_size: int, update_dim: int, threshold: int) -> dict[int, Satellite]:
    """Provision a plane: create ``plane_size`` sats and perform ring ECDH.

    In deployment, public keys are provisioned and signed at manufacturing
    time; this function stands in for that bootstrap.
    """
    sats = {i: Satellite(i, plane_size, update_dim, threshold) for i in range(plane_size)}
    pubs = {i: s.public_key for i, s in sats.items()}
    for s in sats.values():
        s.establish_secrets(pubs[s.left_id], pubs[s.right_id])
    return sats


def begin_round(sats: dict[int, Satellite], round_id: int) -> None:
    """Have every sat generate a fresh self-secret and distribute Shamir
    shares to every peer over ISL.

    All splits run before any distribution, so a sat doesn't reset its
    ``_shares_held`` while peers' shares are still in flight -- a subtle
    ordering bug if you interleave the two.
    """
    outgoing_per_sat = {i: s.begin_round(round_id) for i, s in sats.items()}
    for i, outgoing in outgoing_per_sat.items():
        for j, share in outgoing.items():
            sats[j].receive_share(from_sat_id=i, share=share)


def run_round(
    sats: dict[int, Satellite],
    deltas: dict[int, np.ndarray],
    round_id: int,
    dropped: set[int] | None = None,
    silent: set[int] | None = None,
) -> np.ndarray:
    """Drive one full aggregation round end-to-end.

    Convenience wrapper for tests and demos. ``dropped`` sats do not
    contribute a masked update and do not disclose ``b_i``; ``silent`` sats
    contribute the masked update but do not disclose ``b_i`` (recovered via
    Shamir).
    """
    dropped = set() if dropped is None else set(dropped)
    silent = set() if silent is None else set(silent)
    if dropped & silent:
        raise ValueError("a sat cannot be both dropped and silent")

    plane_size = len(sats)
    update_dim = next(iter(sats.values())).update_dim
    threshold = next(iter(sats.values())).threshold

    masked = {i: sats[i].mask(deltas[i]) for i in sats if i not in dropped}
    disclosed = {i: sats[i].disclose_self_secret() for i in masked if i not in silent}

    head = ClusterHead(plane_size, update_dim, threshold)
    return head.aggregate(masked, disclosed, sats, round_id)
