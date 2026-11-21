"""Cluster head: sums masked updates, strips self-masks, strips orphaned
pair masks. Honest-but-curious; sees only the plane-level aggregate."""

from __future__ import annotations

import numpy as np

from hfl_sipa.crypto import derive_mask_seed, prg, dequantize
from hfl_sipa.satellite import Satellite
from hfl_sipa.shamir import shamir_reconstruct


class ClusterHead:
    """Rotates each round to the sat with the soonest ground pass.

    The cluster head runs the aggregation but never sees individual updates:
    incoming masked vectors are indistinguishable from noise, and only their
    sum -- after mask cancellation and disclosed corrections -- is meaningful.
    """

    def __init__(self, plane_size: int, update_dim: int, threshold: int):
        if plane_size < 3:
            raise ValueError("ring needs at least 3 sats")
        if not 1 <= threshold <= plane_size - 1:
            raise ValueError(
                f"threshold must be in [1, {plane_size - 1}], got {threshold}")
        self.plane_size = plane_size
        self.update_dim = update_dim
        self.threshold = threshold

    def aggregate(
        self,
        masked_updates: dict[int, np.ndarray],
        disclosed_b: dict[int, bytes],
        sats: dict[int, Satellite],
        round_id: int,
    ) -> np.ndarray:
        """Return sum of the underlying deltas for satellites in
        ``masked_updates``.

        Parameters
        ----------
        masked_updates
            {sat_id: masked int64 vector} for every alive sat.
        disclosed_b
            {sat_id: b_i} for alive sats that self-disclosed. Alive sats
            missing from this dict are treated as silent and their b_i is
            reconstructed from Shamir shares held by other alive sats.
        sats
            {sat_id: Satellite} for the whole plane. Used to invoke the
            enforced disclosure methods on peers.
        round_id
            Must match the round the masks were generated for.

        Raises
        ------
        RuntimeError
            If fewer than ``threshold`` sats are alive -- no round result
            possible, aggregation must be aborted (model does not update
            this round).
        PermissionError
            Propagated from ``Satellite.disclose_*`` when the (alive,
            dropped) partition is inconsistent. Aggregation aborts.
        """
        alive_set = set(masked_updates.keys())
        dropped_set = set(range(self.plane_size)) - alive_set
        silent_set = alive_set - set(disclosed_b.keys())

        if not alive_set.issubset(set(sats.keys())):
            raise ValueError("masked_updates references unknown sats")
        if not alive_set:
            raise RuntimeError("no alive sats; nothing to aggregate")
        if len(alive_set) < self.threshold and silent_set:
            raise RuntimeError(
                f"only {len(alive_set)} alive sats but {len(silent_set)} silent; "
                f"need >= threshold={self.threshold} to reconstruct any silent b_i")

        # 1) Sum received masked vectors in int64. Pair masks between two
        # alive neighbors cancel here; masks between an alive sat and a
        # dropped neighbor are left orphaned for step (3).
        agg_q = np.sum(list(masked_updates.values()), axis=0)

        # 2) Strip the self-mask for every alive sat.
        for i, b_i in disclosed_b.items():
            if i not in alive_set:
                raise ValueError(f"disclosed b for non-alive sat {i}")
            agg_q -= prg(derive_mask_seed(b_i, round_id), self.update_dim)

        for i in silent_set:
            # Sat i sent a masked update but did not disclose b_i (link went
            # down after aggregation, say). Reconstruct via Shamir from any
            # threshold alive peers -- disclose_share enforces that i is
            # in alive_set, so this path cannot unmask a dropped sat.
            shares: list[tuple[int, int]] = []
            for j in alive_set - {i}:
                shares.append(sats[j].disclose_share(of_sat_id=i, alive_set=alive_set))
                if len(shares) >= self.threshold:
                    break
            b_i = shamir_reconstruct(shares)
            agg_q -= prg(derive_mask_seed(b_i, round_id), self.update_dim)

        # 3) Strip orphaned pair masks at every alive-dropped ring boundary.
        # Each surviving neighbor of a dropped sat discloses the pair mask
        # it used, signed correctly per pair_sign convention. Pair masks
        # between two dropped sats cancel by neither contributing.
        for k in dropped_set:
            for nbr in [(k - 1) % self.plane_size, (k + 1) % self.plane_size]:
                if nbr not in alive_set:
                    continue
                pair_mask = sats[nbr].disclose_pair_mask(
                    dropped_neighbor_id=k, alive_set=alive_set)
                sign = +1 if nbr < k else -1
                agg_q -= sign * pair_mask

        return dequantize(agg_q)
