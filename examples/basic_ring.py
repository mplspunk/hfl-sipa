"""Basic ring SecAgg demo: 8 satellites, no dropouts, verifies exact
mask cancellation."""

from __future__ import annotations

import numpy as np

from hfl_sipa import begin_round, run_round, setup_plane


def main() -> None:
    rng = np.random.default_rng(42)
    N, D, T = 8, 100_000, 5
    ROUND = 0

    print(f"Plane: N={N} satellites, dim={D:,}, threshold t={T}")
    sats = setup_plane(N, D, T)
    begin_round(sats, ROUND)

    deltas = {i: rng.standard_normal(D) * 0.01 for i in range(N)}
    agg = run_round(sats, deltas, ROUND)

    expected = sum(deltas.values())
    err = np.max(np.abs(agg - expected))
    print(f"max element-wise error: {err:.2e}  (quantization roundoff only)")

    # The cluster head's view of sat 0's contribution is dominated by mask entropy.
    masked0 = sats[0].mask(deltas[0])
    print(f"true update[0]:   mean={deltas[0].mean():+.4f}, std={deltas[0].std():.4f}")
    print(f"masked update[0]: mean={masked0.mean():+.2e}, std={masked0.std():.2e}")


if __name__ == "__main__":
    main()
