"""Dropout recovery demo: exercises the five aggregation scenarios."""

from __future__ import annotations

import numpy as np

from hfl_sipa import begin_round, run_round, setup_plane


def main() -> None:
    rng = np.random.default_rng(42)
    N, D, T = 8, 50_000, 5
    ROUND = 0

    print(f"Plane: N={N} sats, threshold t={T}, dim={D:,}")
    print(f"Tolerates up to N - t = {N - T} simultaneous dropouts.\n")

    sats = setup_plane(N, D, T)
    begin_round(sats, ROUND)

    deltas = {i: rng.standard_normal(D) * 0.01 for i in range(N)}

    # Case 1: no dropouts
    agg = run_round(sats, deltas, ROUND)
    expected = sum(deltas.values())
    print(f"Case 1 - no dropouts:")
    print(f"  max error = {np.max(np.abs(agg - expected)):.2e}\n")

    # Case 2: two dropped satellites
    agg = run_round(sats, deltas, ROUND, dropped={2, 6})
    expected = sum(d for i, d in deltas.items() if i not in {2, 6})
    print(f"Case 2 - sats [2, 6] dropped:")
    print(f"  max error vs. survivor sum = {np.max(np.abs(agg - expected)):.2e}\n")

    # Case 3: silent sat -- Shamir recovery
    agg = run_round(sats, deltas, ROUND, silent={3})
    expected = sum(deltas.values())
    print(f"Case 3 - sat 3 silent (Shamir-recovered b_i):")
    print(f"  max error vs. full sum = {np.max(np.abs(agg - expected)):.2e}\n")

    # Case 4: asymmetric-disclosure rule enforcement
    print("Case 4 - protocol refuses unsafe disclosures:")
    alive = set(range(N)) - {2}
    try:
        sats[0].disclose_share(of_sat_id=2, alive_set=alive)
    except PermissionError:
        print(f"  share(b_2) when 2 dropped -> PermissionError (correct)")
    try:
        sats[1].disclose_pair_mask(dropped_neighbor_id=0, alive_set=set(range(N)))
    except PermissionError:
        print(f"  pair_mask(0,1) when 0 alive -> PermissionError (correct)\n")

    # Case 5: too few alive for silent recovery
    print("Case 5 - too few alive sats for silent recovery:")
    try:
        run_round(sats, deltas, ROUND, dropped={0, 1, 2, 3}, silent={4})
    except RuntimeError as e:
        print(f"  RuntimeError: {e}")


if __name__ == "__main__":
    main()
