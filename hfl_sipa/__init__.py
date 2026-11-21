"""HFL-SIPA: Hierarchical Federated Learning with Secure Intra-Plane Aggregation.

Research prototype for privacy-preserving federated learning on LEO satellite
constellations. See README for design and threat model.
"""

from hfl_sipa.cluster_head import ClusterHead
from hfl_sipa.crypto import (
    QUANT_BITS, QUANT_SCALE,
    derive_mask_seed, prg, quantize, dequantize,
)
from hfl_sipa.protocol import begin_round, run_round, setup_plane
from hfl_sipa.satellite import Satellite
from hfl_sipa.shamir import shamir_reconstruct, shamir_split

__version__ = "0.1.0"

__all__ = [
    "ClusterHead", "Satellite",
    "setup_plane", "begin_round", "run_round",
    "shamir_split", "shamir_reconstruct",
    "quantize", "dequantize", "derive_mask_seed", "prg",
    "QUANT_BITS", "QUANT_SCALE",
]
