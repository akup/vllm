from dataclasses import dataclass
from enum import Enum
from typing import Optional


class PoCState(Enum):
    IDLE = "IDLE"
    GENERATING = "GENERATING"
    VALIDATING = "VALIDATING"
    STOPPED = "STOPPED"


@dataclass
class PoCConfig:
    block_hash: str
    block_height: int
    public_key: str
    r_target: float
    fraud_threshold: float = 0.01
    node_id: int = 0
    node_count: int = 1
    batch_size: int = 32
    seq_len: int = 256
    callback_url: Optional[str] = None
    # Randomization mode: sign flips recommended (simpler, better consistency)
    use_layer_hooks: bool = False  # Per-layer normalization + Householder (alternative)
    use_sign_flips: bool = True    # Per-nonce sign flips (recommended, <2% cross-block spread)
    # Per-nonce transform on last hidden state (post-forward, last PP rank).
    # Householder reflections were the original implementation.
    use_nonce_householder: bool = True
    # Alternative: apply a random orthogonal transform matrix per nonce
    # (implemented as a seeded permutation+sign matrix for efficiency).
    use_nonce_orthogonal: bool = False
    # Optional: pick k dimensions per nonce (seeded) to operate in a k-D
    # subspace. If None, no picking is performed.
    # Used by some transform modes (e.g., nonce orthogonal transform).
    pick_k_dims: Optional[int] = None

