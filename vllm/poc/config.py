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
    # Randomization mode: either layer hooks OR sign flips (not both)
    use_layer_hooks: bool = True   # Per-layer normalization + Householder
    use_sign_flips: bool = False   # Per-nonce sign flips (alternative to layer hooks)

