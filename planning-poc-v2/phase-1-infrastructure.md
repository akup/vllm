# Phase 1: Infrastructure

## Objective

Create the `vllm/poc/` module structure with configuration and data schemas.

## Deliverable

Working module structure that can be imported:
```python
from vllm.poc import PoCConfig, PoCState, ProofBatch
```

## Files to Create

### 1. `vllm/poc/__init__.py`

```python
from .config import PoCConfig, PoCState
from .data import ProofBatch, ValidatedBatch

__all__ = ["PoCConfig", "PoCState", "ProofBatch", "ValidatedBatch"]
```

### 2. `vllm/poc/config.py`

```python
from enum import Enum
from dataclasses import dataclass
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
    callback_url: Optional[str] = None  # Push valid batches to this URL if set
```

### 3. `vllm/poc/data.py`

Copy from original with minimal modifications:
- Source: `/home/ubuntu/workspace/gonka/mlnode/packages/pow/src/pow/data.py`
- Keep: `ProofBatch`, `ValidatedBatch`
- Remove: `InValidation` (not needed for v1)
- Keep: scipy dependency (needed for fraud detection statistical test)

```python
from dataclasses import dataclass, field
from typing import List
from scipy.stats import binomtest

PROBABILITY_MISMATCH = 5e-4

@dataclass
class ProofBatch:
    public_key: str
    block_hash: str
    block_height: int
    nonces: List[int]
    dist: List[float]
    node_id: int

    def sub_batch(self, r_target: float) -> 'ProofBatch':
        """Filter nonces where distance < r_target"""
        mask = [d < r_target for d in self.dist]
        return ProofBatch(
            public_key=self.public_key,
            block_hash=self.block_hash,
            block_height=self.block_height,
            nonces=[n for n, m in zip(self.nonces, mask) if m],
            dist=[d for d, m in zip(self.dist, mask) if m],
            node_id=self.node_id,
        )

    def __len__(self) -> int:
        return len(self.nonces)

    @staticmethod
    def empty() -> 'ProofBatch':
        return ProofBatch(
            public_key="",
            block_hash="",
            block_height=-1,
            nonces=[],
            dist=[],
            node_id=-1,
        )

@dataclass
class ValidatedBatch:
    """Matches original pow/data.py ValidatedBatch schema."""
    public_key: str
    block_hash: str
    block_height: int
    nonces: List[int]
    received_dist: List[float]
    dist: List[float]  # computed distances (matches original field name)
    r_target: float
    fraud_threshold: float
    node_id: int
    n_invalid: int = field(default=-1)
    probability_honest: float = field(default=-1.0)
    fraud_detected: bool = field(default=False)

    def __post_init__(self):
        if self.n_invalid >= 0:
            return
        self.n_invalid = sum(
            1 for rd, cd in zip(self.received_dist, self.dist)
            if rd >= self.r_target or cd > self.r_target
        )
        if len(self.nonces) > 0:
            self.probability_honest = float(
                binomtest(
                    k=self.n_invalid,
                    n=len(self.nonces),
                    p=PROBABILITY_MISMATCH,
                    alternative='greater'
                ).pvalue
            )
            self.fraud_detected = self.probability_honest < self.fraud_threshold
```

## Directory Structure After Phase 1

```
vllm/poc/
├── __init__.py
├── config.py
└── data.py

tests/poc/
├── __init__.py
└── test_data.py
```

## Unit Tests

### File: `tests/poc/test_data.py`

**Cross-check**: Compare `ProofBatch.sub_batch()` and `ValidatedBatch.__post_init__()` logic with original:
`/home/ubuntu/workspace/gonka/mlnode/packages/pow/src/pow/data.py`

```python
import pytest
from vllm.poc import PoCConfig, PoCState, ProofBatch, ValidatedBatch

class TestPoCConfig:
    def test_defaults(self):
        config = PoCConfig(
            block_hash="hash1",
            block_height=100,
            public_key="node1",
            r_target=0.5,
        )
        assert config.batch_size == 32
        assert config.seq_len == 256
        assert config.fraud_threshold == 0.01
        assert config.node_id == 0
        assert config.node_count == 1
        assert config.callback_url is None  # Optional, defaults to None
    
    def test_all_fields(self):
        config = PoCConfig(
            block_hash="hash1",
            block_height=100,
            public_key="node1",
            r_target=0.5,
            fraud_threshold=0.05,
            node_id=2,
            node_count=4,
            batch_size=64,
            seq_len=128,
            callback_url="http://localhost:8080/callback",
        )
        assert config.node_id == 2
        assert config.node_count == 4
        assert config.callback_url == "http://localhost:8080/callback"

class TestPoCState:
    def test_states_exist(self):
        assert PoCState.IDLE.value == "IDLE"
        assert PoCState.GENERATING.value == "GENERATING"
        assert PoCState.VALIDATING.value == "VALIDATING"
        assert PoCState.STOPPED.value == "STOPPED"

class TestProofBatch:
    def test_empty(self):
        batch = ProofBatch.empty()
        assert len(batch) == 0
        assert batch.nonces == []
        assert batch.dist == []
    
    def test_len(self):
        batch = ProofBatch(
            public_key="node1",
            block_hash="hash1",
            block_height=100,
            nonces=[1, 2, 3],
            dist=[0.1, 0.2, 0.3],
            node_id=0,
        )
        assert len(batch) == 3
    
    def test_sub_batch_filters_correctly(self):
        """Cross-check with original: sub_batch filtering logic"""
        batch = ProofBatch(
            public_key="node1",
            block_hash="hash1",
            block_height=100,
            nonces=[1, 2, 3, 4],
            dist=[0.3, 0.6, 0.4, 0.8],
            node_id=0,
        )
        filtered = batch.sub_batch(r_target=0.5)
        assert filtered.nonces == [1, 3]
        assert filtered.dist == [0.3, 0.4]
        assert filtered.public_key == "node1"
        assert filtered.block_hash == "hash1"
    
    def test_sub_batch_empty_result(self):
        batch = ProofBatch(
            public_key="node1",
            block_hash="hash1",
            block_height=100,
            nonces=[1, 2],
            dist=[0.6, 0.8],
            node_id=0,
        )
        filtered = batch.sub_batch(r_target=0.5)
        assert len(filtered) == 0
    
    def test_sub_batch_all_pass(self):
        batch = ProofBatch(
            public_key="node1",
            block_hash="hash1",
            block_height=100,
            nonces=[1, 2, 3],
            dist=[0.1, 0.2, 0.3],
            node_id=0,
        )
        filtered = batch.sub_batch(r_target=0.5)
        assert len(filtered) == 3

class TestValidatedBatch:
    def test_fraud_detection_all_valid(self):
        """Cross-check with original: fraud detection logic"""
        batch = ValidatedBatch(
            public_key="node1",
            block_hash="hash1",
            block_height=100,
            nonces=[1, 2, 3],
            received_dist=[0.3, 0.4, 0.45],
            dist=[0.3, 0.4, 0.45],  # computed distances
            r_target=0.5,
            fraud_threshold=0.01,
            node_id=0,
        )
        assert batch.n_invalid == 0
        assert batch.fraud_detected == False
    
    def test_fraud_detection_received_invalid(self):
        """Received dist >= r_target counts as invalid"""
        batch = ValidatedBatch(
            public_key="node1",
            block_hash="hash1",
            block_height=100,
            nonces=[1, 2, 3],
            received_dist=[0.3, 0.6, 0.45],  # 0.6 >= r_target
            dist=[0.3, 0.4, 0.45],
            r_target=0.5,
            fraud_threshold=0.01,
            node_id=0,
        )
        assert batch.n_invalid == 1
    
    def test_fraud_detection_computed_invalid(self):
        """Computed dist > r_target counts as invalid"""
        batch = ValidatedBatch(
            public_key="node1",
            block_hash="hash1",
            block_height=100,
            nonces=[1, 2, 3],
            received_dist=[0.3, 0.4, 0.45],
            dist=[0.3, 0.4, 0.55],  # 0.55 > r_target
            r_target=0.5,
            fraud_threshold=0.01,
            node_id=0,
        )
        assert batch.n_invalid == 1
    
    def test_empty_batch_no_fraud(self):
        batch = ValidatedBatch(
            public_key="node1",
            block_hash="hash1",
            block_height=100,
            nonces=[],
            received_dist=[],
            dist=[],
            r_target=0.5,
            fraud_threshold=0.01,
            node_id=0,
        )
        assert batch.n_invalid == 0
        assert batch.fraud_detected == False
```

## Running Tests

```bash
pytest tests/poc/test_data.py -v
```

## Acceptance Criteria

- [ ] `vllm/poc/` directory exists
- [ ] All imports work without errors
- [ ] `PoCConfig` has all required fields including `fraud_threshold`
- [ ] `ProofBatch.sub_batch()` correctly filters by r_target
- [ ] `ValidatedBatch` computes fraud detection statistics
- [ ] scipy dependency available (for binomtest)
- [ ] All unit tests pass: `pytest tests/poc/test_data.py`

