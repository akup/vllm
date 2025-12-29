import time
import torch
from typing import Optional, List, Tuple, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass

from .config import PoCConfig, PoCState
from .data import ProofBatch

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.executor.executor_base import ExecutorBase


@dataclass
class PoCStats:
    total_checked: int = 0
    total_valid: int = 0
    start_time: float = 0.0
    
    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time if self.start_time > 0 else 0.0
    
    @property
    def rate(self) -> float:
        return self.total_checked / self.elapsed if self.elapsed > 0 else 0.0
    
    def report(self) -> str:
        """Generate progress report matching original PoW format."""
        success_rate = self.total_checked / self.total_valid if self.total_valid > 0 else 0
        elapsed_min = self.elapsed / 60
        valid_rate = self.total_valid / elapsed_min if elapsed_min > 0 else 0
        raw_rate = self.total_checked / elapsed_min if elapsed_min > 0 else 0
        return (f"Generated: {self.total_valid} / {self.total_checked} "
                f"(1 in {success_rate:.0f}) Time: {elapsed_min:.2f}min "
                f"({valid_rate:.2f} valid/min, {raw_rate:.2f} raw/min)")


class PoCManager:
    """Manager for PoC (Proof of Computation) rounds.
    
    Uses model_executor.collective_rpc() to execute PoC batches on workers,
    ensuring proper tensor parallel (TP) and pipeline parallel (PP) support.
    """
    
    def __init__(
        self,
        model_executor: "ExecutorBase",
        model_config,
        vllm_config: "VllmConfig",
    ):
        self.model_executor = model_executor
        self.model_config = model_config
        self.vllm_config = vllm_config
        
        # Get device from driver worker
        self.device = self._get_device()
        
        self.state = PoCState.IDLE
        self.config: Optional[PoCConfig] = None
        self.stats = PoCStats()
        
        self.valid_nonces: List[int] = []
        self.valid_distances: List[float] = []
        self._nonce_counter = 0
    
    def _get_device(self) -> torch.device:
        """Get device from driver worker."""
        if hasattr(self.model_executor, 'driver_worker'):
            return self.model_executor.driver_worker.device
        return torch.device("cuda:0")
    
    def init_round(self, config: PoCConfig) -> None:
        """Initialize round with config. Does not start generating."""
        if self.state == PoCState.GENERATING:
            raise RuntimeError("Round already in progress")
        
        self.config = config
        self.stats = PoCStats(start_time=time.time())
        self.valid_nonces = []
        self.valid_distances = []
        self._nonce_counter = config.node_id
        self.state = PoCState.IDLE
    
    def start_generate(self) -> None:
        """Switch to GENERATING state. Call after init_round()."""
        if self.config is None:
            raise RuntimeError("Round not initialized")
        self.state = PoCState.GENERATING
    
    def start_validate(self) -> None:
        """Switch to VALIDATING state. Call after init_round()."""
        if self.config is None:
            raise RuntimeError("Round not initialized")
        self.state = PoCState.VALIDATING
    
    def stop_round(self) -> None:
        """Stop current round."""
        self.state = PoCState.STOPPED
    
    def get_next_nonces(self) -> List[int]:
        """Get next batch of nonces for this node."""
        nonces = []
        for _ in range(self.config.batch_size):
            nonces.append(self._nonce_counter)
            self._nonce_counter += self.config.node_count
        return nonces
    
    def run_batch(self) -> ProofBatch:
        """Run one batch of nonce computation using collective_rpc.
        
        Executes PoC forward pass on all workers via collective_rpc,
        ensuring proper TP/PP coordination.
        """
        if self.state != PoCState.GENERATING:
            return ProofBatch.empty()
        
        from .worker_ops import poc_forward_batch
        
        nonces = self.get_next_nonces()
        
        # Execute via model_executor - handles TP/PP
        results = self.model_executor.collective_rpc(
            poc_forward_batch,
            args=(
                self.config.block_hash,
                self.config.public_key,
                nonces,
                self.config.seq_len,
                self.model_config.get_vocab_size(),
                self.model_config.get_hidden_size(),
                self.config.r_target,
                self.vllm_config,
            ),
        )
        
        # Find result from last PP rank (non-None)
        result = next((r for r in results if r is not None), None)
        if result is None:
            return ProofBatch.empty()
        
        batch = ProofBatch(
            public_key=self.config.public_key,
            block_hash=self.config.block_hash,
            block_height=self.config.block_height,
            nonces=result["nonces"],
            dist=result["distances"],
            node_id=self.config.node_id,
        )
        
        self.stats.total_checked += len(nonces)
        
        valid_batch = batch.sub_batch(self.config.r_target)
        self.stats.total_valid += len(valid_batch)
        self.valid_nonces.extend(valid_batch.nonces)
        self.valid_distances.extend(valid_batch.dist)
        
        return batch
    
    def run_batch_with_state(self) -> dict:
        """Run batch and return result with state for continuation check.
        
        Returns both batch results and current state, enabling single RPC
        per iteration in the generation loop (no separate status check needed).
        """
        if self.state != PoCState.GENERATING:
            return {
                "should_continue": False,
                "state": self.state.value,
            }
        
        batch = self.run_batch()
        valid_batch = batch.sub_batch(self.config.r_target)
        
        return {
            "should_continue": self.state == PoCState.GENERATING,
            "state": self.state.value,
            "public_key": self.config.public_key,
            "block_hash": self.config.block_hash,
            "block_height": self.config.block_height,
            "node_id": self.config.node_id,
            "nonces": batch.nonces,
            "distances": batch.dist,
            "valid_nonces": valid_batch.nonces,
            "valid_distances": valid_batch.dist,
        }
    
    def validate(self, nonces: List[int], public_key: str, 
                 received_dist: Optional[List[float]] = None) -> Dict[str, Any]:
        """Validate nonces by recomputing distances using collective_rpc.
        
        Returns dict with validation results and fraud detection.
        """
        if self.config is None:
            raise RuntimeError("No round configured")
        
        from .worker_ops import poc_validate_batch
        
        results = self.model_executor.collective_rpc(
            poc_validate_batch,
            args=(
                self.config.block_hash,
                public_key,
                nonces,
                self.config.seq_len,
                self.model_config.get_vocab_size(),
                self.model_config.get_hidden_size(),
                self.config.r_target,
                self.vllm_config,
            ),
        )
        
        # Find result from last PP rank (non-None)
        result = next((r for r in results if r is not None), None)
        if result is None:
            raise RuntimeError("No result from validation")
        
        computed_dist = result["distances"]
        valid_flags = result["valid"]
        
        # Update validation stats
        self.stats.total_checked += len(nonces)
        self.stats.total_valid += sum(valid_flags)
        
        # Check for fraud if received distances provided
        fraud_detected = False
        if received_dist is not None and len(received_dist) == len(computed_dist):
            # Fraud = claimed valid (dist < r_target) but computed invalid (dist >= r_target)
            for recv, comp in zip(received_dist, computed_dist):
                if recv < self.config.r_target and comp >= self.config.r_target:
                    fraud_detected = True
                    break
        
        return {
            "nonces": nonces,
            "computed_distances": computed_dist,
            "received_distances": received_dist or [],
            "valid": valid_flags,
            "fraud_detected": fraud_detected,
            "public_key": public_key,
            "block_hash": self.config.block_hash,
            "block_height": self.config.block_height,
        }
    
    def get_status(self) -> dict:
        return {
            "state": self.state.value,
            "valid_nonces": self.valid_nonces,
            "valid_distances": self.valid_distances,
            "total_checked": self.stats.total_checked,
            "total_valid": self.stats.total_valid,
            "elapsed_seconds": self.stats.elapsed,
            "rate_per_second": self.stats.rate,
            "r_target": self.config.r_target if self.config else None,
        }
