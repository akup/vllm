"""PoC Manager - handles artifact generation for proof of compute."""
import time
import numpy as np
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass

from .config import PoCConfig, PoCState
from .data import Artifact, encode_vector

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.executor.executor_base import ExecutorBase


@dataclass
class PoCStats:
    """Statistics for PoC generation."""
    total_processed: int = 0
    start_time: float = 0.0
    
    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time if self.start_time > 0 else 0.0
    
    @property
    def nonces_per_second(self) -> float:
        return self.total_processed / self.elapsed if self.elapsed > 0 else 0.0


class PoCManager:
    """Manages PoC artifact generation."""
    
    def __init__(
        self,
        model_executor: "ExecutorBase",
        model_config,
        vllm_config: "VllmConfig",
    ):
        self.model_executor = model_executor
        self.model_config = model_config
        self.vllm_config = vllm_config
        
        self.state = PoCState.IDLE
        self.config: Optional[PoCConfig] = None
        self.stats = PoCStats()
        self._nonce_counter = 0
    
    def init_round(self, config: PoCConfig) -> None:
        """Initialize a new generation round."""
        if self.state == PoCState.GENERATING:
            raise RuntimeError("Round already in progress")
        
        self.config = config
        self.stats = PoCStats(start_time=time.time())
        self._nonce_counter = config.node_id
        self.state = PoCState.IDLE
    
    def start_generate(self) -> None:
        """Start generating artifacts."""
        if self.config is None:
            raise RuntimeError("Round not initialized")
        self.state = PoCState.GENERATING
    
    def stop_round(self) -> None:
        """Stop the current round."""
        self.state = PoCState.STOPPED
    
    def get_next_nonces(self) -> List[int]:
        """Get next batch of nonces to process."""
        nonces = []
        for _ in range(self.config.batch_size):
            nonces.append(self._nonce_counter)
            self._nonce_counter += self.config.node_count
        return nonces
    
    def _run_forward(
        self,
        block_hash: str,
        public_key: str,
        nonces: List[int],
        seq_len: int,
        k_dim: int,
    ) -> Optional[Dict[str, Any]]:
        """Run forward pass via collective_rpc.
        
        Returns dict with 'nonces' and 'vectors' (FP16 numpy array).
        """
        from .poc_model_runner import execute_poc_forward
        
        results = self.model_executor.collective_rpc(
            execute_poc_forward,
            args=(
                block_hash,
                public_key,
                nonces,
                seq_len,
                self.model_config.get_hidden_size(),
                k_dim,
            ),
        )
        
        # Only the last PP rank returns a result
        return next((r for r in results if r is not None), None)
    
    def run_batch(self) -> Dict[str, Any]:
        """Run a batch and return artifacts.
        
        Returns:
            Dict with 'should_continue', 'nonces', 'artifacts', and metadata.
        """
        if self.state != PoCState.GENERATING:
            return {
                "should_continue": False,
                "state": self.state.value,
            }
        
        nonces = self.get_next_nonces()
        
        result = self._run_forward(
            self.config.block_hash,
            self.config.public_key,
            nonces,
            self.config.seq_len,
            self.config.k_dim,
        )
        
        if result is None:
            return {
                "should_continue": self.state == PoCState.GENERATING,
                "state": self.state.value,
                "nonces": [],
                "artifacts": [],
            }
        
        # Convert vectors to artifacts
        vectors = result["vectors"]  # FP16 numpy array [batch_size, k_dim]
        artifacts = []
        for i, nonce in enumerate(result["nonces"]):
            vector_b64 = encode_vector(vectors[i])
            artifacts.append(Artifact(nonce=nonce, vector_b64=vector_b64))
        
        self.stats.total_processed += len(nonces)
        
        return {
            "should_continue": self.state == PoCState.GENERATING,
            "state": self.state.value,
            "public_key": self.config.public_key,
            "block_hash": self.config.block_hash,
            "block_height": self.config.block_height,
            "node_id": self.config.node_id,
            "nonces": result["nonces"],
            "artifacts": artifacts,
        }
    
    def generate_artifacts(
        self,
        nonces: List[int],
        block_hash: str,
        public_key: str,
        seq_len: int,
        k_dim: int,
    ) -> List[Artifact]:
        """Generate artifacts for specific nonces.
        
        Used by /generate endpoint for explicit nonce computation.
        """
        result = self._run_forward(
            block_hash,
            public_key,
            nonces,
            seq_len,
            k_dim,
        )
        
        if result is None:
            return []
        
        vectors = result["vectors"]  # FP16 numpy array
        artifacts = []
        for i, nonce in enumerate(result["nonces"]):
            vector_b64 = encode_vector(vectors[i])
            artifacts.append(Artifact(nonce=nonce, vector_b64=vector_b64))
        
        self.stats.total_processed += len(nonces)
        return artifacts
    
    def get_status(self) -> dict:
        """Get current PoC status."""
        status = {
            "status": self.state.value,
            "stats": {
                "total_processed": self.stats.total_processed,
                "nonces_per_second": self.stats.nonces_per_second,
            },
        }
        
        if self.config:
            status["config"] = {
                "block_hash": self.config.block_hash,
                "block_height": self.config.block_height,
                "public_key": self.config.public_key,
                "node_id": self.config.node_id,
                "node_count": self.config.node_count,
                "seq_len": self.config.seq_len,
                "k_dim": self.config.k_dim,
            }
        else:
            status["config"] = None
            status["stats"] = None
        
        return status
