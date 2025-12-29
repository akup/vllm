import asyncio
import time
import torch
from typing import Optional, List, Tuple, TYPE_CHECKING
from dataclasses import dataclass

from .config import PoCConfig, PoCState
from .data import ProofBatch
from .gpu_random import generate_target

if TYPE_CHECKING:
    from .sender import PoCCallbackSender
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
        self._generation_task: Optional[asyncio.Task] = None
        self._callback_sender: Optional['PoCCallbackSender'] = None
    
    def _get_device(self) -> torch.device:
        """Get device from driver worker."""
        if hasattr(self.model_executor, 'driver_worker'):
            return self.model_executor.driver_worker.device
        return torch.device("cuda:0")
    
    def init_round(self, config: PoCConfig) -> None:
        """Initialize round with config. Does not start generating."""
        if self.state == PoCState.GENERATING:
            raise RuntimeError("Round already in progress")
        
        if self._generation_task and not self._generation_task.done():
            self._generation_task.cancel()
            self._generation_task = None
        
        self.config = config
        self.stats = PoCStats(start_time=time.time())
        self.valid_nonces = []
        self.valid_distances = []
        self._nonce_counter = config.node_id
        
        if config.callback_url:
            from .sender import PoCCallbackSender
            self._callback_sender = PoCCallbackSender(
                callback_url=config.callback_url,
                r_target=config.r_target,
                fraud_threshold=config.fraud_threshold,
            )
        else:
            self._callback_sender = None
        
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
        if self._generation_task and not self._generation_task.done():
            self._generation_task.cancel()
            self._generation_task = None
        self.state = PoCState.VALIDATING
    
    def stop_round(self) -> None:
        """Stop current round and cancel any running tasks."""
        if self._generation_task and not self._generation_task.done():
            self._generation_task.cancel()
            self._generation_task = None
        self.state = PoCState.STOPPED
    
    def set_generation_task(self, task: asyncio.Task) -> None:
        """Track the background generation task for cleanup."""
        self._generation_task = task
    
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
    
    async def run_batch_async(self) -> ProofBatch:
        """Async version of run_batch. Sends to callback URL if configured."""
        batch = self.run_batch()
        if self._callback_sender and len(batch) > 0:
            await self._callback_sender.send_generated(batch)
        return batch
    
    def validate(self, nonces: List[int], public_key: str) -> Tuple[List[float], List[bool]]:
        """Validate nonces by recomputing distances using collective_rpc."""
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
        
        return result["distances"], result["valid"]
    
    def get_status(self) -> dict:
        return {
            "state": self.state.value,
            "valid_nonces": self.valid_nonces,
            "valid_distances": self.valid_distances,
            "total_checked": self.stats.total_checked,
            "total_valid": self.stats.total_valid,
            "elapsed_seconds": self.stats.elapsed,
            "rate_per_second": self.stats.rate,
        }
