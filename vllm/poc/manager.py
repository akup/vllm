import time
import torch
from typing import Optional, List, Dict, Any, TYPE_CHECKING
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


class PoCManager:
    def __init__(
        self,
        model_executor: "ExecutorBase",
        model_config,
        vllm_config: "VllmConfig",
    ):
        self.model_executor = model_executor
        self.model_config = model_config
        self.vllm_config = vllm_config
        self.device = self._get_device()
        
        self.state = PoCState.IDLE
        self.config: Optional[PoCConfig] = None
        self.stats = PoCStats()
        
        self.valid_nonces: List[int] = []
        self.valid_distances: List[float] = []
        self._nonce_counter = 0
        self._generate_block_hash: Optional[str] = None
    
    def _get_device(self) -> torch.device:
        if hasattr(self.model_executor, 'driver_worker'):
            return self.model_executor.driver_worker.device
        return torch.device("cuda:0")
    
    def init_round(self, config: PoCConfig) -> None:
        if self.state == PoCState.GENERATING:
            raise RuntimeError("Round already in progress")
        
        self.config = config
        self.stats = PoCStats(start_time=time.time())
        self.valid_nonces = []
        self.valid_distances = []
        self._nonce_counter = config.node_id
        self.state = PoCState.IDLE
        
        # self._setup_layer_hooks()
    
    def _setup_layer_hooks(self) -> None:
        pass
        # from .worker_ops import poc_setup_layer_hooks
        
        # self.model_executor.collective_rpc(
        #     poc_setup_layer_hooks,
        #     args=(
        #         self.config.block_hash,
        #         self.model_config.get_hidden_size(),
        #     ),
        # )
    
    def _teardown_layer_hooks(self) -> None:
        pass
        # from .worker_ops import poc_teardown_layer_hooks
        
        # self.model_executor.collective_rpc(
        #     poc_teardown_layer_hooks,
        #     args=(),
        # )
    
    def start_generate(self) -> None:
        if self.config is None:
            raise RuntimeError("Round not initialized")
        self.state = PoCState.GENERATING
    
    def start_validate(self) -> None:
        if self.config is None:
            raise RuntimeError("Round not initialized")
        self.state = PoCState.VALIDATING
    
    def stop_round(self) -> None:
        self.state = PoCState.STOPPED
        if self.config is not None:
            self._teardown_layer_hooks()
    
    def get_next_nonces(self) -> List[int]:
        nonces = []
        for _ in range(self.config.batch_size):
            nonces.append(self._nonce_counter)
            self._nonce_counter += self.config.node_count
        return nonces
    
    def run_batch(self) -> ProofBatch:
        if self.state != PoCState.GENERATING:
            return ProofBatch.empty()
        
        from .worker_ops import poc_forward_batch
        
        nonces = self.get_next_nonces()
        
        results = self.model_executor.collective_rpc(
            poc_forward_batch,
            args=(
                self.config.block_hash,
                self.config.public_key,
                nonces,
                self.config.seq_len,
                self.model_config.get_hidden_size(),
                self.config.r_target,
                self.vllm_config,
            ),
        )
        
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
                self.model_config.get_hidden_size(),
                self.config.r_target,
                self.vllm_config,
            ),
        )
        
        result = next((r for r in results if r is not None), None)
        if result is None:
            raise RuntimeError("No result from validation")
        
        computed_dist = result["distances"]
        valid_flags = result["valid"]
        
        self.stats.total_checked += len(nonces)
        self.stats.total_valid += sum(valid_flags)
        
        fraud_detected = False
        if received_dist is not None and len(received_dist) == len(computed_dist):
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
    
    def generate_for_nonces(
        self,
        nonces: List[int],
        block_hash: str,
        public_key: str,
        r_target: float,
        seq_len: int,
        return_vectors: bool = False,
    ) -> Dict[str, Any]:
        """Generate distances for specific nonces (cached hooks for performance)."""
        from .worker_ops import poc_forward_batch, poc_setup_layer_hooks
        
        # Only setup hooks if block_hash changed (avoid overhead)
        if self._generate_block_hash != block_hash:
            self.model_executor.collective_rpc(
                poc_setup_layer_hooks,
                args=(block_hash, self.model_config.get_hidden_size()),
            )
            self._generate_block_hash = block_hash
        
        results = self.model_executor.collective_rpc(
            poc_forward_batch,
            args=(
                block_hash,
                public_key,
                nonces,
                seq_len,
                self.model_config.get_hidden_size(),
                r_target,
                self.vllm_config,
                return_vectors,
            ),
        )
        
        result = next((r for r in results if r is not None), None)
        if result is None:
            return {"nonces": nonces, "distances": []}
        
        response = {
            "nonces": result["nonces"],
            "distances": result["distances"],
        }
        if return_vectors and "vectors" in result:
            response["vectors"] = result["vectors"]
        return response
    
    def teardown_generate_hooks(self) -> None:
        """Teardown cached hooks from generate_for_nonces."""
        if self._generate_block_hash is not None:
            self._teardown_layer_hooks()
            self._generate_block_hash = None
    
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
