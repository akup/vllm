import asyncio
import time
import torch
from typing import Optional, List, Tuple, TYPE_CHECKING
from dataclasses import dataclass

from .config import PoCConfig, PoCState
from .data import ProofBatch
from .gpu_random import (
    generate_inputs,
    generate_permutations,
    generate_target,
    compute_distances,
)

# Forward context imports for model execution
from vllm.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.forward_context import set_forward_context

if TYPE_CHECKING:
    from .sender import PoCCallbackSender
    from vllm.config import VllmConfig


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
    
    TODO: Update for vLLM v1 compatibility:
    - v0: model accessed via model_executor.driver_worker.model_runner.model
    - v1: model accessed via worker.get_model() or collective_rpc pattern
    
    TODO: Multistep inference with KV cache
    Current implementation uses PAD_SLOT_ID which skips KV storage.
    For proper multistep:
    1. Use CacheEngine to allocate KV cache blocks
    2. Compute slot_mapping from block_tables (slot = block_id * block_size + offset)
    3. Track sequence positions across steps (context_len grows each step)
    4. Support decode batches (num_decode_tokens > 0)
    5. Consider using model_runner.execute_model() for full vLLM parity
    """
    
    def __init__(self, model, model_config, vllm_config: "VllmConfig"):
        self.model = model
        self.model_config = model_config
        self.vllm_config = vllm_config  # Needed for set_forward_context
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        
        self.state = PoCState.IDLE
        self.config: Optional[PoCConfig] = None
        self.target: Optional[torch.Tensor] = None
        self.stats = PoCStats()
        
        self.valid_nonces: List[int] = []
        self.valid_distances: List[float] = []
        self._nonce_counter = 0
        self._generation_task: Optional[asyncio.Task] = None
        self._callback_sender: Optional['PoCCallbackSender'] = None
    
    def init_round(self, config: PoCConfig) -> None:
        """Initialize round with config and generate target. Does not start generating."""
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
        
        self.target = generate_target(
            config.block_hash,
            self.model_config.get_vocab_size(),
            self.device,
        )
        
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
    
    def _compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute logits from hidden states (model-agnostic)."""
        if hasattr(self.model, 'compute_logits'):
            return self.model.compute_logits(hidden_states, sampling_metadata=None)
        elif hasattr(self.model, 'lm_head'):
            return self.model.lm_head(hidden_states)
        else:
            raise RuntimeError("Model does not have compute_logits or lm_head")
    
    def _create_prefill_attn_metadata(
        self,
        batch_size: int,
        seq_len: int,
    ) -> FlashAttentionMetadata:
        """Create minimal attention metadata for prefill-only forward pass.
        
        Uses PAD_SLOT_ID for all slots, so KV cache writes are skipped.
        See class docstring for multistep KV cache TODO.
        """
        num_tokens = batch_size * seq_len
        seq_lens = [seq_len] * batch_size
        
        # Cumulative sequence lengths: [0, seq_len, 2*seq_len, ...]
        seq_start_loc = torch.zeros(batch_size + 1, dtype=torch.int32, device=self.device)
        seq_start_loc[1:] = torch.cumsum(
            torch.tensor(seq_lens, dtype=torch.int32, device=self.device), dim=0
        )
        
        return FlashAttentionMetadata(
            num_prefills=batch_size,
            num_prefill_tokens=num_tokens,
            num_decode_tokens=0,
            slot_mapping=torch.full((num_tokens,), PAD_SLOT_ID, dtype=torch.long, device=self.device),
            seq_lens=seq_lens,
            seq_lens_tensor=torch.tensor(seq_lens, dtype=torch.int, device=self.device),
            max_prefill_seq_len=seq_len,
            max_decode_seq_len=0,
            query_start_loc=seq_start_loc.clone(),
            seq_start_loc=seq_start_loc,
            context_lens_tensor=torch.zeros(batch_size, dtype=torch.int, device=self.device),
            block_tables=torch.empty((batch_size, 0), dtype=torch.int, device=self.device),
            use_cuda_graph=False,
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=False,
        )
    
    @torch.inference_mode()
    def run_batch(self) -> ProofBatch:
        """Run one batch of nonce computation.
        
        Uses minimal forward context with PAD_SLOT_ID to skip KV cache storage.
        TODO: For multistep, allocate real KV caches via CacheEngine.
        """
        if self.state != PoCState.GENERATING:
            return ProofBatch.empty()
        
        nonces = self.get_next_nonces()
        batch_size = len(nonces)
        hidden_size = self.model_config.get_hidden_size()
        
        inputs_embeds = generate_inputs(
            self.config.block_hash,
            self.config.public_key,
            nonces,
            dim=hidden_size,
            seq_len=self.config.seq_len,
            device=self.device,
            dtype=self.dtype,
        )
        
        positions = torch.arange(
            self.config.seq_len, 
            device=self.device
        ).unsqueeze(0).expand(batch_size, -1)
        
        # Create minimal attention metadata with PAD_SLOT_ID
        attn_metadata = self._create_prefill_attn_metadata(
            batch_size=batch_size,
            seq_len=self.config.seq_len,
        )
        
        # Run model with forward context
        with set_forward_context(attn_metadata, self.vllm_config):
            hidden_states = self.model(
                input_ids=None,
                positions=positions.flatten(),
                inputs_embeds=inputs_embeds.view(-1, hidden_size),
            )
        
        hidden_states = hidden_states.view(batch_size, self.config.seq_len, -1)
        last_hidden = hidden_states[:, -1, :]
        
        logits = self._compute_logits(last_hidden)
        
        permutations = generate_permutations(
            self.config.block_hash,
            self.config.public_key,
            nonces,
            self.model_config.get_vocab_size(),
            self.device,
        )
        
        distances = compute_distances(logits.float(), permutations, self.target)
        
        batch = ProofBatch(
            public_key=self.config.public_key,
            block_hash=self.config.block_hash,
            block_height=self.config.block_height,
            nonces=nonces,
            dist=distances.cpu().tolist(),
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
    
    @torch.inference_mode()
    def validate(self, nonces: List[int], public_key: str) -> Tuple[List[float], List[bool]]:
        """Validate nonces by recomputing distances.
        
        Uses minimal forward context with PAD_SLOT_ID (same as run_batch).
        TODO: For multistep, share KV cache with generation.
        """
        if self.config is None:
            raise RuntimeError("No round configured")
        
        batch_size = len(nonces)
        hidden_size = self.model_config.get_hidden_size()
        
        inputs_embeds = generate_inputs(
            self.config.block_hash,
            public_key,
            nonces,
            dim=hidden_size,
            seq_len=self.config.seq_len,
            device=self.device,
            dtype=self.dtype,
        )
        
        positions = torch.arange(
            self.config.seq_len,
            device=self.device
        ).unsqueeze(0).expand(batch_size, -1)
        
        # Create minimal attention metadata with PAD_SLOT_ID
        attn_metadata = self._create_prefill_attn_metadata(
            batch_size=batch_size,
            seq_len=self.config.seq_len,
        )
        
        # Run model with forward context
        with set_forward_context(attn_metadata, self.vllm_config):
            hidden_states = self.model(
                input_ids=None,
                positions=positions.flatten(),
                inputs_embeds=inputs_embeds.view(-1, hidden_size),
            )
        
        hidden_states = hidden_states.view(batch_size, self.config.seq_len, -1)
        last_hidden = hidden_states[:, -1, :]
        logits = self._compute_logits(last_hidden)
        
        permutations = generate_permutations(
            self.config.block_hash,
            public_key,
            nonces,
            self.model_config.get_vocab_size(),
            self.device,
        )
        
        distances = compute_distances(logits.float(), permutations, self.target)
        distances_list = distances.cpu().tolist()
        valid_list = [d < self.config.r_target for d in distances_list]
        
        return distances_list, valid_list
    
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

