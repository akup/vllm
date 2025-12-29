"""PoC worker operations - callables for collective_rpc.

These functions follow the same execution patterns as Worker.execute_model()
for proper tensor parallel (TP) and pipeline parallel (PP) support.
"""
import torch
from typing import List, Optional, Dict, Any

from vllm.distributed import get_pp_group, get_tp_group
from vllm.sequence import IntermediateTensors
from vllm.forward_context import set_forward_context
from vllm.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.attention.backends.utils import PAD_SLOT_ID

from .gpu_random import (
    generate_inputs,
    generate_permutations,
    generate_target,
    compute_distances,
)


def _create_prefill_attn_metadata(
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> FlashAttentionMetadata:
    """Create minimal attention metadata for prefill-only forward pass.
    
    Uses PAD_SLOT_ID for all slots, so KV cache writes are skipped.
    """
    num_tokens = batch_size * seq_len
    seq_lens = [seq_len] * batch_size
    
    seq_start_loc = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    seq_start_loc[1:] = torch.cumsum(
        torch.tensor(seq_lens, dtype=torch.int32, device=device), dim=0
    )
    
    return FlashAttentionMetadata(
        num_prefills=batch_size,
        num_prefill_tokens=num_tokens,
        num_decode_tokens=0,
        slot_mapping=torch.full((num_tokens,), PAD_SLOT_ID, dtype=torch.long, device=device),
        seq_lens=seq_lens,
        seq_lens_tensor=torch.tensor(seq_lens, dtype=torch.int, device=device),
        max_prefill_seq_len=seq_len,
        max_decode_seq_len=0,
        query_start_loc=seq_start_loc.clone(),
        seq_start_loc=seq_start_loc,
        context_lens_tensor=torch.zeros(batch_size, dtype=torch.int, device=device),
        block_tables=torch.empty((batch_size, 0), dtype=torch.int, device=device),
        use_cuda_graph=False,
        multi_modal_placeholder_index_maps=None,
        enable_kv_scales_calculation=False,
    )


def _compute_logits(model, hidden_states: torch.Tensor) -> torch.Tensor:
    """Compute logits from hidden states (model-agnostic)."""
    if hasattr(model, 'compute_logits'):
        return model.compute_logits(hidden_states, sampling_metadata=None)
    elif hasattr(model, 'lm_head'):
        return model.lm_head(hidden_states)
    else:
        raise RuntimeError("Model does not have compute_logits or lm_head")


@torch.inference_mode()
def poc_forward_batch(
    worker,  # Injected by collective_rpc
    block_hash: str,
    public_key: str,
    nonces: List[int],
    seq_len: int,
    vocab_size: int,
    hidden_size: int,
    r_target: float,
    vllm_config,
) -> Optional[Dict[str, Any]]:
    """Execute PoC forward pass following Worker.execute_model() patterns.
    
    Each worker generates identical inputs (deterministic RNG).
    PP coordination follows same pattern as Worker.execute_model().
    
    Args:
        worker: Worker instance (injected by collective_rpc)
        block_hash: Block hash for deterministic generation
        public_key: Public key for deterministic generation
        nonces: List of nonces to process
        seq_len: Sequence length
        vocab_size: Model vocabulary size
        hidden_size: Model hidden size
        r_target: Target distance threshold
        vllm_config: VllmConfig for forward context
        
    Returns:
        Dict with nonces and distances on last PP rank, None on other ranks.
    """
    device = worker.device
    dtype = worker.model_runner.model_config.dtype
    model = worker.model_runner.model
    batch_size = len(nonces)
    
    # Generate inputs locally (deterministic - all workers get same)
    inputs_embeds = generate_inputs(
        block_hash, public_key, nonces,
        dim=hidden_size, seq_len=seq_len,
        device=device, dtype=dtype,
    )
    
    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    
    # Create attention metadata
    attn_metadata = _create_prefill_attn_metadata(batch_size, seq_len, device)
    
    # === PP Pattern from Worker.execute_model() ===
    # Receive intermediate tensors from previous PP rank
    intermediate_tensors = None
    if not get_pp_group().is_first_rank:
        intermediate_tensors = IntermediateTensors(
            get_pp_group().recv_tensor_dict(all_gather_group=get_tp_group())
        )
    
    # Forward pass
    with set_forward_context(attn_metadata, vllm_config):
        hidden_states = model(
            input_ids=None,
            positions=positions.flatten(),
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds.view(-1, hidden_size),
        )
    
    # PP: send to next rank or compute final output
    if not get_pp_group().is_last_rank:
        # Send intermediate tensors to next PP rank
        if isinstance(hidden_states, IntermediateTensors):
            get_pp_group().send_tensor_dict(
                hidden_states.tensors, all_gather_group=get_tp_group()
            )
        return None  # Not last rank
    
    # Last PP rank: compute logits and distances
    hidden_states = hidden_states.view(batch_size, seq_len, -1)
    last_hidden = hidden_states[:, -1, :]
    
    logits = _compute_logits(model, last_hidden)
    
    # Generate permutations and target locally (deterministic)
    permutations = generate_permutations(
        block_hash, public_key, nonces, vocab_size, device
    )
    target = generate_target(block_hash, vocab_size, device)
    
    distances = compute_distances(logits.float(), permutations, target)
    
    return {
        "nonces": nonces,
        "distances": distances.cpu().tolist(),
    }


@torch.inference_mode()
def poc_validate_batch(
    worker,  # Injected by collective_rpc
    block_hash: str,
    public_key: str,
    nonces: List[int],
    seq_len: int,
    vocab_size: int,
    hidden_size: int,
    r_target: float,
    vllm_config,
) -> Optional[Dict[str, Any]]:
    """Validate nonces by recomputing distances.
    
    Same as poc_forward_batch but for validation context.
    Uses the provided public_key (which may differ from generator's key).
    
    Returns:
        Dict with nonces, distances, and valid flags on last PP rank.
    """
    result = poc_forward_batch(
        worker, block_hash, public_key, nonces,
        seq_len, vocab_size, hidden_size, r_target, vllm_config
    )
    
    if result is None:
        return None
    
    # Add valid flags
    result["valid"] = [d < r_target for d in result["distances"]]
    return result

