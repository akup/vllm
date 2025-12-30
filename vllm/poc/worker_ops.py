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
    generate_target,
    generate_nonce_transform_vectors,
    apply_householder,
    compute_distances_direct,
    generate_sign_flips,
    _seed_from_string,
    _normal,
)
from .layer_hooks import LayerHouseholderHook

# Worker-level state for layer hooks (persists across RPC calls)
_layer_hooks: Optional[LayerHouseholderHook] = None

# Output dimension for PoC random projection (smaller = cheaper & more stable)
# Using 8192 instead of full vocab_size (151936 for Qwen) saves ~18x memory
POC_OUTPUT_DIM = 8192

# Worker-level cache for random lm_head (regenerated per block_hash)
_random_lm_head_cache: Dict[str, torch.Tensor] = {}


def _generate_random_lm_head(
    block_hash: str,
    hidden_size: int,
    output_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate deterministic random projection matrix for block_hash.
    
    This replaces the trained lm_head to ensure consistent distribution
    across all block_hashes. The random projection is seeded by block_hash
    so validators can reproduce the same distances.
    
    Args:
        block_hash: Block hash for deterministic seeding
        hidden_size: Model hidden dimension
        output_dim: Output dimension (use POC_OUTPUT_DIM, not vocab_size)
        device: Target device
        
    Returns:
        Random weight matrix of shape [output_dim, hidden_size]
    """
    global _random_lm_head_cache
    
    cache_key = f"{block_hash}_{hidden_size}_{output_dim}"
    if cache_key in _random_lm_head_cache:
        return _random_lm_head_cache[cache_key]
    
    seed = _seed_from_string(f"{block_hash}_lm_head")
    # Generate random weight matrix using deterministic RNG
    weight = _normal(seed, hidden_size * output_dim, device)
    weight = weight.view(output_dim, hidden_size)
    # Scale for numerical stability (similar to typical init)
    weight = weight * 0.02
    
    # Cache for this block_hash (clear old entries to limit memory)
    if len(_random_lm_head_cache) > 3:
        _random_lm_head_cache.clear()
    _random_lm_head_cache[cache_key] = weight
    
    return weight


def poc_setup_layer_hooks(
    worker,  # Injected by collective_rpc
    block_hash: str,
    hidden_size: int,
) -> Dict[str, Any]:
    """Setup per-round layer Householder hooks on the model.
    
    Called once at round init via collective_rpc.
    Hooks persist across all batch calls until teardown.
    
    Uses Householder reflections at each layer for orthogonal
    transformations that mix all dimensions.
    """
    global _layer_hooks
    
    # Teardown existing hooks if any
    if _layer_hooks is not None:
        _layer_hooks.detach()
    
    device = worker.device
    model = worker.model_runner.model
    
    _layer_hooks = LayerHouseholderHook(model, block_hash, device, hidden_size)
    
    return {
        "status": "ok",
        "num_layers": _layer_hooks.num_layers,
        "block_hash": block_hash,
    }


def poc_teardown_layer_hooks(
    worker,  # Injected by collective_rpc
) -> Dict[str, Any]:
    """Remove layer hooks from the model.
    
    Called at round end via collective_rpc.
    """
    global _layer_hooks
    
    num_removed = 0
    if _layer_hooks is not None:
        num_removed = _layer_hooks.num_layers
        _layer_hooks.detach()
        _layer_hooks = None
    
    return {
        "status": "ok",
        "num_removed": num_removed,
    }


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
    use_sign_flips: bool = False,
) -> Optional[Dict[str, Any]]:
    """Execute PoC forward pass following Worker.execute_model() patterns.
    
    Each worker generates identical inputs (deterministic RNG).
    PP coordination follows same pattern as Worker.execute_model().
        
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
    
    # Last PP rank: apply per-nonce transform and compute distances
    hidden_states = hidden_states.view(batch_size, seq_len, -1)
    last_hidden = hidden_states[:, -1, :].float()  # [batch, hidden_size]
    
    # Apply random sign flips if enabled (alternative to layer hooks)
    if use_sign_flips:
        signs = generate_sign_flips(block_hash, public_key, nonces, hidden_size, device)
        last_hidden = last_hidden * signs
    
    # Per-nonce Householder transform
    transform_vectors = generate_nonce_transform_vectors(
        block_hash, public_key, nonces, hidden_size, device, num_reflections=8
    )
    for r in range(transform_vectors.shape[1]):
        v = transform_vectors[:, r, :]
        last_hidden = apply_householder(last_hidden, v)
    
    # Normalize to unit sphere
    last_hidden = last_hidden / last_hidden.norm(dim=-1, keepdim=True)
    
    # Random lm_head projection for consistent distribution
    random_lm_head = _generate_random_lm_head(block_hash, hidden_size, POC_OUTPUT_DIM, device)
    logits = last_hidden @ random_lm_head.T
    
    # Compute distances
    target = generate_target(block_hash, POC_OUTPUT_DIM, device)
    distances = compute_distances_direct(logits.float(), target)
    
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
    use_sign_flips: bool = False,
) -> Optional[Dict[str, Any]]:
    """Validate nonces by recomputing distances.
    
    Returns:
        Dict with nonces, distances, and valid flags on last PP rank.
    """
    result = poc_forward_batch(
        worker, block_hash, public_key, nonces,
        seq_len, vocab_size, hidden_size, r_target, vllm_config,
        use_sign_flips
    )
    
    if result is None:
        return None
    
    # Add valid flags
    result["valid"] = [d < r_target for d in result["distances"]]
    return result

