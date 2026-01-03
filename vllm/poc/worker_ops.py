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
    random_pick_indices,
    generate_haar_orthogonal_matrices,
)
from .layer_hooks import LayerHouseholderHook, poc_forward_context

_layer_hooks: Optional[LayerHouseholderHook] = None

POC_PICK_K_DIMS = 64


def poc_setup_layer_hooks(
    worker,
    block_hash: str,
    hidden_size: int,
) -> Dict[str, Any]:
    global _layer_hooks
    
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
    worker,
) -> Dict[str, Any]:
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


@torch.inference_mode()
def poc_forward_batch(
    worker,
    block_hash: str,
    public_key: str,
    nonces: List[int],
    seq_len: int,
    hidden_size: int,
    r_target: float,
    vllm_config,
) -> Optional[Dict[str, Any]]:
    device = worker.device
    dtype = worker.model_runner.model_config.dtype
    model = worker.model_runner.model
    batch_size = len(nonces)
    
    # Generate deterministic inputs
    inputs_embeds = generate_inputs(
        block_hash, public_key, nonces,
        dim=hidden_size, seq_len=seq_len,
        device=device, dtype=dtype,
    )
    
    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    attn_metadata = _create_prefill_attn_metadata(batch_size, seq_len, device)
    
    # PP: receive from previous rank
    intermediate_tensors = None
    if not get_pp_group().is_first_rank:
        intermediate_tensors = IntermediateTensors(
            get_pp_group().recv_tensor_dict(all_gather_group=get_tp_group())
        )
    
    # Forward pass with PoC context active (hooks will transform)
    with poc_forward_context():
        with set_forward_context(attn_metadata, vllm_config):
            hidden_states = model(
                input_ids=None,
                positions=positions.flatten(),
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds.view(-1, hidden_size),
            )
    
    # PP: send to next rank
    if not get_pp_group().is_last_rank:
        if isinstance(hidden_states, IntermediateTensors):
            get_pp_group().send_tensor_dict(
                hidden_states.tensors, all_gather_group=get_tp_group()
            )
        return None
    
    # Extract last hidden state
    hidden_states = hidden_states.view(batch_size, seq_len, -1)
    last_hidden = hidden_states[:, -1, :].float()
    
    # Normalize to unit sphere
    last_hidden = last_hidden / (last_hidden.norm(dim=-1, keepdim=True) + 1e-8)
    
    # Per-nonce k-dim pick + Haar rotation
    indices = random_pick_indices(block_hash, public_key, nonces, hidden_size, POC_PICK_K_DIMS, device)
    xk = torch.gather(last_hidden, 1, indices)
    
    Q = generate_haar_orthogonal_matrices(block_hash, public_key, nonces, POC_PICK_K_DIMS, device, dtype=xk.dtype)
    yk = torch.bmm(Q, xk.unsqueeze(-1)).squeeze(-1)
    
    # Target in k-dim space (shared across all nonces)
    target = generate_target(block_hash, public_key, POC_PICK_K_DIMS, device)
    
    # Normalize and compute distances
    yk = yk / (yk.norm(dim=-1, keepdim=True) + 1e-8)
    target = target / (target.norm() + 1e-8)
    distances = (yk - target.unsqueeze(0)).norm(dim=-1)
    
    return {
        "nonces": nonces,
        "distances": distances.cpu().tolist(),
    }


@torch.inference_mode()
def poc_validate_batch(
    worker,
    block_hash: str,
    public_key: str,
    nonces: List[int],
    seq_len: int,
    hidden_size: int,
    r_target: float,
    vllm_config,
) -> Optional[Dict[str, Any]]:
    result = poc_forward_batch(
        worker, block_hash, public_key, nonces,
        seq_len, hidden_size, r_target, vllm_config,
    )
    
    if result is None:
        return None
    
    result["valid"] = [d < r_target for d in result["distances"]]
    return result
