# Phase 6: Enhancements (Optional - Future)

## Status: NOT IMPLEMENTED

This phase is planned for future enhancement after core PoC 2.0 is stable and tested.

## Objectives

1. Add lightweight random projection layers for additional security
2. Enable parallel inference during VALIDATING phase

---

## 6.1 Random Projection Layers

Add lightweight random projection layers that transform inputs/outputs without modifying base model weights. This provides an additional security margin against potential attacks on input-only randomization.

## Why Defer This

1. Core PoC 2.0 works without it - random `inputs_embeds` generation per-nonce is sufficient for v1
2. Adds complexity to model forward pass
3. Requires careful design for vLLM integration
4. Current approach is simpler and easier to validate

## Motivation

From `proof-of-compute.md` Option B:

```
Inject lightweight random layers that don't modify base weights:
- Base model weights UNCHANGED - can serve inference simultaneously
- Random layers are small (~MBs) but force full model execution
- Switch between modes is instant (no weight reordering)
- vLLM integration friendly
```

## Future Implementation

### RandomProjection Module

```python
import torch
import torch.nn as nn

class RandomProjection(nn.Module):
    """Lightweight random layers for PoC 2.0 Phase 6."""
    
    def __init__(self, block_seed: int, dim: int):
        super().__init__()
        rng = torch.Generator().manual_seed(block_seed)
        
        # Small random matrices (~few MB total)
        self.pre_proj = nn.Parameter(
            torch.randn(dim, dim, generator=rng) * 0.01,
            requires_grad=False
        )
        self.post_proj = nn.Parameter(
            torch.randn(dim, dim, generator=rng) * 0.01,
            requires_grad=False
        )
    
    def pre(self, x: torch.Tensor) -> torch.Tensor:
        """Apply pre-model random transform (residual connection)"""
        return x + x @ self.pre_proj
    
    def post(self, x: torch.Tensor) -> torch.Tensor:
        """Apply post-model random transform (residual connection)"""
        return x + x @ self.post_proj
```

### Integration with PoCManager

```python
class PoCManager:
    def start_round(self, config: PoCConfig) -> None:
        # ... existing code ...
        
        # Phase 6: Create random projection for this round
        if self.use_random_projection:
            self.random_proj = RandomProjection(
                block_seed=self._seed_from_hash(config.block_hash),
                dim=self.model_config.hidden_size
            ).to(self.device)
    
    def run_batch(self) -> ProofBatch:
        # ... existing code ...
        
        # Phase 6: Apply pre-transform
        if self.random_proj:
            inputs_embeds = self.random_proj.pre(inputs_embeds)
        
        # Forward pass through model
        hidden_states = self.model(...)
        
        # Phase 6: Apply post-transform
        if self.random_proj:
            hidden_states = self.random_proj.post(hidden_states)
        
        # ... rest of processing ...
```

## When to Implement

- After Phase 1-5 are complete and tested in production
- If additional security margin is needed
- If attack vectors are identified against input-only randomization
- If chain validators require additional protection

## Security Considerations

Random projection layers provide:

| Property | Guarantee |
|----------|-----------|
| Unpredictable | Projection matrices depend on future block seed |
| Lightweight | Only ~MBs of additional memory per round |
| Non-disruptive | Base model weights remain unchanged |
| Verifiable | Validators can reconstruct same projections |

## Acceptance Criteria for 6.1 (When Implemented)

- [ ] RandomProjection module created
- [ ] Integration with PoCManager
- [ ] Determinism tests pass (same seed = same projection)
- [ ] Performance overhead < 5%
- [ ] Memory overhead documented
- [ ] Backward compatibility with Phase 1-5 (optional flag)

---

## 6.2 Parallel Inference During Validation

Currently (Phase 1-5), inference is blocked (503) during both GENERATING and VALIDATING states. This is simple but blocks production traffic during validation.

### Goal

Allow inference requests to proceed during VALIDATING phase while validation batches are being processed.

### Why This Is Safe

- VALIDATING phase only recomputes distances for submitted nonces
- No model weights are modified
- Validation can share GPU with inference (may need batching/scheduling)

### Implementation Approach

```python
# In poc_blocking_middleware (routes.py)

@router.middleware("http")
async def poc_blocking_middleware(request: Request, call_next):
    manager = getattr(request.app.state, 'poc_manager', None)
    
    # Block inference only during GENERATING (not VALIDATING)
    if manager and manager.state == PoCState.GENERATING:
        if request.url.path.startswith("/v1/"):
            raise HTTPException(status_code=503, detail="PoC in progress")
    
    return await call_next(request)
```

### Challenges

1. **GPU Contention**: Validation and inference compete for GPU
2. **Latency Impact**: Validation may slow inference and vice versa
3. **Scheduling**: May need to batch/queue validation requests

### Acceptance Criteria for 6.2 (When Implemented)

- [ ] Inference allowed during VALIDATING state
- [ ] Validation still produces correct distances
- [ ] Performance impact on inference documented
- [ ] Performance impact on validation documented
- [ ] No race conditions or deadlocks

