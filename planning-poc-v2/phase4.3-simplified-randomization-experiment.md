# Phase 4.3: Randomization Layer - Sign Flips (Final)

## Recommended Approach: Sign Flips

| Mode | Config | Description |
|------|--------|-------------|
| **Sign Flips** ✓ | `use_layer_hooks=False, use_sign_flips=True` | Per-nonce sign flips. **Recommended.** |
| Layer Hooks | `use_layer_hooks=True` | Per-layer normalization + Householder. Alternative. |

## E2E Test Results (18 tests each, 2025-12-30)

| Model | Sign Flips | Layer Hooks |
|-------|------------|-------------|
| Qwen 0.6B | **1.1% spread** ✓ | 3.9% spread |
| Llama 1B | **0.9% spread** ✓ | 10.3% spread |

**Sign Flips is clearly superior** - consistent <2% cross-block spread on both models.

## Why Sign Flips Wins

1. **Better consistency**: 0.9-1.1% spread vs 3.9-10.3%
2. **Model-agnostic**: Works equally well on Qwen and Llama
3. **Simpler**: No forward hooks, no layer modifications
4. **Faster**: No overhead from per-layer normalization

## How It Works

```python
# Per-nonce sign flips decorrelate hidden state directions
signs = generate_sign_flips(block_hash, public_key, nonces, hidden_size, device)
last_hidden = last_hidden * signs  # Random +1/-1 per dimension

# Normalize to unit sphere (breaks magnitude structure)
last_hidden = last_hidden / (last_hidden.norm(dim=-1, keepdim=True) + 1e-8)

# Householder transforms mix dimensions
for r in range(8):
    last_hidden = apply_householder(last_hidden, transform_vectors[:, r, :])
```

## Implementation

```python
# Sign flips mode (recommended)
config = PoCConfig(block_hash=..., use_layer_hooks=False, use_sign_flips=True)

# Layer hooks mode (alternative)
config = PoCConfig(block_hash=..., use_layer_hooks=True)
```
