# Phase 4.3: Randomization Layer Options

## Two Approaches

| Mode | Config | Description |
|------|--------|-------------|
| **Layer Hooks** | `use_layer_hooks=True` | Per-layer normalization + Householder. Default. |
| **Sign Flips** | `use_layer_hooks=False, use_sign_flips=True` | Per-nonce sign flips. Lightweight alternative. |

## E2E Test Results (18 tests each)

| Model | Layer Hooks | Sign Flips |
|-------|-------------|------------|
| Qwen 0.6B | 3.1% spread | 1.1% spread |
| Llama 1B | 2.3% spread | 0.9% spread |

Both approaches achieve < 5% cross-block spread (success criteria).

## When to Use

**Layer Hooks** (default):
- Proven to work on all tested models
- Breaks clustering during forward pass
- ~1% performance overhead

**Sign Flips**:
- Simpler, no model hooks
- Works by decorrelating final hidden states
- Slightly better distribution in tests

## Implementation

```python
# Layer hooks mode (default)
config = PoCConfig(block_hash=..., use_layer_hooks=True)

# Sign flips mode
config = PoCConfig(block_hash=..., use_layer_hooks=False, use_sign_flips=True)
```
