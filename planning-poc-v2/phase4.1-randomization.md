# Phase 4.1: Randomization Layer Experiments

**Date**: 2025-12-29  
**Status**: Implemented

## Motivation

The base PoC implementation uses nonce-dependent **permutation** on logits to create randomized distance calculations. However, experiments revealed that trained models have **structured output distributions** that compress distances compared to theoretical random distributions:

| Model | Observed Mean | Theoretical Mean | Compression |
|-------|---------------|------------------|-------------|
| Qwen 0.6B | 1.2155 | 1.4142 | -14% |
| Llama 1B | 1.4478 | 1.4142 | +2% |

**Problem**: Permutation only reorders elements - it doesn't mix magnitudes. If trained models concentrate probability mass on certain tokens, permutation preserves that pattern.

**Goal**: Find a lightweight transformation that breaks the model's structure while maintaining:
1. Minimal VRAM overhead
2. Negligible performance impact
3. TP/PP compatibility (deterministic, locally generated)

---

## Experiments Conducted

### 1. Householder Reflections on Logits

**Hypothesis**: True orthogonal transformations (Householder) should break structure.

**Implementation**: Replace permutation with 8 Householder reflections:
```python
# H @ x = x - 2*(v·x)*v  for each reflection vector v
for r in range(8):
    v = random_unit_vector(seed + r)
    logits = logits - 2 * (logits @ v) * v
```

**Results**:
| Metric | Permutation | Householder (8) | Theoretical |
|--------|-------------|-----------------|-------------|
| Qwen Mean | 1.2155 | **1.4149** | 1.4142 |
| Qwen p10 | 1.1739 | **1.4116** | 1.4119 |

**Verdict**: ✅ Structure completely broken! But memory overhead is too high:
- Per nonce: 4.86 MB (8 × vocab_size × float32)
- Batch=1000: **4.86 GB** ❌

### 2. Between-Layer Random Signs (Hooks)

**Hypothesis**: Apply random ±1 signs between transformer layers during forward pass.

**Implementation**: PyTorch forward hooks on each decoder layer:
```python
for layer in model.layers:
    layer.register_forward_hook(apply_random_signs)
```

**Results**:
| Metric | Baseline | Layer Signs | Combined* |
|--------|----------|-------------|-----------|
| Qwen Mean | 1.2155 | 1.3052 | **1.4405** |
| Overhead | 0% | ~1% | **<1%** |

*Combined = layer signs + permutation on logits

**Verdict**: ✅ Partially breaks structure with minimal overhead.

**Limitation**: Hooks apply same signs to all nonces in batch (block-level, not nonce-level).

### 3. Random Signs on Logits

**Hypothesis**: Element-wise sign flip might break structure.

**Results**: No effect - signs don't change relative magnitudes after normalization.

### 4. Hidden State Transforms (Before lm_head)

**Hypothesis**: Transform hidden states (small dimension) instead of logits (large dimension).

**Challenge**: In vLLM's distributed architecture, `collective_rpc` runs forward pass in worker process. Patches on main process model don't affect worker.

**Memory Comparison**:
| Transform | Dimension | Per Nonce | Batch=1000 |
|-----------|-----------|-----------|------------|
| Perm (logits) | 151,936 | 1.2 MB | 1.2 GB |
| Householder (hidden) | 1,024 | 4 KB | **4 MB** |
| Perm+signs (hidden) | 1,024 | 9 KB | **9 MB** |

**Verdict**: Would be 150x cheaper, but requires modifying `worker_ops.py` directly.

### 5. Random Model Comparison

**Hypothesis**: Trained models have structure that compresses distances.

**Implementation**: Randomly initialize Qwen's weights and run PoC.

**Results**:
| Model | Mean Distance |
|-------|---------------|
| Trained Qwen | 1.2155 |
| Random Qwen | **1.4142** (theoretical!) |

**Verdict**: ✅ Confirms trained models have exploitable structure.

---

## Summary of Results

| Approach | Breaks Structure? | Memory/nonce | Overhead | Per-Nonce? | Cross-Block Spread |
|----------|------------------|--------------|----------|------------|-------------------|
| Permutation only | ❌ No | 1.2 MB | 0% | ✅ Yes | 54% |
| Householder (8 refl) | ✅ Yes | 4.86 MB | ~0% | ✅ Yes | ~5% |
| Layer signs (hooks) | Partial | 112 KB* | ~1% | ❌ No | ~40% |
| Combined (layer+perm) | ✅ Yes | 1.3 MB | <1% | Partial | ~30% |
| Hidden Householder only | ✅ Partial | 32 KB | ~0% | ✅ Yes | 5.3% |
| **Per-layer norm + random lm_head** | **✅ Yes** | **34 MB** | **<1%** | **✅ Yes** | **2-3.5%** |

*Per-round, not per-nonce

---

## Recommended Approach (IMPLEMENTED)

### Per-Layer Normalization + Random lm_head (POC_OUTPUT_DIM=8192)

```python
# At round init - layer hooks with normalization (layer_hooks.py)
class LayerHouseholderHook:
    def normalize_and_transform(x):
        x_norm = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
        return apply_householder(x_norm, v)

# At batch execution - random lm_head (worker_ops.py)
POC_OUTPUT_DIM = 8192
random_lm_head = _generate_random_lm_head(block_hash, hidden_size, POC_OUTPUT_DIM, device)
logits = last_hidden @ random_lm_head.T
distances = compute_distances_direct(logits.float(), target)
```

**Benefits**:
- Cross-block spread: 2-3.5% (was 49-54%)
- Overhead: <1%
- Memory: 34 MB for lm_head (18x smaller than vocab_size)
- Model-agnostic r_target: ~1.404-1.407 for 10% valid rate

### Why This Works

The key insight is that **orthogonal transforms preserve structure**. Householder reflections, rotations, and permutations are all orthogonal - they don't change the "shape" of the distribution.

**Normalization is non-linear** - it projects all vectors to the unit sphere, removing magnitude-based structure. When applied at each layer, it prevents structure accumulation through the transformer.

---

## How to Reproduce

### 1. Householder on Logits

The modified `gpu_random.py` for Householder was:
```python
def generate_householder_vectors(block_hash, public_key, nonces, vocab_size, device, num_reflections=8):
    # Generate unit vectors for each nonce and reflection
    ...

def compute_distances(logits, householder_vectors, target):
    for r in range(num_reflections):
        v = householder_vectors[:, r, :]
        dot = (logits * v).sum(dim=1, keepdim=True)
        logits = logits - 2 * dot * v
    ...
```

### 2. Combined Signs Experiment

```bash
cd /home/ubuntu/workspace/vllm
python scripts/poc_combined_signs_experiment.py
```

This runs all 4 configurations (baseline, logit signs, layer signs, combined) on both Qwen and Llama.

### 3. Random Model Experiment

```bash
python scripts/poc_random_qwen_experiment.py
```

Loads Qwen, randomizes weights, runs PoC to verify random model matches theoretical distribution.

---

## Files

| File | Purpose | Keep? |
|------|---------|-------|
| `scripts/poc_combined_signs_experiment.py` | Main experiment comparing all approaches | ✅ |
| `scripts/poc_random_qwen_experiment.py` | Random vs trained model comparison | ✅ |
| `scripts/poc_distribution_experiment.py` | r_target calibration | ✅ |
| `logs/householder_experiment_report.md` | Detailed results | ✅ |
| `logs/experiment_report.md` | r_target calibration results | ✅ |

---

## Final Implementation

Based on experiments, the **hidden state Householder transform** was implemented in `worker_ops.py`.

### Critical Bug Fix: `_murmur3_32`

During implementation, a critical bug was discovered in the random number generation:

- **Bug**: `_murmur3_32` was returning only positive `int32` values
- **Effect**: `_uniform()` produced values only > 0.5, so all random signs were +1
- **Fix**: Use `torch.int64` for intermediate calculations to preserve full uint32 range

```python
# Before (broken): int32 overflow caused positive-only values
h = torch.full_like(keys, seed, dtype=torch.int32)

# After (fixed): int64 preserves full range
h = torch.full_like(keys, seed & 0xFFFFFFFF, dtype=torch.int64)
```

### Current Architecture

The implemented approach combines **layer hooks** and **worker_ops.py**:

**1. Per-layer normalization + Householder (in layer_hooks.py):**
```python
# Applied at each transformer layer via forward hooks
def normalize_and_transform(x):
    x_norm = x / (x.norm(dim=-1, keepdim=True) + 1e-8)  # Break structure
    return apply_householder(x_norm, v)
```

**2. Per-nonce Householder + random lm_head (in worker_ops.py):**
```python
# Per-nonce Householder transform on hidden states
transform_vectors = generate_nonce_transform_vectors(
    block_hash, public_key, nonces, hidden_size, device, num_reflections=8
)
for r in range(transform_vectors.shape[1]):
    v = transform_vectors[:, r, :]  # [batch, hidden]
    last_hidden = apply_householder(last_hidden, v)

# Normalize before projection
last_hidden = last_hidden / last_hidden.norm(dim=-1, keepdim=True)

# Random lm_head projection (POC_OUTPUT_DIM=8192, not vocab_size)
random_lm_head = _generate_random_lm_head(block_hash, hidden_size, POC_OUTPUT_DIM, device)
logits = last_hidden @ random_lm_head.T  # [batch, 8192]

# Distance computation
target = generate_target(block_hash, POC_OUTPUT_DIM, device)
distances = compute_distances_direct(logits.float(), target)
```

**Key points:**
- Per-layer normalization breaks structure at each layer
- 8 Householder reflections per nonce on hidden states (~1024 dim)
- Random lm_head with POC_OUTPUT_DIM=8192 (18x smaller than vocab_size)
- Memory: ~34 MB for lm_head (vs 622 MB with vocab_size)

### Distribution Results

After bug fix, per-nonce hidden state transform achieves near-perfect theoretical distribution:

| Metric | Before Fix | After Fix | Theoretical |
|--------|------------|-----------|-------------|
| Mean   | 1.2311     | 1.4142    | 1.4142      |
| p10    | 1.1726     | 1.4119    | 1.4119      |
| Mean deviation | -12.9% | ~0.0% | - |
| p10 deviation  | -16.9% | ~0.0% | - |

### Layer Hooks Status - WORKING

Layer hooks (`LayerHouseholderHook`) are implemented and **working correctly**. Initial concerns about vLLM's `@support_torch_compile` bypassing hooks were unfounded - the hooks are applied to individual decoder layers (`Qwen2DecoderLayer`, `LlamaDecoderLayer`) which are plain `nn.Module` classes without the decorator.

**Key insight**: The `@support_torch_compile` decorator is on `Qwen3Model`/`LlamaModel` (the container), not on individual decoder layers.

---

## DONE: Layer Hooks Working

Layer hooks now provide per-round structure breaking with **per-layer normalization**:

```python
# In layer_hooks.py - LayerHouseholderHook._create_hook()
def normalize_and_transform(x):
    # First normalize to unit sphere (break magnitude structure)
    x_norm = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
    # Then apply Householder reflection
    return apply_householder(x_norm, v.to(x_norm.dtype))
```

**Results**:
- Per-layer normalization breaks structure accumulation
- Combined with random lm_head: 2-3.5% cross-block spread (was 49-54%)
- Works on both Qwen and Llama architectures

---

## Files (Updated)

| File | Purpose | Status |
|------|---------|--------|
| `scripts/poc_smoke_test.py` | Core PoCManager verification | Keep |
| `scripts/poc_e2e_test.py` | Full E2E tests with server restart | Keep |
| `scripts/poc_callback_receiver.py` | Helper for E2E tests | Keep |
| `scripts/poc_distribution_experiment.py` | r_target calibration via API | Keep |
| `scripts/poc_distribution_analysis.py` | Quick distribution verification | Keep |
| `scripts/poc_random_qwen_experiment.py` | Proves trained model structure | Keep |

