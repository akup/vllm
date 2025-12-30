# Phase 4.2: Distribution Analysis Report

**Date**: 2025-12-30  
**Status**: SOLVED

## Executive Summary

We discovered that PoC 2.0's trained model approach caused inconsistent distribution across block_hashes. **This has been solved** using:
1. **Per-layer normalization** - Normalizes hidden states to unit sphere at each layer
2. **Random lm_head projection** - Replaces trained lm_head with random projection (POC_OUTPUT_DIM=8192)

**Result**: Cross-block spread reduced from 49-54% to 2-3.5%.

## Experiment Results

### Test 1: Trained Weights (Current PoC 2.0)

```
block_alpha: mean=1.4131, std=0.0009, p10=1.4119
block_beta:  mean=1.4161, std=0.0009, p10=1.4149
block_gamma: mean=1.4168, std=0.0010, p10=1.4156
```

**Spread: 0.0037** (significant - causes 1% to 99% valid rate swings with single r_target)

### Test 2: Fully Random Weights (Like Original PoW)

```
block_alpha: mean=1.4142, std=0.0019, p10=1.4118
block_beta:  mean=1.4143, std=0.0018, p10=1.4120
block_gamma: mean=1.4143, std=0.0018, p10=1.4119
```

**Spread: 0.0001** (negligible - consistent across block_hashes)

### Test 3: Random embed_tokens Only

```
block_alpha: mean=1.4142, std=0.0013, p10=1.4126
block_beta:  mean=1.4133, std=0.0014, p10=1.4116
block_gamma: mean=1.4142, std=0.0014, p10=1.4125
```

**Spread: 0.0009** (better but still varies)

## Root Cause Analysis

### Original PoW vs PoC 2.0

| Aspect | Original PoW | PoC 2.0 |
|--------|-------------|---------|
| Model weights | **RANDOM** per block_hash | **TRAINED** (fixed) |
| vocab_size | 8,196 | 151,936 |
| Output structure | Random noise → uniform sphere | Structured → shifted distribution |
| block_hash effect | Changes weights → same distribution | Changes transforms → distribution shifts |

### Why Trained Weights Cause Variance

1. **Trained model has structure**: Logits aren't uniformly distributed - some tokens are more likely
2. **Householder transforms rotate but don't eliminate structure**: The underlying distribution shape persists
3. **Different block_hash → different rotation → different mean distance**

### The Tight Distribution Problem

With std ~0.001 and spread ~0.004 across block_hashes:
- r_target=1.4119 (block_alpha p10) gives ~10% valid for block_alpha but ~0% for block_gamma
- r_target=1.4160 (above all p10s) gives ~99% valid for block_alpha but ~10% for block_gamma

**No single r_target works for all block_hashes.**

## Key Code Findings

### Original PoW Weight Randomization
```python
# From /home/ubuntu/workspace/gonka/mlnode/packages/pow/src/pow/random_pool_optimized.py
def initialize_model_with_pool(model, hash_, ...):
    for name, param in model.named_parameters():
        seed_str = f"{hash_}_{name}"
        seed = int(hashlib.sha256(seed_str.encode()).hexdigest()[:16], 16)
        param.copy_(random_values)  # ALL weights replaced!
```

### Qwen Tied Embeddings
```python
# From vllm/model_executor/models/qwen3.py
if config.tie_word_embeddings:
    self.lm_head = self.model.embed_tokens  # Same tensor!
```

## Proposed Solutions

### Option 1: Randomize lm_head per block_hash (Recommended)
- Untie lm_head from embed_tokens in PoC mode
- Create random lm_head weight seeded by block_hash
- Transformer layers stay trained (real computation)
- Output projection is random → consistent distribution

**Pros**: Preserves transformer computation, consistent distribution  
**Cons**: Requires vLLM model modification

### Option 2: Full Weight Randomization (Like Original PoW)
- Randomize ALL weights per block_hash
- Known to work perfectly

**Pros**: Guaranteed consistent distribution  
**Cons**: Defeats purpose of PoC 2.0 (no "real" model)

### Option 3: Accept Variance with Wide r_target Margin
- Use r_target well above max p10 (e.g., 1.418)
- Accept 10-95% valid rate variance across block_hashes

**Pros**: No code changes needed  
**Cons**: Imprecise, validator can't verify exact rate

---

## Solution Implemented

We implemented a combination approach that achieves consistent distribution:

### 1. Per-Layer Normalization (Key Innovation)

Normalizing hidden states to unit sphere at each transformer layer breaks the structure accumulation:

```python
# In layer_hooks.py - LayerHouseholderHook._create_hook()
def normalize_and_transform(x):
    # First normalize to unit sphere (break magnitude structure)
    x_norm = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
    # Then apply Householder reflection
    return apply_householder(x_norm, v.to(x_norm.dtype))
```

**Why it works**: Orthogonal transforms (Householder, rotations) preserve structure. Normalization is a non-linear operation that removes magnitude-based "clumping" at each layer.

### 2. Random lm_head with Smaller Output Dimension

```python
# In worker_ops.py
POC_OUTPUT_DIM = 8192  # Instead of vocab_size (151,936)

random_lm_head = _generate_random_lm_head(block_hash, hidden_size, POC_OUTPUT_DIM, device)
logits = last_hidden @ random_lm_head.T  # [batch, 8192]
```

**Benefits**:
- 18x memory savings (8192 vs 151,936)
- Consistent distribution (random projection has no trained structure)
- Deterministic (seeded by block_hash)

### 3. Final Results

| Model | Spread (Before) | Spread (After) | Improvement |
|-------|-----------------|----------------|-------------|
| Qwen/Qwen3-0.6B | 54% | **3.5%** | 15x |
| unsloth/Llama-3.2-1B-Instruct | 49% | **2.0%** | 25x |

### 4. Experiment Details

**Qwen Results** (r_target calibrated to p10=1.404):
```
block_alpha: 10.0%
block_beta:  9.2%
block_gamma: 6.5%
Spread: 3.5%
```

**Llama Results** (r_target calibrated to p10=1.407):
```
block_alpha: 10.0%
block_beta:  11.9%
block_gamma: 12.0%
Spread: 2.0%
```

### 5. What Didn't Work

| Approach | Result |
|----------|--------|
| End-only normalization | No improvement (structure already baked in) |
| Random lm_head alone | 49% spread (hidden states still structured) |
| Householder on hidden states alone | 5.3% spread (better but not enough) |
| **Per-layer norm + random lm_head** | **2-3.5% spread** |

---

## Running the 18-Test Experiment

### Script Location
```
/home/ubuntu/workspace/vllm/scripts/poc_full_e2e_test.py
```

### Configuration (in script)
```python
MODELS = [
    ("Qwen/Qwen3-0.6B", "qwen"),
    ("unsloth/Llama-3.2-1B-Instruct", "llama"),
]
BLOCK_HASHES = ["block_alpha", "block_beta", "block_gamma"]
PUBLIC_KEYS = ["node_A", "node_B", "node_C"]
R_TARGET = 1.4160  # Current setting - adjust as needed
TEST_DURATION = 80  # seconds per test
```

### Run Command
```bash
cd /home/ubuntu/workspace/vllm
source .venv/bin/activate

# Full test (18 tests = ~30 minutes)
python scripts/poc_full_e2e_test.py

# Single model test
python scripts/poc_full_e2e_test.py --models qwen
python scripts/poc_full_e2e_test.py --models llama

# Shorter duration
python scripts/poc_full_e2e_test.py --duration 40
```

### Monitor Progress
```bash
# Watch results in real-time
tail -f logs/e2e_results.jsonl

# View formatted results
cat logs/e2e_results.jsonl | python3 -c "
import sys, json
for line in sys.stdin:
    r = json.loads(line)
    print(f\"Test {r['test']:2d}: {r['model']:5s} | {r['block_hash']:12s} | {r['public_key']:6s} | rate={r['valid_rate']:5.1f}%\")
"

# Check individual test logs
ls -lt logs/test_*_server.log | head -5
```

### Output Files
- `logs/e2e_results.jsonl` - JSON Lines with all test results
- `logs/test_XX_MODEL_BLOCK_KEY_server.log` - vLLM server logs
- `logs/test_XX_MODEL_BLOCK_KEY_callback.log` - Callback receiver logs
- `logs/full_e2e_test_run.log` - Main script output

## Quick Distribution Test (Without Server)

For faster iteration on distribution analysis:

```bash
cd /home/ubuntu/workspace/vllm
source .venv/bin/activate

python -c "
import os
os.environ['VLLM_USE_V1'] = '0'
import numpy as np
from vllm import LLM
from vllm.poc.config import PoCConfig
from vllm.poc.manager import PoCManager

llm = LLM(model='Qwen/Qwen3-0.6B', gpu_memory_utilization=0.3, max_model_len=256, enforce_eager=True)
manager = PoCManager(llm.llm_engine.model_executor, llm.llm_engine.model_config, llm.llm_engine.vllm_config)

for block_hash in ['block_alpha', 'block_beta', 'block_gamma']:
    config = PoCConfig(block_hash=block_hash, block_height=1, public_key='node_A', r_target=2.0, batch_size=64, seq_len=256)
    manager.init_round(config)
    manager.start_generate()
    
    distances = []
    for _ in range(30):
        batch = manager.run_batch()
        distances.extend(batch.dist)
    manager.stop_round()
    
    d = np.array(distances)
    print(f'{block_hash}: mean={np.mean(d):.4f}, std={np.std(d):.4f}, p10={np.percentile(d, 10):.4f}')
"
```

## Testing Weight Randomization

To test with fully random weights (proves the fix):

```bash
cd /home/ubuntu/workspace/vllm
source .venv/bin/activate

python -c "
import os
os.environ['VLLM_USE_V1'] = '0'
import hashlib
import numpy as np
import torch
from vllm import LLM
from vllm.poc.config import PoCConfig
from vllm.poc.manager import PoCManager

def randomize_all_weights(model, block_hash: str):
    for name, param in model.named_parameters():
        seed = int(hashlib.sha256(f'{block_hash}_{name}'.encode()).hexdigest()[:8], 16)
        torch.manual_seed(seed)
        with torch.no_grad():
            param.copy_(torch.randn_like(param) * 0.02)

llm = LLM(model='Qwen/Qwen3-0.6B', gpu_memory_utilization=0.3, max_model_len=256, enforce_eager=True)
model = llm.llm_engine.model_executor.driver_worker.model_runner.model
manager = PoCManager(llm.llm_engine.model_executor, llm.llm_engine.model_config, llm.llm_engine.vllm_config)

for block_hash in ['block_alpha', 'block_beta', 'block_gamma']:
    randomize_all_weights(model, block_hash)
    
    config = PoCConfig(block_hash=block_hash, block_height=1, public_key='node_A', r_target=2.0, batch_size=64, seq_len=256)
    manager.init_round(config)
    manager.start_generate()
    
    distances = []
    for _ in range(30):
        batch = manager.run_batch()
        distances.extend(batch.dist)
    manager.stop_round()
    
    d = np.array(distances)
    print(f'{block_hash}: mean={np.mean(d):.4f}, p10={np.percentile(d, 10):.4f}')
"
```

## Next Steps

1. ~~**Decision Required**: Choose solution approach (Option 1, 2, or 3)~~ **DONE** - Combined approach
2. ~~**If Option 1**: Implement untied random lm_head in `worker_ops.py`~~ **DONE** - Plus per-layer norm
3. ~~**Re-run 18-test experiment** to verify consistent distribution~~ **DONE** - 2-3.5% spread
4. ~~**Calibrate r_target** based on new distribution~~ **DONE** - ~1.404-1.407 for 10% valid rate

## Files Modified/Created

- `vllm/poc/worker_ops.py` - Main PoC GPU operations (added `POC_OUTPUT_DIM`, `_generate_random_lm_head`)
- `vllm/poc/gpu_random.py` - Random generation utilities  
- `vllm/poc/layer_hooks.py` - Layer-level transforms (added per-layer normalization)
- `scripts/poc_full_e2e_test.py` - 18-test experiment script
- `scripts/poc_callback_receiver.py` - Callback receiver for tests

