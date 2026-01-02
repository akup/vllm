# PoC Testing Guide

## Running Tests

### Unit Tests
```bash
cd /home/ubuntu/workspace/vllm
source .venv/bin/activate
python -m pytest tests/poc/ -v
```

### Smoke Test (quick validation with real model)
```bash
python scripts/poc_smoke_test.py
```

### Comprehensive E2E Test
```bash
# Run with all models (qwen, llama, qwen4b)
python scripts/poc_e2e_test.py

# Run with specific models
python scripts/poc_e2e_test.py --models qwen
python scripts/poc_e2e_test.py --models qwen llama

# Adjust duration per seed (default: 15s)
python scripts/poc_e2e_test.py --duration 100

# Custom run name
python scripts/poc_e2e_test.py --run-name my_test
```

### Test Matrix

Each model tests **9 seed combinations** (3x3 matrix) without server restart:

| Block Hashes | Public Keys |
|--------------|-------------|
| block_alpha  | node_A      |
| block_beta   | node_B      |
| block_gamma  | node_C      |

### Output Structure

```
logs/e2e_YYYYMMDD_HHMMSS/
├── run_config.json
├── test_results.json
├── qwen/
│   ├── server.log
│   ├── block_alpha_node_A.json   # Contains valid_nonces array
│   ├── block_alpha_node_B.json
│   └── ... (9 seed files + repeat)
└── llama/
    └── ...
```

Each seed JSON contains:
```json
{
    "block_hash": "block_alpha",
    "public_key": "node_A",
    "total_checked": 12960,
    "total_valid": 2700,
    "valid_rate_percent": 20.8,
    "valid_nonces": [123, 456, 789, ...],
    "valid_distances": [1.23, 1.31, ...],
    "elapsed_seconds": 100.5
}
```

## E2E Test Phases

1. **Generation Phase**: Run 9 seed combinations, save nonces per seed
2. **Determinism Test**: Repeat first seed, verify nonces match
3. **Independence Check**: Verify different seeds produce different nonces
4. **Fraud Detection**: Test wrong block_hash/public_key triggers fraud

## Available Models

| Key | Model | max_model_len | gpu_util |
|-----|-------|---------------|----------|
| qwen | Qwen/Qwen3-0.6B | 512 | 0.4 |
| llama | unsloth/Llama-3.2-1B-Instruct | 512 | 0.4 |
| qwen4b | Qwen/Qwen3-4B-Instruct-2507 | 10256 | 0.9 |

## r_target Calibration

Use `scripts/estimate_valid_rate.py` to estimate r_target for desired valid rate:

```bash
# Estimate for 20% valid rate in 64D
python scripts/estimate_valid_rate.py --dim 64 --target-rate 20
# Output: r_target for 20.0% valid rate: 1.337635
```

Current default: `r_target = 1.34` (~20% valid rate in k=64)

## Troubleshooting

### "No available memory for the cache blocks"
Model too large for GPU. Options:
- Increase `--gpu-memory-utilization`
- Use smaller `--max-model-len`
- Use smaller model

### Validation after restart fails
Test with `--enforce-eager` flag to disable CUDA graphs for debugging.

### Server startup timeout
Increase `SERVER_STARTUP_TIMEOUT` in script or check server logs for errors.
