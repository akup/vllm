# PoC 2.0 vLLM Integration

Proof of Compute integration into vLLM for Gonka Chain nodes.

## Overview

PoC 2.0 enables nodes to prove computational capacity by running inference on a loaded model with deterministic random inputs. Unlike PoC 1.0 (random weights), PoC 2.0 uses the **real deployed model**, making it suitable for production inference nodes.

## Motivation

Production nodes already run inference using vLLM. We want:
- Seamless switch between inference and PoC modes
- No model reload overhead (same deployed model)
- Minimal changes to vLLM core
- Easy migration path from standalone PoC

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Same deployed model | Access via `model_runner.model`, no re-deployment |
| Blocking inference | Return 503 during PoC (Phase 6 investigates parallel) |
| API prefix `/api/v1/pow/*` | Migration compatibility with original service |
| GPU-native random | All generation on GPU for performance |
| Isolated module `vllm/poc/` | Easy to migrate, minimal vLLM coupling |
| Push + Pull model | Optional `callback_url` for push; GET /status for pull |

## Breaking Change: RNG Implementation

PoC 2.0 uses a new RNG implementation (Murmur3-based deterministic generation) that is NOT compatible with PoC 1.0 (numpy SeedSequence). All nodes and validators must upgrade together. Cross-validation between old and new implementations will fail.

## Reference Implementation

Original standalone PoC implementation:
```
/home/ubuntu/workspace/gonka/mlnode/packages/pow/
├── src/pow/
│   ├── data.py              # ProofBatch, ValidatedBatch schemas
│   ├── legacy_random.py     # Target vector, distance calculation
│   ├── compute/
│   │   ├── compute.py       # Core computation logic
│   │   ├── utils.py         # Phase, Stats, NonceIterator
│   │   └── worker.py        # Multi-process worker
│   └── service/
│       ├── routes.py        # API endpoints
│       └── manager.py       # PowManager, PowInitRequest
```

**Important**: Maintain same interfaces as original pow package for seamless migration.

## Architecture

```
vLLM Server (running)
    │
    ├── Normal Mode: /v1/chat/completions → Inference
    │
    └── PoC Mode: /api/v1/pow/* → Nonce search
            │
            └── Uses SAME loaded model (no restart)
```

### Model Access Path

Uses `collective_rpc` for TP/PP support:
```
engine_client → RPCPoCRequest → MQLLMEngine
    → PoCManager.run_batch()
    → model_executor.collective_rpc(poc_forward_batch)
    → Workers execute on GPU
```

## PoC Round Flow

```
1. Client → POST /api/v1/pow/init/generate {block_hash, r_target, ...}
2. Server → init_round() sets config, start_generate() sets state = GENERATING
3. Server → Block inference (503) during GENERATING and VALIDATING [Phase 6]
4. Loop (background task with tracking):
   - Generate inputs_embeds from nonces (GPU)
   - model.forward(inputs_embeds) → hidden_states
   - model.compute_logits() → logits
   - Compute distances to target vector
   - Collect nonces where distance < r_target
5. Client → GET /api/v1/pow/status → {valid_nonces, stats}
6. Client → POST /api/v1/pow/stop
7. Server → Cancel task, set state = STOPPED, resume inference
```

Note: State is stateless. If vLLM restarts during a round, the round is lost.

## Distance Calculation (Sphere Geometry)

Both output and target are normalized to unit vectors on a sphere:

```python
# Target: uniform random point on unit sphere (vocab_size dimensions)
target = randn(vocab_size)
target = target / norm(target)

# Output: normalized model logits with per-nonce permutation
logits_permuted = logits[:, permutation]
output = logits_permuted / norm(logits_permuted)

# Distance: chord length between points on unit sphere
distance = norm(output - target)  # Range: [0, 2]
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/pow/init` | POST | Initialize PoC round |
| `/api/v1/pow/init/generate` | POST | Init + start generating |
| `/api/v1/pow/init/validate` | POST | Init + start validating |
| `/api/v1/pow/phase/generate` | POST | Switch to generate mode |
| `/api/v1/pow/phase/validate` | POST | Switch to validate mode |
| `/api/v1/pow/validate` | POST | Validate submitted nonces (accepts ProofBatch) |
| `/api/v1/pow/status` | GET | Get status and valid nonces |
| `/api/v1/pow/stop` | POST | Stop current round |

## Callback URL (Push Model)

For compatibility with the original PoW service, valid batches can be pushed to an external URL:

```bash
# Start with callback - valid batches POSTed automatically
curl -X POST http://localhost:8000/api/v1/pow/init/generate \
  -H "Content-Type: application/json" \
  -d '{"block_hash": "abc", "block_height": 100, "public_key": "node1", 
       "r_target": 0.5, "node_id": 0, "node_count": 1,
       "callback_url": "http://aggregator:9000/pow"}'
```

Callback endpoints:
- `POST {callback_url}/generated` - Valid nonces as found (during GENERATING)
- `POST {callback_url}/validated` - Validation results (during VALIDATING)

Note: `callback_url` is optional. Without it, use `GET /status` to poll for results.

## Phases

| Phase | Document | Deliverable |
|-------|----------|-------------|
| 1 | [phase-1-infrastructure.md](phase-1-infrastructure.md) | `vllm/poc/` directory with config and data schemas |
| 2 | [phase-2-gpu-random.md](phase-2-gpu-random.md) | GPU-native random generation with determinism tests |
| 3 | [phase-3-manager.md](phase-3-manager.md) | PoCManager with model forward pass integration |
| 4 | [phase-4-integration.md](phase-4-integration.md) | API routes and vLLM server integration |
| 5 | [phase-5-testing.md](phase-5-testing.md) | Comprehensive test suite |
| 6 | [phase-6-optional.md](phase-6-optional.md) | Random projection layers (future) |

## CLI Usage

```bash
# Start vLLM with PoC enabled
vllm serve Qwen/Qwen3-0.6B --enable-poc --poc-batch-size 32 --poc-seq-len 256
```

## Test Scenarios

| Test | Expected |
|------|----------|
| Basic round | Valid nonces found |
| Same seed + nonce | Identical distance |
| Wrong block_hash | Distance mismatch |
| Wrong public_key | Distance mismatch |
| Fabricated nonces | distance > r_target |
| Restart + validate | Same distances |

## Distribution Note (PoC 2.0 vs 1.0)

With a real trained model (not random weights), output distribution may deviate from uniform. The `r_target` should be calibrated per-model to achieve desired valid nonce rate.

## File Structure (Target)

```
vllm/
├── poc/
│   ├── __init__.py
│   ├── config.py        # PoCConfig, PoCState
│   ├── data.py          # ProofBatch, ValidatedBatch
│   ├── gpu_random.py    # Deterministic GPU generation
│   ├── manager.py       # PoCManager
│   └── routes.py        # FastAPI endpoints
└── entrypoints/openai/
    ├── api_server.py    # Add: include_router(poc_router)
    └── cli_args.py      # Add: --enable-poc, --poc-batch-size, --poc-seq-len
```

