# Development Rules

## Core Philosophy

- **Crazy Simple**: Minimal code, maximum clarity
- **Minimalistic**: Single responsibility, no boilerplate, no unnecessary abstraction
- **Standard**: Follow established patterns and project structures
- **Clean**: Pure functionality, no comments explaining obvious code
- **Modern**: Use contemporary tooling and best practices
- **Concentrated**: Information-dense content with no fluff

## PoC Integration Rules

- **Minimal vLLM changes**: PoC is an additional mode, not a core feature
- **Simple over clever**: No complicated logic, straightforward data flow
- **Effective but lean**: Optimize for readability, not premature optimization
- **Isolated module**: All PoC code in `vllm/poc/`, easy to remove or migrate
- **Same interfaces**: Maintain API compatibility with original pow package

## Code Standards

- No emoji in code, documentation, or commits
- Prefer explicit over implicit
- Keep files focused and small
- No unnecessary abstractions or wrapper classes
- Direct function calls over complex class hierarchies

## What NOT to Do

- Do not modify vLLM's core inference path
- Do not add new dependencies unless absolutely necessary
- Do not create abstraction layers "for future flexibility"
- Do not add configuration options that aren't immediately needed
- Do not write code that requires comments to understand

## Dev Environment

```bash
source .venv/bin/activate
```

## Reference Implementation

During implementation of each phase, cross-check logic with the original PoW implementation:

```
/home/ubuntu/workspace/gonka/mlnode/packages/pow/src/pow/
```

Key files to reference (not limited to them during cross check):
- `data.py` - ProofBatch, ValidatedBatch schemas
- `legacy_random.py` - Seed generation, target vector, distance calculation
- `compute/compute.py` - Core computation logic
- `service/routes.py` - API endpoints
- `service/manager.py` - PowManager, state management

Note: Logic may differ where explicitly discussed (e.g., RNG implementation uses murmur3 instead of numpy SeedSequence for cross-device portability). Use original as reference, not as exact specification.

## Phase Completion

After completing each phase:
1. Create `phase-{i}-final.md` documenting implemented files, decisions, and acceptance criteria
2. Commit all changes with descriptive message
3. Check the last git commit to get context for previous step implementation when starting a new phase

## Gonka Chain Context

Gonka is decentralized AI infrastructure for computational power verification. PoC proves nodes have sufficient memory bandwidth to run LLM inference.
