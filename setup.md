source .venv/bin/activate
export UV_INDEX_STRATEGY="unsafe-best-match"

# Install mamba build deps + mamba first
uv pip install packaging wheel ninja torch --extra-index-url https://download.pytorch.org/whl/cu128
uv pip install --no-build-isolation "git+https://github.com/state-spaces/mamba@v2.2.4"

# Then install the rest
uv pip install -r requirements/dev.txt --extra-index-url https://download.pytorch.org/whl/cu128


---

# Test case for dev env

Verify local code changes are active by running:

```bash
cd /home/ubuntu/workspace/vllm
source .venv/bin/activate
python test_qwen3_inference.py
```

Look for this log line in the output to confirm local vLLM code is being used:

```
INFO ... [qwen3.py:53] === QWEN3 MODEL LOADED FROM LOCAL VLLM CODE ===
```

This marker is added in `vllm/model_executor/models/qwen3.py` at line 53.