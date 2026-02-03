#!/usr/bin/env python3
"""
Start vLLM OpenAI API server with a runtime fix for tokenizers that lack
all_special_tokens_extended (e.g. Qwen2Tokenizer). Use this instead of
rebuilding vLLM when you hit:

  AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended

Usage (same args as the real server):
  python run_api_server_qwen_fix.py --model Qwen/Qwen3-8B --dtype auto --port 8080 --host 0.0.0.0
"""
import sys


def _apply_patch():
    import vllm.transformers_utils.tokenizer as tokenizer_mod

    original = tokenizer_mod.get_cached_tokenizer

    def patched_get_cached_tokenizer(tokenizer):
        # Ensure tokenizer has all_special_tokens_extended so original won't raise.
        # Fallback order: additional_special_tokens, then all_special_tokens.
        if getattr(tokenizer, "all_special_tokens_extended", None) is None:
            tokenizer.all_special_tokens_extended = getattr(
                tokenizer, "additional_special_tokens", None
            ) or getattr(tokenizer, "all_special_tokens", [])
        return original(tokenizer)

    tokenizer_mod.get_cached_tokenizer = patched_get_cached_tokenizer


if __name__ == "__main__":
    _apply_patch()
    import runpy
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
