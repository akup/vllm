import argparse
from typing import Dict, Optional

import requests

MODEL_ALIASES: Dict[str, str] = {
    "Qwen3-235B": "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
    "Qwen3-32B": "Qwen/Qwen3-32B-FP8",
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B-Instruct",
    "QwQ-32B": "Qwen/QwQ-32B",
}


def resolve_model(name: str) -> str:
    """Translate model alias into full model identifier."""
    if name in MODEL_ALIASES:
        return MODEL_ALIASES[name]

    if name in MODEL_ALIASES.values():
        return name

    allowed = ", ".join(list(MODEL_ALIASES.keys()) + list(MODEL_ALIASES.values()))
    raise ValueError(f"Unsupported model '{name}'. Allowed: {allowed}")


def inference_up(base_url: str, model: str, tensor_parallel_size: Optional[int]) -> dict:
    url = f"{base_url.rstrip('/')}/api/v1/inference/up"

    additional_args = []
    if tensor_parallel_size and tensor_parallel_size > 1:
        additional_args = ["--tensor-parallel-size", str(tensor_parallel_size), "--quantization", "fp8", "--kv-cache-dtype", "fp8"]

    payload = {
        "model": resolve_model(model),
        "dtype": "float16",
        "additional_args": additional_args,
    }

    additional_args_json = ", ".join(f'"{arg}"' for arg in additional_args)
    curl_cmd = (
        "curl -X POST "
        f"'{url}' "
        "-H 'Content-Type: application/json' "
        "-d '{"
        f"\"model\": \"{payload['model']}\", "
        f"\"dtype\": \"{payload['dtype']}\", "
        f"\"additional_args\": [{additional_args_json}]"
        "}'"
    )
    print(curl_cmd)

    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trigger inference up request.")
    parser.add_argument(
        "--model",
        required=True,
        help="Model alias or full name (Qwen3-235B, Qwen3-32B, "
        "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8, Qwen/Qwen3-32B-FP8, Qwen/Qwen2.5-7B-Instruct, Qwen/QwQ-32B).",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Tensor parallel size (2-8). If omitted or 1, no argument is passed.",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8080",
        help="Base URL for the inference service (default: http://localhost:8080).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.tensor_parallel_size is not None and (
        args.tensor_parallel_size < 1 or args.tensor_parallel_size > 8
    ):
        raise ValueError("tensor-parallel-size must be between 1 and 8.")

    result = inference_up(
        base_url=args.base_url,
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    print(result)


if __name__ == "__main__":
    main()