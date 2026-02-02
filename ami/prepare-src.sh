#!/bin/bash
# Helper script to create a tarball of the vLLM source for Packer
# Run this from the vLLM repo root before running packer build

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TARBALL_PATH="$REPO_ROOT/ami/vllm-src.tar.gz"

cd "$REPO_ROOT"

echo "Creating vLLM source tarball for Packer..."
echo "Repository root: $REPO_ROOT"
echo "Output: $TARBALL_PATH"

# Create tarball excluding common build artifacts and git files
tar -czf "$TARBALL_PATH" \
  --exclude='.git' \
  --exclude='.gitignore' \
  --exclude='*.pyc' \
  --exclude='__pycache__' \
  --exclude='*.egg-info' \
  --exclude='.venv' \
  --exclude='venv' \
  --exclude='env' \
  --exclude='build' \
  --exclude='dist' \
  --exclude='*.so' \
  --exclude='.pytest_cache' \
  --exclude='.mypy_cache' \
  --exclude='logs' \
  --exclude='ami/vllm-src.tar.gz' \
  vllm/ \
  setup.py \
  pyproject.toml \
  MANIFEST.in \
  README.md \
  requirements.txt \
  2>/dev/null || {
  # Fallback: include everything except .git and the tarball itself
  tar -czf "$TARBALL_PATH" \
    --exclude='.git' \
    --exclude='ami/vllm-src.tar.gz' \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    .
}

echo "Tarball created: $TARBALL_PATH"
echo "Size: $(du -h "$TARBALL_PATH" | cut -f1)"
