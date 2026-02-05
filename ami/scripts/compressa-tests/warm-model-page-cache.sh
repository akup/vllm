#!/usr/bin/env bash
# Same as warm-page-cache.sh: warm Linux page cache for model files under HF_HOME.
# Kept for compatibility (e.g. packer chmod, README).
exec "$(dirname "$0")/warm-page-cache.sh" "$@"
