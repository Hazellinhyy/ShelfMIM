#!/usr/bin/env bash
set -euo pipefail

# Usage example:
# bash scripts/run_full_pipeline.sh \
#   --data-root /root/projects/ShelfMIM/dataset/archive-Retail\ Product\ Checkout\ Dataset/retail_product_checkout \
#   --output-root outputs/oneclick_rpc \
#   --gpu-id 0 \
#   --device cuda:0 \
#   --num-workers 8

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" scripts/run_full_pipeline.py "$@"
