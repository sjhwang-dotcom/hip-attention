#!/bin/bash
# Launch SGLang server with HiP attention for InfiniteBench evaluation.
#
# Usage:
#   bash scripts/launch_hip_server.sh [model] [context_length] [port]
#
# Examples:
#   # Default: Llama-3.1-8B-Instruct, 128K context, port 30000
#   bash scripts/launch_hip_server.sh
#
#   # Custom model and context
#   bash scripts/launch_hip_server.sh meta-llama/Llama-3.1-8B-Instruct 262144 8000
#
# After server is ready, run the benchmark:
#   python scripts/bench_infinitebench.py \
#       --model meta-llama/Llama-3.1-8B-Instruct \
#       --tasks passkey,kv_retrieval,number_string \
#       --max-samples 20 \
#       --server-url http://localhost:30000

set -euo pipefail

MODEL="${1:-meta-llama/Llama-3.1-8B-Instruct}"
CONTEXT="${2:-131072}"
PORT="${3:-30000}"
CHUNK_SIZE="${CHUNK_SIZE:-32768}"
TP_SIZE="${TP_SIZE:-1}"

echo "============================================================"
echo "  Launching SGLang with HiP attention"
echo "  Model:   ${MODEL}"
echo "  Context: ${CONTEXT}"
echo "  Chunk:   ${CHUNK_SIZE}"
echo "  Port:    ${PORT}"
echo "  TP:      ${TP_SIZE}"
echo "============================================================"

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
HIP_DISABLE_AUTOTUNE=0 \
uv run -m sglang.launch_server \
    --model-path "${MODEL}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --context-length "${CONTEXT}" \
    --max-total-tokens "${CONTEXT}" \
    --chunked-prefill-size "${CHUNK_SIZE}" \
    --max-prefill-tokens "${CHUNK_SIZE}" \
    --attention-backend hip_attention \
    --hip-attention-config '{"mask_refresh_interval": [96, 24, 8]}' \
    --cuda-graph-bs 1 \
    --max-running-requests 1 \
    --tp-size "${TP_SIZE}" \
    --trust-remote-code
