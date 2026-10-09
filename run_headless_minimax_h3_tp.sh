#!/usr/bin/env bash
set -euo pipefail

container_name="comfyui-omni-minimax-h3-tp"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
host_root="$SCRIPT_PATH/.."

mkdir -p "$host_root/comfyui-omni/gpu0-tp-output" "$host_root/comfyui-omni/gpu0-tp-user"
docker rm -f "$container_name" >/dev/null 2>&1 || true


docker run --rm \
  --device=/dev/dri \
  --network=host \
  --name="$container_name" \
  -v "$SCRIPT_PATH/headless_minimax_h3.py:/llm/ComfyUI/headless_minimax_h3.py:ro" \
  -v "$SCRIPT_PATH/video_minimax_h3_t2v2.json:/llm/ComfyUI/video_minimax_h3_t2v2.json:ro" \
  -v "$host_root/ComfyUI/models:/models/host:ro" \
  -v "$host_root/comfyui-omni/gpu0-tp-output:/data/output" \
  -v "$host_root/comfyui-omni/gpu0-tp-user:/data/user" \
  -v "$host_root/xzh/minimax-h3-tp.fM9AzH/patches.py:/llm/ComfyUI/custom_nodes/ComfyUI-OmniXPU/patches/__init__.py:ro" \
  -v "$host_root/xzh/minimax-h3-tp.fM9AzH/h3_row_tp.py:/llm/ComfyUI/custom_nodes/ComfyUI-OmniXPU/adapters/h3_row_tp.py:ro" \
  -v "$host_root/xzh/minimax-h3-tp.fM9AzH/h3_row_tp_runtime.py:/llm/ComfyUI/custom_nodes/ComfyUI-OmniXPU/h3_row_tp_runtime.py:ro" \
  --workdir=/llm/ComfyUI \
  llm-scaler-omni:0.2.0-b2-comfyui-bmg \
  python headless_minimax_h3.py \
    --workflow video_minimax_h3_t2v2.json \
    --runs 2 \
    --extra-model-paths-config /llm/configs/comfyui_host_models.yaml \
    --output-directory /data/output \
    --user-directory /data/user