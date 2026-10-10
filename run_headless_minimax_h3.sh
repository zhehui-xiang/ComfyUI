#!/usr/bin/env bash
set -euo pipefail

gpu=0

container_name="comfyui-omni-minimax-h3"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
host_root="$SCRIPT_PATH/.."

mkdir -p "$host_root/comfyui-omni/gpu$gpu-output" "$host_root/comfyui-omni/gpu$gpu-user"
docker rm -f "$container_name" >/dev/null 2>&1 || true

docker run --rm \
  --device=/dev/dri \
  --network=host \
  --name="$container_name" \
  -e ZE_AFFINITY_MASK=$gpu \
  -e OMNI_ATTN_TRACE_CONTRACTS=1 \
  -v "$host_root/../ComfyUI/models:/models/host:ro" \
  -v "$host_root/ComfyUI:/workspace/ComfyUI:rw" \
  -v "$host_root/comfyui-omni/gpu$gpu-output:/data/output" \
  -v "$host_root/comfyui-omni/gpu$gpu-user:/data/user" \
  --workdir=/workspace/ComfyUI \
  llm-scaler-omni:0.2.0-b2-comfyui-bmg \
  python headless_minimax_h3.py \
    --workflow "video_minimax_h3_t2v2.json" \
    --runs 1 \
    --extra-model-paths-config /llm/configs/comfyui_host_models.yaml \
    --output-directory /data/output \
    --user-directory /data/user \
    "$@"


#bash run_headless_minimax_h3.sh   --profile-h3-forward-dir /data/output/profiler   --profile-h3-forward-start 5   --profile-h3-forward-count 1

#attention backend A/B， cute or torch
#-e OMNI_ATTN_BACKEND=torch \

#记录 attention 的调用位置、shape、stride、dtype 和布局标记。
#-e OMNIXPU_TRACE_CONTRACTS=1 \ xxxxx
#-e OMNI_ATTN_TRACE_CONTRACTS=1 \