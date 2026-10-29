#!/usr/bin/env bash
set -euo pipefail

gpu=${GPU:-0}
test=${TEST:-"base"}

container_name="comfyui-omni-minimax-h3-gpu-$test"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
host_root="$SCRIPT_PATH/.."

mkdir -p "$host_root/comfyui-omni/gpu-$test-output" "$host_root/comfyui-omni/gpu-$test-user"
docker rm -f "$container_name" >/dev/null 2>&1 || true

# 先在容器里跑 optimize/build.sh（编译 h3_fused.so + 给 ComfyUI-OmniXPU 打 patch，均幂等），再跑 headless。
# H3_FUSED=0 时关掉融合路径（patch 保留，走原始 OmniXPU 路径），用于 A/B。
# H3_AB_CHECK=1 时改跑 optimize/ab_check.py：在一次 forward 上对比 stock / fused / torch 三条路径的输出。
# 接着跑 optimize_int8/build.sh（编译 h3_int8_attn.so + 在融合 patch 之上再打 patch）：融合路径里 rms_rope 和 Q/K 量化合成一个 kernel，
# attention 的 QKᵀ 走 INT8（block 45/48/49 除外）。依赖融合路径，H3_FUSED=0 时 INT8 也不生效。
# INT8 patch 改的是融合 patch 之后的代码，所以先撤 INT8 patch，再让 optimize/build.sh 检查融合 patch，最后重新打 INT8 patch。
# H3_INT8_ATTN=0 时关掉 INT8 attention（patch 保留，走原 rms_rope + bf16 FMHA），用于 A/B。
# H3_INT8_ATTN_CHECK=N 时前 N 次 INT8 attention 额外跑一次 bf16，日志打印两者的相对误差和最小余弦相似度。
entry=headless_minimax_h3.py
docker run --rm \
  --device=/dev/dri \
  --network=host \
  --name="$container_name" \
  -e http_proxy="http://child-prc.intel.com:913" \
  -e https_proxy="http://child-prc.intel.com:913" \
  -e no_proxy="localhost,127.0.0.1" \
  -e OMNIXPU_H3_FUSED_DIAG=1 \
  -e OMNIXPU_H3_FUSED_STAGE=1 \
  -e OMNI_ATTN_TRACE_CONTRACTS=1 \
  -e OMNIXPU_H3_FUSED="${H3_FUSED:-1}" \
  -e OMNIXPU_H3_INT8_ATTN="${H3_INT8_ATTN:-1}" \
  -e OMNIXPU_H3_INT8_ATTN_CHECK="${H3_INT8_ATTN_CHECK:-0}" \
  -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
  -e H3_AB_FORWARD="${H3_AB_FORWARD:-3}" \
  -v "$host_root/../ComfyUI/models:/models/host:ro" \
  -v "$host_root/ComfyUI:/workspace/ComfyUI:rw" \
  -v "$host_root/comfyui-omni/gpu-$test-output:/data/output" \
  -v "$host_root/comfyui-omni/gpu-$test-user:/data/user" \
  -v /dev/dri/by-path:/dev/dri/by-path --shm-size=32g \
  --workdir=/workspace/ComfyUI \
  --entrypoint bash \
  llm-scaler-omni:0.2.0-b2-comfyui-bmg \
  -c 'pip install -r custom_nodes/raylight/requirements.txt && exec python "$0" "$@"' "$entry" \
    --workflow "video_minimax_h3_t2v_raylight_api.json" \
    --runs 2 \
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
