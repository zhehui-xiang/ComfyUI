#!/usr/bin/env bash
# 在 llm-scaler-omni 容器里执行（ComfyUI 挂在 /workspace/ComfyUI）：
#   build  : 编译 h3_int8_attn.cpp -> optimize_int8/build/h3_int8_attn.so（源文件没变就跳过）
#   patch  : 把 patches/*.patch 打到 custom_nodes/ComfyUI-OmniXPU（已打过就跳过，冲突则报错）；
#            patch 改的是 ../optimize 融合 patch 之后的 h3_rms_modulation.py，要先打融合 patch
#   revert : 撤掉 patch（要在 ../optimize/build.sh revert 之前）
#   all    : build + patch（默认）
set -euo pipefail

dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
comfy_root="$(dirname "$dir")"
omnixpu="$comfy_root/custom_nodes/ComfyUI-OmniXPU"
so="$dir/build/h3_int8_attn.so"
AOT_TARGET=${AOT_TARGET:-bmg-g21,bmg-g31}  # B580/B60 + B70
CUTLASS_SYCL_ROOT=${CUTLASS_SYCL_ROOT:-/llm/sycl-tla}

step_build() {
  if [[ -f $so && $so -nt $dir/h3_int8_attn.cpp && $so -nt $dir/int8_qk_fmha.hpp && $so -nt $dir/build.sh ]]; then
    echo "[optimize_int8] $so is up to date"
    return
  fi
  mkdir -p "$dir/build"
  read -r torch_inc torch_lib py_inc abi < <(python - <<'PY'
import sysconfig, torch, os
from torch.utils.cpp_extension import include_paths
print(include_paths()[0], os.path.join(os.path.dirname(torch.__file__), "lib"),
      sysconfig.get_paths()["include"], int(torch._C._GLIBCXX_USE_CXX11_ABI))
PY
)
  echo "[optimize_int8] building $so (AOT $AOT_TARGET)"
  # 编译参数同 omni_xpu_kernel/setup.py 的 cute_fmha_torch
  icpx -fsycl -std=c++17 -O3 -DNDEBUG -fPIC -shared \
    -fsycl-targets=spir64_gen -Xsycl-target-backend "-device $AOT_TARGET" \
    -Xspirv-translator \
    -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate \
    -fno-sycl-instrument-device-code \
    -DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET \
    -D_GLIBCXX_USE_CXX11_ABI="$abi" \
    -I"$dir" -I"$CUTLASS_SYCL_ROOT/include" -I"$CUTLASS_SYCL_ROOT/tools/util/include" \
    -I"$CUTLASS_SYCL_ROOT/applications" \
    -I"$py_inc" -I"$torch_inc" -I"$torch_inc/torch/csrc/api/include" \
    -Wno-unknown-pragmas -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unused-local-typedef \
    -Wno-uninitialized -Wno-reorder-ctor -Wno-logical-op-parentheses -Wno-unused-function \
    -Wno-deprecated-copy -Wno-deprecated-declarations -Wno-c++20-extensions \
    -L"$torch_lib" -ltorch -ltorch_python -ltorch_cpu -ltorch_xpu -lc10 -lc10_xpu -Wl,-rpath,"$torch_lib" \
    -o "$so.tmp" "$dir/h3_int8_attn.cpp" 2>&1 | grep -vE "^(Compilation from IR|Build succeeded)" || true
  [[ -f $so.tmp ]] || { echo "[optimize_int8] build failed" >&2; exit 1; }
  mv "$so.tmp" "$so"
  [[ -n ${HOST_UID:-} ]] && chown -R "$HOST_UID:${HOST_GID:-$HOST_UID}" "$dir/build"
  echo "[optimize_int8] built $so"
}

git_omnixpu() { git -C "$omnixpu" -c safe.directory='*' "$@"; }

step_patch() {
  for p in "$dir"/patches/*.patch; do
    if git_omnixpu apply --reverse --check "$p" 2>/dev/null; then
      echo "[optimize_int8] already applied: $(basename "$p")"
      continue
    fi
    for fused in "$comfy_root"/optimize/patches/*.patch; do
      git_omnixpu apply --reverse --check "$fused" 2>/dev/null ||
        { echo "[optimize_int8] needs $(basename "$fused") applied first (optimize/build.sh patch)" >&2; exit 1; }
    done
    if git_omnixpu apply --check "$p"; then
      git_omnixpu apply "$p"
      echo "[optimize_int8] applied: $(basename "$p")"
    else
      echo "[optimize_int8] cannot apply $(basename "$p") to $omnixpu (local changes?)" >&2
      exit 1
    fi
  done
}

step_revert() {
  for p in $(ls -r "$dir"/patches/*.patch); do
    if git_omnixpu apply --reverse --check "$p" 2>/dev/null; then
      git_omnixpu apply --reverse "$p"
      echo "[optimize_int8] reverted: $(basename "$p")"
    fi
  done
}

case ${1:-all} in
  build) step_build ;;
  patch) step_patch ;;
  revert) step_revert ;;
  all) step_build; step_patch ;;
  *) echo "usage: $0 [build|patch|revert|all]" >&2; exit 2 ;;
esac
