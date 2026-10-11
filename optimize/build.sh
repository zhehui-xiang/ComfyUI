#!/usr/bin/env bash
# 在 llm-scaler-omni 容器里执行（ComfyUI 挂在 /workspace/ComfyUI）：
#   build  : 编译 h3_fused.cpp -> optimize/build/h3_fused.so（源文件没变就跳过）
#   patch  : 把 patches/*.patch 打到 custom_nodes/ComfyUI-OmniXPU（已打过就跳过，冲突则报错）
#   revert : 撤掉 patch
#   all    : build + patch（默认）
set -euo pipefail

dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
comfy_root="$(dirname "$dir")"
omnixpu="$comfy_root/custom_nodes/ComfyUI-OmniXPU"
so="$dir/build/h3_fused.so"
AOT_TARGET=${AOT_TARGET:-bmg-g21,bmg-g31}  # B580/B60 + B70

step_build() {
  if [[ -f $so && $so -nt $dir/h3_fused.cpp && $so -nt $dir/build.sh ]]; then
    echo "[optimize] $so is up to date"
    return
  fi
  mkdir -p "$dir/build"
  read -r torch_inc torch_lib py_inc abi < <(python - <<'EOF'
import sysconfig, torch, os
from torch.utils.cpp_extension import include_paths
print(include_paths()[0], os.path.join(os.path.dirname(torch.__file__), "lib"),
      sysconfig.get_paths()["include"], int(torch._C._GLIBCXX_USE_CXX11_ABI))
EOF
)
  echo "[optimize] building $so (AOT $AOT_TARGET)"
  icpx -fsycl -std=c++17 -O3 -DNDEBUG -fPIC -shared \
    -fsycl-targets=spir64_gen -Xsycl-target-backend "-device $AOT_TARGET" \
    -D_GLIBCXX_USE_CXX11_ABI="$abi" \
    -I"$py_inc" -I"$torch_inc" -I"$torch_inc/torch/csrc/api/include" \
    -Wno-deprecated-declarations -Wno-c++20-extensions \
    -L"$torch_lib" -ltorch -ltorch_python -ltorch_cpu -ltorch_xpu -lc10 -lc10_xpu -Wl,-rpath,"$torch_lib" \
    -o "$so.tmp" "$dir/h3_fused.cpp" 2>&1 | grep -vE "^(Compilation from IR|Build succeeded)" || true
  [[ -f $so.tmp ]] || { echo "[optimize] build failed" >&2; exit 1; }
  mv "$so.tmp" "$so"
  [[ -n ${HOST_UID:-} ]] && chown -R "$HOST_UID:${HOST_GID:-$HOST_UID}" "$dir/build"
  echo "[optimize] built $so"
}

git_omnixpu() { git -C "$omnixpu" -c safe.directory='*' "$@"; }

step_patch() {
  for p in "$dir"/patches/*.patch; do
    if git_omnixpu apply --reverse --check "$p" 2>/dev/null; then
      echo "[optimize] already applied: $(basename "$p")"
    elif git_omnixpu apply --check "$p"; then
      git_omnixpu apply "$p"
      echo "[optimize] applied: $(basename "$p")"
    else
      echo "[optimize] cannot apply $(basename "$p") to $omnixpu (local changes?)" >&2
      exit 1
    fi
  done
}

step_revert() {
  for p in $(ls -r "$dir"/patches/*.patch); do
    if git_omnixpu apply --reverse --check "$p" 2>/dev/null; then
      git_omnixpu apply --reverse "$p"
      echo "[optimize] reverted: $(basename "$p")"
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
