// MiniMax H3 attention 的 INT8 QKᵀ 路径，注册为 torch.ops.h3i8attn.{quant_qk, fmha}。
//
// quant_qk(q, k) -> (q8, k8, q_scale, k_scale)
//   q, k     bf16 [1, H, L, 128] 视图（H3 里是 qkv buffer 的视图，rms_rope 之后）
//   q8, k8   int8 [1, H, L, 128]，BLHD 存储
//   q_scale  fp32 [H, L]                逐 token
//   k_scale  fp32 [H, ceil(L/kKvTile)]  每个 KV tile 一个
// fmha(q8, k8, v, q_scale, k_scale) -> o
//   v        bf16 [1, H, L, 128] 视图；o 为 BLHD-backed 的 [1, H, L, 128]（同 cute sdp_bhld_d128）
//
// FMHA 配置在 B70 上扫过 40 个：Q256 / KV32 / V32 / MmaK64 / 16 subgroup / 2 stage（见 prof_an/int8_attn）。
#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cmath>
#include <limits>

#include <cute/tensor.hpp>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/device_kernel.h"
#include "cute/util/compat.hpp"

#include "int8_qk_fmha.hpp"

using namespace cute;

namespace {

constexpr int kHeadDim = 128;
constexpr int kQTile = 256, kKvTile = 32, kVTile = 32, kMmaK = 64, kSgQ = 16, kStages = 2;

// ---- Q/K 量化：一个 work-group 处理一个 (head, KV tile) ----------------------------------------
constexpr int kSg = 16;                   // sub-group 大小
constexpr int kVec = kHeadDim / kSg;      // 每 lane 8 个元素
constexpr int kSgPerWg = 16;
constexpr int kRowsPerSg = kKvTile / kSgPerWg;

using bf16x8 = sycl::vec<uint16_t, kVec>;  // 每 lane 一次 16 字节读
using i8x8 = sycl::vec<int8_t, kVec>;

inline float bf2f(uint16_t b) { return sycl::bit_cast<float>(static_cast<uint32_t>(b) << 16); }

class QuantQKKernel {};

void quant_qk_launch(const uint16_t* q, const uint16_t* k, int64_t q_seq, int64_t q_head, int64_t k_seq, int64_t k_head,
                     int8_t* q8, int8_t* k8, float* q_scale, float* k_scale, int H, int L) {
  const int blocks = (L + kKvTile - 1) / kKvTile;
  const int64_t row8 = int64_t(H) * kHeadDim;  // q8/k8 的 seq stride
  auto& queue = at::xpu::getCurrentXPUStream().queue();
  queue.parallel_for<QuantQKKernel>(
      sycl::nd_range<1>(int64_t(H) * blocks * kSgPerWg * kSg, kSgPerWg * kSg),
      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSg)]] {
        // 相邻 work-group 是同一批 token 的不同 head，读写的 qkv / q8 / k8 行在内存里连续
        const int h = it.get_group_linear_id() % H, blk = it.get_group_linear_id() / H;
        auto sg = it.get_sub_group();
        const int lane = sg.get_local_linear_id();
        const int d0 = lane * kVec;
        float kv[kRowsPerSg][kVec];
        float kmax = 0.f;
#pragma unroll
        for (int r = 0; r < kRowsPerSg; ++r) {
          const int t = blk * kKvTile + sg.get_group_linear_id() * kRowsPerSg + r;
          bf16x8 qraw(0), kraw(0);
          if (t < L) {
            qraw = *reinterpret_cast<const bf16x8*>(q + t * q_seq + h * q_head + d0);
            kraw = *reinterpret_cast<const bf16x8*>(k + t * k_seq + h * k_head + d0);
          }
          float qv[kVec];
          float qmax = 0.f;
#pragma unroll
          for (int e = 0; e < kVec; ++e) {
            qv[e] = bf2f(qraw[e]);
            kv[r][e] = bf2f(kraw[e]);
            qmax = sycl::fmax(qmax, sycl::fabs(qv[e]));
            kmax = sycl::fmax(kmax, sycl::fabs(kv[r][e]));
          }
          const float qs = sycl::fmax(sycl::reduce_over_group(sg, qmax, sycl::maximum<float>()), 1e-8f) / 127.f;
          const float qinv = 1.f / qs;
          if (t < L) {
            i8x8 out;
#pragma unroll
            for (int e = 0; e < kVec; ++e) out[e] = int8_t(sycl::clamp(sycl::rint(qv[e] * qinv), -127.f, 127.f));
            *reinterpret_cast<i8x8*>(q8 + t * row8 + h * kHeadDim + d0) = out;
            if (lane == 0) q_scale[int64_t(h) * L + t] = qs;
          }
        }
        const float ks = sycl::fmax(sycl::reduce_over_group(it.get_group(), kmax, sycl::maximum<float>()), 1e-8f) / 127.f;
        const float kinv = 1.f / ks;
#pragma unroll
        for (int r = 0; r < kRowsPerSg; ++r) {
          const int t = blk * kKvTile + sg.get_group_linear_id() * kRowsPerSg + r;
          if (t < L) {
            i8x8 out;
#pragma unroll
            for (int e = 0; e < kVec; ++e) out[e] = int8_t(sycl::clamp(sycl::rint(kv[r][e] * kinv), -127.f, 127.f));
            *reinterpret_cast<i8x8*>(k8 + t * row8 + h * kHeadDim + d0) = out;
          }
        }
        if (it.get_local_linear_id() == 0) k_scale[int64_t(h) * blocks + blk] = ks;
      });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> quant_qk(const at::Tensor& q, const at::Tensor& k) {
  TORCH_CHECK(q.dim() == 4 && q.size(0) == 1 && q.size(3) == kHeadDim && k.sizes() == q.sizes(),
              "quant_qk expects matching [1, H, L, 128] q/k");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16, "quant_qk expects bf16");
  TORCH_CHECK(q.stride(3) == 1 && k.stride(3) == 1, "quant_qk expects unit head_dim stride");
  for (const at::Tensor* t : {&q, &k})  // 16 字节向量读
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0 && t->stride(1) % 8 == 0 && t->stride(2) % 8 == 0,
                "quant_qk expects 16-byte aligned rows");
  const int H = q.size(1), L = q.size(2);
  const int blocks = (L + kKvTile - 1) / kKvTile;
  auto i8 = q.options().dtype(at::kChar), f32 = q.options().dtype(at::kFloat);
  at::Tensor q8 = at::empty({1, L, H, kHeadDim}, i8), k8 = at::empty({1, L, H, kHeadDim}, i8);
  at::Tensor q_scale = at::empty({H, L}, f32), k_scale = at::empty({H, blocks}, f32);
  quant_qk_launch(reinterpret_cast<const uint16_t*>(q.data_ptr()), reinterpret_cast<const uint16_t*>(k.data_ptr()),
                  q.stride(2), q.stride(1), k.stride(2), k.stride(1),
                  q8.data_ptr<int8_t>(), k8.data_ptr<int8_t>(), q_scale.data_ptr<float>(), k_scale.data_ptr<float>(),
                  H, L);
  return {q8.transpose(1, 2), k8.transpose(1, 2), q_scale, k_scale};
}

// ---- INT8 QKᵀ FMHA ------------------------------------------------------------------------------
template <typename Kernel>
class Int8QKKernelTag {};

template <typename Kernel>
void launch_on_torch_queue(typename Kernel::Params params) {
  compat::dim3 const block = Kernel::get_block_shape();
  compat::dim3 const grid = Kernel::get_grid_shape(params);
  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;
  compat::experimental::launch_properties launch_props{syclex::work_group_scratch_size(Kernel::SharedStorageSize)};
  compat::experimental::kernel_properties kernel_props{
      syclex::sub_group_size<cute::intel::sg_size>, intelex::grf_size<256>};
  compat::experimental::launch_policy policy{grid, block, launch_props, kernel_props};
  syclex::launch_config config(policy.get_range(), policy.get_launch_properties());
  auto cgf = [&](::sycl::handler& cgh) {
    auto functor =
        compat::experimental::detail::build_kernel_functor<cutlass::device_kernel<Kernel>>(cgh, policy, params);
    syclex::detail::LaunchConfigAccess<sycl::nd_range<3>, decltype(policy.get_launch_properties())> access(config);
    cgh.parallel_for<Int8QKKernelTag<Kernel>>(access.getRange(), access.getProperties(), functor);
  };
  at::xpu::getCurrentXPUStream().queue().submit(cgf);
}

struct Int8QKTile {
  using ShapeQK = Shape<Int<kQTile>, Int<kKvTile>, Int<kMmaK>>;
  using ShapePV = Shape<Int<kQTile>, Int<kVTile>, Int<kKvTile>>;
  using ShapeOutput = Shape<Int<kQTile>, Int<kHeadDim>>;
  using SubgroupLayoutQK = Layout<Shape<Int<kSgQ>, _1, _1>>;
  static constexpr int AtomM = cute::gcd(kQTile / kSgQ, 8);

  using StrideQ = Stride<int, _1, int, int>;
  using StrideK = Stride<int, _1, int, int>;
  using StrideV = Stride<_1, int, int, int>;
  using StrideO = Stride<int, _1, int, int>;

  using TiledMMAQK = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<AtomM, int32_t, int8_t, int8_t>>,
                                             Layout<ShapeQK>, SubgroupLayoutQK>::TiledMMA;
  using TiledMMAQKf = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<AtomM, float, cutlass::bfloat16_t>>,
                                              Layout<ShapeQK>, SubgroupLayoutQK>::TiledMMA;
  using SubgroupLayoutPV = decltype(cutlass::fmha::collective::get_sg_layout_pv(SubgroupLayoutQK{}));
  using TiledMMAPV = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<AtomM, float, cutlass::bfloat16_t>>,
                                             Layout<ShapePV>, SubgroupLayoutPV>::TiledMMA;

  template <class T, class S>
  using GTensor = decltype(make_tensor(make_gmem_ptr((T*)nullptr), make_layout(repeat<rank_v<S>>(1), S{})));
  using TensorQ = GTensor<int8_t, StrideQ>;
  using TensorK = GTensor<int8_t, StrideK>;
  using TensorV = GTensor<cutlass::bfloat16_t, StrideV>;
  using TensorO = GTensor<cutlass::bfloat16_t, StrideO>;

  using Mainloop = cutlass::fmha::collective::Int8QKMainloop<
      kStages, TiledMMAQK, TiledMMAQKf, TiledMMAPV, kHeadDim / kVTile, TensorQ, TensorK, TensorV>;
  using Epilogue = cutlass::fmha::collective::FMHAFwdEpilogue<Mainloop, ShapeOutput, TensorO, void>;
  using ProblemShape = cutlass::fmha::kernel::FMHAProblemShape<false>;
  using Kernel = cutlass::fmha::kernel::XeFMHAFwdInt8QKKernel<
      ProblemShape, Mainloop, Epilogue, cutlass::fmha::kernel::XeFHMAIndividualTileScheduler>;
};

int checked_int(int64_t value, const char* label) {
  TORCH_CHECK(value >= 0 && value <= std::numeric_limits<int>::max(), label, " exceeds int32: ", value);
  return static_cast<int>(value);
}

template <typename S>
S seq_major_stride(const at::Tensor& t) {
  return cute::make_stride(checked_int(t.stride(2), "seq stride"), _1{},
                           checked_int(t.stride(1), "head stride"), checked_int(t.stride(0), "batch stride"));
}

at::Tensor fmha(const at::Tensor& q8, const at::Tensor& k8, const at::Tensor& v,
                const at::Tensor& q_scale, const at::Tensor& k_scale) {
  using KT = Int8QKTile;
  using K = KT::Kernel;
  TORCH_CHECK(q8.dim() == 4 && q8.size(0) == 1 && q8.size(3) == kHeadDim, "fmha expects [1, H, L, 128]");
  TORCH_CHECK(q8.scalar_type() == at::kChar && k8.scalar_type() == at::kChar && v.scalar_type() == at::kBFloat16,
              "fmha expects int8 q/k and bf16 v");
  TORCH_CHECK(k8.sizes() == q8.sizes() && v.sizes() == q8.sizes(), "fmha expects self-attention shapes");
  TORCH_CHECK(q8.stride(3) == 1 && k8.stride(3) == 1 && v.stride(3) == 1, "fmha expects unit head_dim stride");
  const int H = q8.size(1), L = q8.size(2);
  TORCH_CHECK(q_scale.is_contiguous() && q_scale.sizes() == at::IntArrayRef({H, L}) &&
                  k_scale.is_contiguous() && k_scale.size(0) == H && k_scale.size(1) == (L + kKvTile - 1) / kKvTile,
              "fmha scales must come from quant_qk");

  at::Tensor o = at::empty({1, L, H, kHeadDim}, v.options()).permute({0, 2, 1, 3});

  typename KT::ProblemShape shape;
  shape.batch = 1;
  shape.num_heads_q = shape.num_heads_kv = H;
  shape.seq_len_qo = shape.seq_len_kv = L;
  shape.seq_len_kv_cache = 0;
  shape.head_size_qk = shape.head_size_vo = kHeadDim;

  typename KT::StrideV stride_v = cute::make_stride(
      _1{}, checked_int(v.stride(2), "V seq stride"), checked_int(v.stride(1), "V head stride"),
      checked_int(v.stride(0), "V batch stride"));

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  typename K::Arguments arguments{
      {shape,
       static_cast<const int8_t*>(q8.data_ptr()), seq_major_stride<KT::StrideQ>(q8),
       static_cast<const int8_t*>(k8.data_ptr()), seq_major_stride<KT::StrideK>(k8),
       static_cast<const cutlass::bfloat16_t*>(v.data_ptr()), stride_v,
       static_cast<cutlass::bfloat16_t*>(o.data_ptr()), seq_major_stride<KT::StrideO>(o)},
      {1.0f / std::sqrt(float(kHeadDim)), q_scale.data_ptr<float>(), k_scale.data_ptr<float>(),
       L, checked_int(k_scale.size(1), "k_scale blocks")},
      {},
      hw_info};
  TORCH_CHECK(K::can_implement(arguments), "h3i8attn: can_implement failed");
  launch_on_torch_queue<K>(K::to_underlying_arguments(arguments, nullptr));
  return o;
}

}  // namespace

TORCH_LIBRARY(h3i8attn, m) {
  m.def("quant_qk(Tensor q, Tensor k) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def("fmha(Tensor q8, Tensor k8, Tensor v, Tensor q_scale, Tensor k_scale) -> Tensor");
  m.impl("quant_qk", c10::DispatchKey::XPU, &quant_qk);
  m.impl("fmha", c10::DispatchKey::XPU, &fmha);
}
