# H3 attention INT8 QKᵀ

MiniMax H3 每个 DiT block 的 attention（`[1,56,15787,128]`）里，QKᵀ 改用 INT8 DPAS，P·V 仍是 bf16；
q/k 的 rms_rope 和 INT8 量化合成一个 kernel，不再把 bf16 Q/K 写回 qkv buffer。
叠加在 `../optimize`（GEMM 周边融合）之上：patch 改的是融合路径 `_run_fused_block` 里的 attention 部分。
分析过程见 `/home/test/xzh/prof_an/int8_attn`。

| 文件 | 作用 |
|---|---|
| `h3_int8_attn.cpp` | 注册 `torch.ops.h3i8attn.rms_rope_quant`（rms_rope + Q 逐 token / K 每 32 token 一个 scale 的 INT8 量化）和 `fmha`（INT8 QKᵀ FMHA） |
| `int8_qk_fmha.hpp` | FMHA 的 mainloop / kernel，基于 sycl-tla `flash_attention_v2` 的 XeDefault 改写，不修改 sycl-tla；mainloop 优化见 `/home/test/xzh/prof_an/qk_mainloop` |
| `build.sh` | 容器内执行：编译 `build/h3_int8_attn.so` + 打 patch（已打则跳过，融合 patch 没打则报错）；`revert` 撤 patch。`build/h3_int8_attn.so.stamp` 是源文件、`build.sh`、`AOT_TARGET`、编译器版本的哈希，一致才跳过编译（和 `.so` 一起提交） |
| `patches/0001-omnixpu-h3-int8-attention.patch` | ComfyUI-OmniXPU `adapters/h3_rms_modulation.py`（融合 patch 之后）：满足条件的 block 走 `rms_rope_quant` + `fmha`，否则走原 rms_rope + 选中的 attention |

## 使用

`run_headless_minimax_h3.sh` 每次启动干净容器后依次执行 `optimize_int8/build.sh revert`、`optimize/build.sh all`、
`optimize_int8/build.sh all`，再跑 headless（INT8 patch 改的是融合 patch 之后的代码，先撤掉它，融合 patch 的检查才能通过）：

```bash
bash run_headless_minimax_h3.sh                         # 融合 + INT8 attention（默认 GPU 0）
GPU=1 bash run_headless_minimax_h3.sh                   # 换卡（容器名带卡号，不会删掉其它卡上的容器）
H3_INT8_ATTN=0 bash run_headless_minimax_h3.sh          # 关掉 INT8 attention（patch 保留），A/B 用
H3_INT8_ATTN_CHECK=100 bash run_headless_minimax_h3.sh  # 前 100 次 INT8 attention 在 qkv 副本上再跑一遍原路径，日志打印 rel_err / min_cos
```

手动撤 patch：在容器里先 `optimize_int8/build.sh revert`，再 `optimize/build.sh revert`。

走 INT8 的条件：融合路径生效（`H3_FUSED=0` 时 INT8 也不生效）、日志有 `H3 int8 attention enabled`、
`transformer_options["block_index"]` 不在 `OMNIXPU_H3_INT8_ATTN_BF16_BLOCKS`（默认 `45,48,49`）里、rope freqs 为 bf16 且 rot_dim 96。
`get_stats()["int8_attention"]` 是走 INT8 的 block 数。

## 数值

- `rms_rope_quant` 与原 rms_rope（omni_xpu_kernel H3 快速路径）+ 同口径量化相比：int8 值 99.99% 相同（其余差 1），
  K scale 完全一致，Q scale 偶尔差 1 个 bf16 ulp（平方和归约顺序不同）；attention 输出相对差 1.2e-3。
- block 45 / 48 / 49 的 INT8 QKᵀ 会让部分 token 方向出错（20 步里最差 cos 0.10 / 0.80 / 负值），默认留 bf16；
  其余 block 相对误差 ≤ 1.7%，最差 token cos ≥ 0.98。

## 验证结果（B70，video_minimax_h3_t2v2.json，20 步）

单层（L=15787）：

| | 原 bf16 | 单独量化（上一版） | rms_rope 融合量化（当前） |
|---|---|---|---|
| rms_rope / 量化 | 2.66 ms | 2.66 + 1.37 ms | 1.76 ms |
| attention | 56.3 ms | 51.0 ms | 51.0 ms → 43.0 ms（mainloop 优化） |

端到端（GPU 1 上同一张卡前后连续跑；这张卡的基线比 GPU 0 慢约 7%，绝对值不能跨卡比）：

| | 融合 | 融合 + INT8 attention |
|---|---|---|
| 稳态每次 H3 forward（第 3~20 次） | 5.726 s | 5.366 s（−6.3%） |
| 整个 workflow | 154.6 s | 147.8 s |

mainloop 优化（反量化并进 softmax FMA、MMA-K 循环编译期展开、最后一个 KV tile 单独 mask）前后，GPU 0 上连续跑：

| | 改前 | 改后（两次） |
|---|---|---|
| 稳态每次 H3 forward（第 3~20 次） | 5.040 s | 4.641 / 4.650 s（−7.8%） |
| 整个 workflow | 138.8 s | 130.8 / 132.3 s |

改前改后 attention 输出只差 bf16 舍入（相对 8e-5，最大差 1 ulp）。

- `H3_INT8_ATTN_CHECK` 检查全部 20 步 × 47 个 INT8 block：相对误差最大 1.65%、平均 0.83%，最差 token cos 0.980（mainloop 优化前 1.64% / 0.82% / 0.983）。
- 视频：INT8 两次运行之间 PSNR 19.8 dB，INT8 vs bf16 20.7 dB，在流水线本身不可复现的噪声范围内。

## 还没做的

- `rms_rope_quant` 1.76 ms，只算量化不做 rope 的下限是 1.36 ms，按读写字节数的理想值约 0.7 ms。
- 只在 B70 上调过配置和验证过；AOT 同时编了 `bmg-g21`（B580/B60），未在这些卡上测试。
