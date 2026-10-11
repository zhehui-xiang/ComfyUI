# H3 attention INT8 QKᵀ

MiniMax H3 每个 DiT block 的 attention（`[1,56,15787,128]`）里，QKᵀ 改用 INT8 DPAS，P·V 仍是 bf16。
在 `../optimize`（GEMM 周边融合）之上叠加，两者互相独立。分析过程见 `/home/test/xzh/prof_an/int8_attn`。

| 文件 | 作用 |
|---|---|
| `h3_int8_attn.cpp` | 注册 `torch.ops.h3i8attn.quant_qk`（Q 逐 token、K 每 32 token 一个 scale 的 INT8 量化）和 `fmha`（INT8 QKᵀ FMHA） |
| `int8_qk_fmha.hpp` | FMHA 的 mainloop / kernel，基于 sycl-tla `flash_attention_v2` 的 XeDefault 改写，不修改 sycl-tla |
| `build.sh` | 容器内执行：编译 `build/h3_int8_attn.so`（源码没变则跳过）+ 打 patch（已打则跳过）；`revert` 撤 patch |
| `patches/0001-omnixpu-h3-int8-attention.patch` | ComfyUI-OmniXPU `adapters/attention.py`：H3 H56 的 CUTE BHLD 路由里，满足条件时走 INT8，否则走原 `sdp_bhld_d128` |

## 使用

`run_headless_minimax_h3.sh` 在 `optimize/build.sh all` 之后跑 `optimize_int8/build.sh all`，再跑 headless：

```bash
bash run_headless_minimax_h3.sh                         # 融合 + INT8 attention
H3_INT8_ATTN=0 bash run_headless_minimax_h3.sh          # 关掉 INT8 attention（patch 保留），A/B 用
H3_INT8_ATTN_CHECK=100 bash run_headless_minimax_h3.sh  # 前 100 次 INT8 attention 额外跑 bf16，日志打印 rel_err / min_cos
```

撤掉 patch：在容器里执行 `optimize_int8/build.sh revert`。

走 INT8 的条件：OmniXPU 用 CUTE 后端（BMG）、日志有 `H3 int8 QK route enabled`、`transformer_options` 里有 `block_index`
且不在 `OMNIXPU_H3_INT8_ATTN_BF16_BLOCKS`（默认 `45,48,49`）里、Q/K/V 都是同一个 qkv projection 的视图（token refiner 不走）。

## 验证结果（B70，video_minimax_h3_t2v2.json，20 步）

- 速度：稳态每次 H3 forward 5.326 s → 5.109 s（−4.1%）；整个 workflow 147.0 s → 142.4 s。
  单层 attention 56.3 ms → 52.4 ms（`quant_qk` 1.37 ms + `fmha` 51.0 ms）。
- 精度：`H3_INT8_ATTN_CHECK` 检查全部 20 步 × 49 个 block（56 个 head）：block 45 / 48 / 49 有 token 方向错误
  （最差 cos 0.10 / 0.80 / 负值），默认留 bf16；其余 block 相对误差 ≤ 1.7%，最差 token cos ≥ 0.983。
- 视频：bf16 同 seed 两次运行之间 PSNR 21.4 dB（流水线不可逐位复现，构图本身就会变），INT8 vs bf16 约 18 dB；
  场景、风格、镜头结构一致，没有可见瑕疵。

## 还没做的

- 量化合进 rms_rope：现在 `rms_rope_split_half_` 先把 bf16 Q/K 写回 qkv buffer，`quant_qk` 再读一遍；
  合并后不用写回 bf16，预计每层再省 ~2 ms（rms_rope 2.84 ms + quant 1.37 ms → ~1 ms）。
- QKᵀ 本身：kernel 里 INT8 QKᵀ 约 17 ms，按实测 int8 峰值理想值 10.6 ms（Q 每个 KV tile 都重新从 L1 读）。
- 只在 B70 上调过配置和验证过；AOT 同时编了 `bmg-g21`（B580/B60），未在这些卡上测试。
