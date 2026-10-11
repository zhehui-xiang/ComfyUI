# H3 int8 GEMM 周边融合

把 MiniMax H3 每个 DiT block 里 int8 GEMM 前的激活预处理（RMSNorm/modulation、ConvRot 旋转、int8 量化、
attn 残差 addcmul）各合成一个 kernel，int8 GEMM 本身仍是 omni_xpu_kernel 的 `int8_linear_prequantized`。

| 文件 | 作用 |
|---|---|
| `h3_fused.cpp` | 融合 kernel，注册为 `torch.ops.h3fuse.{norm_rot_quant, rot_quant, silu_mul_rot_quant}` |
| `build.sh` | 容器内执行：编译 `build/h3_fused.so`（源码没变则跳过）+ 打 patch（已打则跳过）；`revert` 撤 patch |
| `patches/0001-omnixpu-h3-fused-block.patch` | ComfyUI-OmniXPU `adapters/h3_rms_modulation.py`：block 满足条件时走融合路径，否则走原路径 |
| `ab_check.py` | 在一次 forward 上用相同输入对比 stock / fused / 纯 torch 三条路径 |

## 使用

`run_headless_minimax_h3.sh` 会先在容器里跑 `optimize/build.sh all`，再跑 headless：

```bash
bash run_headless_minimax_h3.sh                 # 融合路径
H3_FUSED=0 bash run_headless_minimax_h3.sh      # 关掉融合路径（patch 保留），A/B 用
H3_AB_CHECK=1 bash run_headless_minimax_h3.sh   # 打印第 3 次 forward 的三路径输出差异（H3_AB_FORWARD 改次数）
```

撤掉 patch：在容器里执行 `optimize/build.sh revert`。

融合路径的前提（任一不满足就整块走原路径，残差流不会被改动）：四个 projection 都是常驻显存的
TensorWise INT8 + G256 ConvRot 权重、没有 LoRA `weight_function`、Attention/MLP 的 forward 没被替换、有 rope。
日志里出现 `H3 fused block route enabled` 表示 sidecar 已加载；`get_stats()["fused_block"]` 是走融合路径的 block 数。

## 验证结果（B70，video_minimax_h3_t2v2.json，20 步）

- 速度：稳态每次 H3 forward 5.640 s → 5.323 s（−5.6%）。
- trace（第 5 次 forward）：50 个 block 全部走融合路径；`gemm_kernel` 505 → 255 次（旋转 mm 消失），
  norm / quant / silu_mul 独立 kernel 消失，addcmul 300 → 150 次。
- 数值：各融合 kernel 与原链路相比 scale 完全一致，int8 值 99.999% 相同（其余差 1，来自 Hadamard 的 fp32 累加顺序），
  残差流逐位一致，GEMM 输出相对误差 ≤ 2e-4。单个 block 输出相对差 3e-3，与给 stock 路径 qkv 加同样大小（1e-4）
  随机扰动得到的 3.2e-3 相当，是 attention 对微小扰动的固有放大。
- 视频：stock 同 seed 两次运行之间 PSNR 只有 22–24 dB（流水线本身不可逐位复现），fused vs stock 20 dB，画面内容、构图、质量一致。
