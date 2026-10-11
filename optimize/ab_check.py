"""在第 N 次 H3 forward（默认 3，环境变量 H3_AB_FORWARD）上用相同输入跑三条路径并打印输出差异：
stock（OmniXPU 原路径）、fused（h3fuse 融合路径）、torch（未打补丁的 DiTBlock.forward）。
参数与 headless_minimax_h3.py 相同；通过 H3_AB_CHECK=1 bash run_headless_minimax_h3.sh 调用。
"""
import os
import sys

HEADLESS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "headless_minimax_h3.py")

HOOK = '''
    import comfy.ldm.minimax.model as _h3m
    _ab = {"n": 0, "at": int(os.environ.get("H3_AB_FORWARD", "3"))}

    def _clone(t):
        if isinstance(t, torch.Tensor):
            return t.clone()
        if isinstance(t, (list, tuple)):
            return type(t)(_clone(c) for c in t)
        return t

    def _flat(t):
        return [t] if isinstance(t, torch.Tensor) else [y for c in t for y in _flat(c)]

    def _ab_forward(self, x, timestep, context, *forward_args, **forward_kwargs):
        _ab["n"] += 1
        if _ab["n"] == _ab["at"]:
            mod = sys.modules["ComfyUI-OmniXPU.adapters.h3_rms_modulation"]
            run = lambda: _flat(_ab_original(self, _clone(x), _clone(timestep), _clone(context), *forward_args, **forward_kwargs))
            ops = mod._fused_ops
            mod._fused_ops = None
            stock = run()
            patched = _h3m.DiTBlock.forward
            _h3m.DiTBlock.forward = getattr(patched, mod._PATCH_MARKER)
            torch_ref = run()
            _h3m.DiTBlock.forward = patched
            mod._fused_ops = ops
            fused = run() if ops is not None else None
            print("AB_CHECK stats", mod.get_stats(), flush=True)
            for i, s in enumerate(stock):
                rel = lambda a: ((a[i].float() - s.float()).norm() / s.float().norm()).item()
                fused_rel = f"{rel(fused):.3e}" if fused is not None else "n/a (route off)"
                print(f"AB_CHECK out{i} {tuple(s.shape)} fused_vs_stock={fused_rel} torch_vs_stock={rel(torch_ref):.3e}", flush=True)
        return _ab_original(self, x, timestep, context, *forward_args, **forward_kwargs)

    _ab_original = MiniMaxH3Model.forward
    MiniMaxH3Model.forward = _ab_forward
'''

source = open(HEADLESS).read()
anchor = "    original_forward = MiniMaxH3Model.forward\n"
assert anchor in source, "headless_minimax_h3.py changed; update the AB hook anchor"
sys.argv[0] = HEADLESS
sys.path.insert(0, os.path.dirname(HEADLESS))
exec(compile(source.replace(anchor, HOOK + anchor, 1), HEADLESS, "exec"), {"__name__": "__main__", "__file__": HEADLESS})
