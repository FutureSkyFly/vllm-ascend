"""Host-side per-call cost: stock AscendC op vs the branch's custom-op chain.

This is the cost the field report sees as TTFT regression: at concurrency 10 the
decode/prefill batch is above the hybrid thresholds, so every call runs the same
AscendC kernel but pays the extra dispatch layers first.

Measured host-bound: shapes are small enough that the device drains faster than
python can enqueue, and we do NOT synchronize inside the loop.
"""
import os, shutil, sys, time
import torch, torch_npu  # noqa
import vllm_ascend, vllm_ascend.vllm_ascend_C  # noqa

VA = os.path.dirname(vllm_ascend.__file__)
SRC = os.environ.get("V2_SRC", "")
if SRC:
    for f in ("lora_ops.py", "lora_ops_triton.py", "lora_ops_triton_kernels.py",
              "lora_cpp_launcher.cpp",
              "lora_cpp_launcher.cpython-312-aarch64-linux-gnu.so"):
        p = os.path.join(SRC, f)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(VA, "lora", f))
    print("installed tree:", SRC, flush=True)
os.environ.setdefault("TRITON_LORA_CPP", "1")

DEV, DT, R, L = "npu", torch.bfloat16, 16, 2
# Batch ABOVE the hybrid thresholds -> both paths execute the same AscendC op.
B = int(os.environ.get("HOST_B", "16"))
Ho, YHO, OFF, H = 5120, 5120, 0, 5120
N = 2000

torch.manual_seed(0)
idx = torch.zeros(1, device=DEV, dtype=torch.int64)
seq = torch.full((1,), B, device=DEV, dtype=torch.int64)
xs = torch.randn(B, H, device=DEV, dtype=DT) * 0.05
ws = torch.randn(L, 1, R, H, device=DEV, dtype=DT) * 0.05
ys = torch.zeros(B, R, device=DEV, dtype=torch.float32)
xe = torch.randn(B, R, device=DEV, dtype=torch.float32) * 0.05
we = torch.randn(L, 1, Ho, R, device=DEV, dtype=DT) * 0.05
ye = torch.zeros(B, YHO, device=DEV, dtype=DT)


def host_us(fn, n=N):
    for _ in range(20):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    dt = time.perf_counter() - t0          # host enqueue cost only
    torch.npu.synchronize()
    return dt * 1e6 / n


print("B=%d (above hybrid thresholds -> both paths run the same AscendC kernel)" % B,
      flush=True)

# 1) stock: the exact call vllm_ascend/lora/lora_ops.py makes
ws3, we3 = ws.view(L, R, H), we.view(L, Ho, R)
t_stock_s = host_us(lambda: torch.ops._C_ascend.sgmv_shrink(xs, ws3, idx, seq, ys, 1.0))
t_stock_e = host_us(lambda: torch.ops._C_ascend.sgmv_expand(xe, we3, idx, seq, ye, OFF, Ho))
print("  stock  shrink %7.2f us/call | expand %7.2f us/call" % (t_stock_s, t_stock_e), flush=True)

# 2) branch: through lora_ops.py (custom op -> python impl -> wrapper -> AscendC)
from vllm_ascend.lora import lora_ops as LO   # noqa: E402
t_br_s = host_us(lambda: LO.sgmv_shrink(xs, ws, ys, None, seq, idx, B, H, 1, 1.0))
t_br_e = host_us(lambda: LO.sgmv_expand_slice(xe, we, ye, None, seq, idx, B, Ho, 1, OFF, Ho, True))
print("  branch shrink %7.2f us/call | expand %7.2f us/call" % (t_br_s, t_br_e), flush=True)

ds, de = t_br_s - t_stock_s, t_br_e - t_stock_e
print("  OVERHEAD shrink +%.2f us (%.2fx) | expand +%.2f us (%.2fx)"
      % (ds, t_br_s / t_stock_s, de, t_br_e / t_stock_e), flush=True)

# per-forward projection: 48 layers x (4 shrink + 7 expand_slice)
per_fwd = 48 * (4 * ds + 7 * de) / 1000.0
print("  => per forward (48 layers x 4 shrink + 7 expand): +%.1f ms host" % per_fwd, flush=True)
print("DONE", flush=True)
