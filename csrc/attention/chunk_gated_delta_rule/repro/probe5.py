"""Time the operator on a length vector captured from a live vLLM server.

    GDR_DEV=0 python3 probe5.py <nk> <nv> "<l0,l1,...>" [pad]

Everything else in this campaign timed synthetic length vectors -- [T//B]*B, or
a jitter of it. Those are 64-aligned or jittered by hand; neither is what the
scheduler actually produces. These vectors come from an instrumented gdn.py on
a running server (asl_probe logs), so the shape, the batch count and the ragged
tail are the real thing.

With `pad`, every length is rounded UP to a multiple of 64. That makes the
kernel's packing gate pass, so it measures what a working packed path is worth
on this exact batch -- and it is a LOWER bound on a proper kernel fix, because
the fix would pack these chunks without paying for the pad tokens.

Padding is exact for the operator's own semantics only when the pad rows carry
beta = 0 and g = 0 (the recurrence degenerates to S_t = S_{t-1}); this script is
a timing probe and does not slice the output back, so read it as a cost model,
not as a drop-in transform. probe4.py is the one that checks bit-exactness.
"""
import os
import sys
import time

import torch
import torch_npu

torch.npu.set_device(int(os.environ.get("GDR_DEV", "0")))
DEV = f"npu:{os.environ.get('GDR_DEV', '0')}"
C = 64

nk = int(sys.argv[1])
nv = int(sys.argv[2])
lens = [int(x) for x in sys.argv[3].split(",") if x.strip()]
PAD = len(sys.argv) > 4 and sys.argv[4] == "pad"
tag = sys.argv[5] if len(sys.argv) > 5 else "-"

raw_T = sum(lens)
if PAD:
    lens = [((L + C - 1) // C) * C for L in lens]
T = sum(lens)
B = len(lens)
packed = B > 1 and all(L % C == 0 for L in lens[:-1])

g_cpu = torch.Generator().manual_seed(1)
asl = torch.tensor(lens, dtype=torch.int32, device=DEV)
q = torch.nn.functional.normalize(
    torch.rand((T, nk, 128), generator=g_cpu), p=2, dim=-1).bfloat16().to(DEV)
k = torch.nn.functional.normalize(
    torch.rand((T, nk, 128), generator=g_cpu), p=2, dim=-1).bfloat16().to(DEV)
v = torch.rand((T, nv, 128), generator=g_cpu).bfloat16().to(DEV)
beta = torch.rand((T, nv), generator=g_cpu).bfloat16().to(DEV)
g = (torch.rand((T, nv), generator=g_cpu) * -1.0).to(DEV)
st = torch.zeros((B, nv, 128, 128), dtype=torch.bfloat16, device=DEV)

fn = lambda: torch_npu.npu_chunk_gated_delta_rule(
    q, k, v, beta=beta, initial_state=st,
    actual_seq_lengths=asl, scale=128 ** -0.5, g=g)

for _ in range(8):
    fn()
torch.npu.synchronize()

ts = []
for _ in range(30):
    torch.npu.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.npu.synchronize()
    ts.append((time.perf_counter() - t0) * 1e6)
ts.sort()
print(f"{tag} nk{nk}_nv{nv} b={B} T={T} rawT={raw_T} "
      f"pad={'1' if PAD else '0'} packed={int(packed)} "
      f"{ts[len(ts) // 2]:.1f}")
