"""Time one shape, in one process. Median of 30 after 8 warmups.

    GDR_DEV=0 python probe2.py <T> <B> <nk> <nv>

One shape per process is mandatory, not a style preference: timing several
shapes in a single process lets NPU allocator/workspace state carry across
calls, and a large-batch shape measured earlier inflates a small shape measured
later by up to 2.6x (observed: 626us standalone read as 1631us). The inflation
is not uniform either -- the faster, smaller-workspace build is affected less,
so a same-process sweep exaggerates the win. Fork per shape:

    for s in "8192 1 2 4" "8192 16 2 4"; do python probe2.py $s; done
"""
import os
import sys
import time

import torch
import torch_npu

torch.npu.set_device(int(os.environ.get("GDR_DEV", "0")))
DEV = f"npu:{os.environ.get('GDR_DEV', '0')}"
T, B, nk, nv = (int(x) for x in sys.argv[1:5])

g_cpu = torch.Generator().manual_seed(1)
asl = torch.tensor([T // B] * B, dtype=torch.int32).to(DEV)
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
print(f"T{T}_B{B}_nk{nk}_nv{nv} {ts[len(ts) // 2]:.1f}")
