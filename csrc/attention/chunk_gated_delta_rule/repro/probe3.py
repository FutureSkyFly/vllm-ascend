"""Time one shape, one process, with a controllable actual_seq_lengths pattern.

    GDR_DEV=0 ASL_MODE=even|lastoff|off1|jit python3 probe3.py <T> <B> <nk> <nv>

Why this exists: the shipped kernel's cross-sequence packing is gated on

    packed = b > 1;  for bid < b-1: if (len[bid] % chunkSize) packed = false;

so it only fires when every sequence except the last is a multiple of 64.
probe2.py builds asl as [T//B]*B, which for every shape in the reported table is
a multiple of 64 -- i.e. every published MULTI-REQUEST operator-level number was
measured on the packed path. The four B=1 rows never reach it, since the gate
also needs b > 1. Real traffic has arbitrary lengths.

The four modes hold T, B, nk, nv and every input tensor fixed and move only the
length vector, so the packed gate is the single variable:

  even     [T//B]*B                     -> packed  (the published configuration)
  lastoff  last length -1 (T drops 1)   -> packed  (control: a partial chunk
                                            exists, but the gate ignores the
                                            LAST sequence, so the path is the
                                            same one `even` takes)
  off1     first length -1, second +1   -> NOT packed (gate fails on bid 0);
                                            same T, same B, one token moved
  jit      every length jittered        -> NOT packed; realistic varlen

`off1` vs `lastoff` is the divergence experiment. They are not equal in work:
`lastoff` adds one partial chunk on the final sequence, `off1` adds two (on
sequences 0 and 1) and therefore one extra chunk. What they isolate is the gate
-- `lastoff` keeps packed=1, `off1` drops to 0 -- and the base arm, which has no
packing at all, prices the difference in work at under 2.2%.
"""
import os
import sys
import time

import torch
import torch_npu

MODE = os.environ.get("ASL_MODE", "even")
torch.npu.set_device(int(os.environ.get("GDR_DEV", "0")))
DEV = f"npu:{os.environ.get('GDR_DEV', '0')}"
T, B, nk, nv = (int(x) for x in sys.argv[1:5])
C = 64

base = T // B
lens = [base] * B
lens[-1] = T - base * (B - 1)

if MODE == "even":
    pass
elif MODE == "lastoff":
    # only the FINAL sequence goes off a 64 boundary; the gate ignores it, so
    # `packed` stays true. T drops by one token (0.01-0.04% of the shapes here)
    # -- that is the price of keeping every other length aligned.
    lens[-1] -= 1
elif MODE == "off1":
    if B >= 2:
        lens[0] -= 1
        lens[1] += 1
elif MODE == "jit":
    # deterministic jitter; every non-final length forced off a 64 boundary
    rng = torch.Generator().manual_seed(7)
    span = max(1, min(31, base // 4))
    if B >= 2:
        spent = 0
        for i in range(B - 1):
            d = int(torch.randint(-span, span + 1, (1,), generator=rng).item())
            L = base + d
            if L % C == 0:
                L += 1
            L = max(1, L)
            lens[i] = L
            spent += L
        lens[-1] = T - spent
        if lens[-1] < 1:
            raise SystemExit(f"jit produced non-positive tail {lens[-1]}")
else:
    raise SystemExit(f"unknown ASL_MODE {MODE}")

Teff = sum(lens)
if MODE != "lastoff":
    assert Teff == T, (Teff, T)
packed = B > 1 and all(L % C == 0 for L in lens[:-1])

g_cpu = torch.Generator().manual_seed(1)
asl = torch.tensor(lens, dtype=torch.int32).to(DEV)
q = torch.nn.functional.normalize(
    torch.rand((Teff, nk, 128), generator=g_cpu), p=2, dim=-1).bfloat16().to(DEV)
k = torch.nn.functional.normalize(
    torch.rand((Teff, nk, 128), generator=g_cpu), p=2, dim=-1).bfloat16().to(DEV)
v = torch.rand((Teff, nv, 128), generator=g_cpu).bfloat16().to(DEV)
beta = torch.rand((Teff, nv), generator=g_cpu).bfloat16().to(DEV)
g = (torch.rand((Teff, nv), generator=g_cpu) * -1.0).to(DEV)
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
head = ",".join(str(x) for x in lens[:3])
print(f"T{T}_B{B}_nk{nk}_nv{nv} mode={MODE} packed={int(packed)} Teff={Teff} "
      f"lens[{head},...,{lens[-1]}] {ts[len(ts) // 2]:.1f}")
