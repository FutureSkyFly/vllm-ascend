"""Can host-side padding buy back the packing fast path, net of its own cost?

    GDR_DEV=1 python3 probe4.py <T> <B> <nk> <nv>

The kernel only takes its cross-sequence packing path when every sequence except
the last is a multiple of chunkSize=64. Real traffic is not. Instead of
restructuring the kernel, pad each sequence up to a chunk boundary at the call
site and slice the output back.

Padding is EXACT, not an approximation: with beta = 0 and g = 0 the recurrence

    S_t = exp(g_t) * S_{t-1} * (I - beta_t k_t k_t^T) + beta_t v_t k_t^T

collapses to S_t = S_{t-1}, so pad rows cannot touch the carried state, and the
output rows they produce are sliced away. This script checks that claim against
the hardware rather than asserting it, then reports whether the fast path is
worth the copy.

Reported:
  direct   op on the jittered lengths                        (packed = 0)
  padded   scatter into padded buffers + op + gather back    (packed = 1)
  op-only  the same padded call without the copies, to separate kernel time
           from integration cost
"""
import os
import sys
import time

import torch
import torch_npu

torch.npu.set_device(int(os.environ.get("GDR_DEV", "0")))
DEV = f"npu:{os.environ.get('GDR_DEV', '0')}"
T, B, nk, nv = (int(x) for x in sys.argv[1:5])
C = 64
REPS = int(os.environ.get("GDR_REPS", "30"))

# --- jittered lengths, deterministic, same recipe as probe3.py ASL_MODE=jit ---
base = T // B
lens = [base] * B
rng = torch.Generator().manual_seed(7)
span = max(1, min(31, base // 4))
spent = 0
for i in range(B - 1):
    d = int(torch.randint(-span, span + 1, (1,), generator=rng).item())
    L = base + d
    if L % C == 0:
        L += 1
    lens[i] = max(1, L)
    spent += lens[i]
lens[-1] = T - spent
assert lens[-1] >= 1 and sum(lens) == T

plens = [((L + C - 1) // C) * C for L in lens]
Tp = sum(plens)
packed_direct = B > 1 and all(L % C == 0 for L in lens[:-1])
packed_padded = B > 1 and all(L % C == 0 for L in plens[:-1])

# row indices in the padded layout that hold real tokens, in original order
idx = []
off = 0
for L, P in zip(lens, plens):
    idx.extend(range(off, off + L))
    off += P
idx = torch.tensor(idx, dtype=torch.int64, device=DEV)

g_cpu = torch.Generator().manual_seed(1)
q = torch.nn.functional.normalize(
    torch.rand((T, nk, 128), generator=g_cpu), p=2, dim=-1).bfloat16().to(DEV)
k = torch.nn.functional.normalize(
    torch.rand((T, nk, 128), generator=g_cpu), p=2, dim=-1).bfloat16().to(DEV)
v = torch.rand((T, nv, 128), generator=g_cpu).bfloat16().to(DEV)
beta = torch.rand((T, nv), generator=g_cpu).bfloat16().to(DEV)
g = (torch.rand((T, nv), generator=g_cpu) * -1.0).to(DEV)
st = torch.zeros((B, nv, 128, 128), dtype=torch.bfloat16, device=DEV)

asl = torch.tensor(lens, dtype=torch.int32, device=DEV)
aslp = torch.tensor(plens, dtype=torch.int32, device=DEV)

qp = torch.zeros((Tp, nk, 128), dtype=torch.bfloat16, device=DEV)
kp = torch.zeros((Tp, nk, 128), dtype=torch.bfloat16, device=DEV)
vp = torch.zeros((Tp, nv, 128), dtype=torch.bfloat16, device=DEV)
bp = torch.zeros((Tp, nv), dtype=torch.bfloat16, device=DEV)
gp = torch.zeros((Tp, nv), dtype=g.dtype, device=DEV)


def scatter():
    qp.index_copy_(0, idx, q)
    kp.index_copy_(0, idx, k)
    vp.index_copy_(0, idx, v)
    bp.index_copy_(0, idx, beta)
    gp.index_copy_(0, idx, g)


def direct():
    return torch_npu.npu_chunk_gated_delta_rule(
        q, k, v, beta=beta, initial_state=st,
        actual_seq_lengths=asl, scale=128 ** -0.5, g=g)


def padded_op():
    return torch_npu.npu_chunk_gated_delta_rule(
        qp, kp, vp, beta=bp, initial_state=st,
        actual_seq_lengths=aslp, scale=128 ** -0.5, g=gp)


def padded_full():
    scatter()
    o, fs = padded_op()
    return o.index_select(0, idx), fs


# --- correctness: padding must not change a single bit -----------------------
o_d, fs_d = direct()
o_p, fs_p = padded_full()
torch.npu.synchronize()
same_o = bool(torch.equal(o_d, o_p))
same_s = bool(torch.equal(fs_d, fs_p))


def bench(fn):
    for _ in range(8):
        fn()
    torch.npu.synchronize()
    ts = []
    for _ in range(REPS):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    return ts[len(ts) // 2]


t_direct = bench(direct)
t_full = bench(padded_full)
scatter()
t_oponly = bench(padded_op)

print(f"T{T}_B{B}_nk{nk}_nv{nv} Tpad={Tp} (+{100.0 * (Tp - T) / T:.1f}%) "
      f"packed_direct={int(packed_direct)} packed_padded={int(packed_padded)} "
      f"bitexact_out={int(same_o)} bitexact_state={int(same_s)} "
      f"direct={t_direct:.1f} padded_full={t_full:.1f} padded_oponly={t_oponly:.1f} "
      f"net={100.0 * (t_full - t_direct) / t_direct:+.1f}%")
