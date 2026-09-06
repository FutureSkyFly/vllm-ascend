"""Bit-exactness harness for chunk_gated_delta_rule.

Runs a fixed set of 18 shapes and saves both output tensors of each to disk.
Run it once against the stock operator and once against the patched one, then
compare the two directories with cmp_gdr.py.

    GDR_OUT=/tmp/gdr GDR_DEV=0 python ab_gdr.py stock
    GDR_OUT=/tmp/gdr GDR_DEV=0 python ab_gdr.py patched
    GDR_OUT=/tmp/gdr python cmp_gdr.py stock patched

Env:
    GDR_DEV   NPU device index (default 0)
    GDR_OUT   directory to save tensors under (default ./gdr_ab)
    SKIP_PERF set to 1 to skip the timing section
"""
import os
import sys
import time
import json

import torch
import torch_npu

torch.npu.set_device(int(os.environ.get("GDR_DEV", "0")))
DEV = f"npu:{os.environ.get('GDR_DEV', '0')}"
TAG = sys.argv[1] if len(sys.argv) > 1 else "run"
OUT = os.path.join(os.environ.get("GDR_OUT", "./gdr_ab"), TAG)
os.makedirs(OUT, exist_ok=True)


def make(T, B, nk, nv, dk=128, dv=128, use_g=True, zero_state=False,
         uneven=False, seed=1234, lens=None):
    g_cpu = torch.Generator().manual_seed(seed)
    if lens is not None:
        Ls = list(lens)
        assert sum(Ls) == T and len(Ls) == B
    elif uneven and B > 1:
        base = T // B
        Ls = [base - 64 * i for i in range(B)]
        Ls[0] += T - sum(Ls)
    else:
        Ls = [T // B] * B
        Ls[-1] += T - sum(Ls)
    assert sum(Ls) == T and all(l > 0 for l in Ls)
    asl = torch.tensor(Ls, dtype=torch.int32).to(DEV)
    q = torch.nn.functional.normalize(
        torch.rand((T, nk, dk), generator=g_cpu), p=2, dim=-1).bfloat16().to(DEV)
    k = torch.nn.functional.normalize(
        torch.rand((T, nk, dk), generator=g_cpu), p=2, dim=-1).bfloat16().to(DEV)
    v = torch.rand((T, nv, dv), generator=g_cpu).bfloat16().to(DEV)
    beta = torch.rand((T, nv), generator=g_cpu).bfloat16().to(DEV)
    g = (torch.rand((T, nv), generator=g_cpu) * -1.0).to(DEV) if use_g else None
    if zero_state:
        state = torch.zeros((B, nv, dv, dk), dtype=torch.bfloat16, device=DEV)
    else:
        state = (torch.rand((B, nv, dv, dk), generator=g_cpu) * 0.5 - 0.25).bfloat16().to(DEV)
    return q, k, v, beta, state, asl, g


def run(T, B, nk, nv, **kw):
    q, k, v, beta, state, asl, g = make(T, B, nk, nv, **kw)
    return torch_npu.npu_chunk_gated_delta_rule(
        q, k, v, beta=beta, initial_state=state,
        actual_seq_lengths=asl, scale=128 ** -0.5, g=g)


# 18 shapes chosen to exercise every branch the patch adds or gates:
# odd/even chunk counts, sequences spanning chunk groups, varlen batches, the
# unaligned-tail and non-packable fallbacks, g=None, zero and random initial
# states, and TP1 head counts.
CORR = [
    ("c_T8192",       dict(T=8192, B=1,  nk=2,  nv=6)),                # 4 groups 40/40/40/8, all even
    ("c_T8128",       dict(T=8128, B=1,  nk=2,  nv=6)),                # last group 7 chunks (odd) -> sameBuf fallback
    ("c_T2624",       dict(T=2624, B=1,  nk=2,  nv=6)),                # group2 = 1 chunk, alias + odd
    ("c_T2560",       dict(T=2560, B=1,  nk=2,  nv=6)),                # exactly one full group
    ("c_T100",        dict(T=100,  B=1,  nk=2,  nv=6)),                # 2 chunks, partial tail
    ("c_T64",         dict(T=64,   B=1,  nk=2,  nv=6)),                # single chunk
    ("c_T8192_B4",    dict(T=8192, B=4,  nk=2,  nv=6, uneven=True)),   # varlen batch
    ("c_T8192_nog",   dict(T=8192, B=1,  nk=2,  nv=6, use_g=False)),   # g=None branch
    ("c_T8192_zs",    dict(T=8192, B=1,  nk=2,  nv=6, zero_state=True)),
    ("c_T5184_B2",    dict(T=5184, B=2,  nk=2,  nv=6)),                # per-seq 2592 = 2560+32
    ("c_T4096_tp1",   dict(T=4096, B=1,  nk=16, nv=48)),               # TP1 head counts
    ("c_T8192_B16",   dict(T=8192, B=16, nk=2,  nv=6)),                # packed: 16 x 512 aligned
    ("c_T9216_B3",    dict(T=9216, B=3,  nk=2,  nv=6)),                # packed: seqs span groups
    ("c_T640_B5",     dict(T=640,  B=5,  nk=2,  nv=6)),                # packed: many short subchains
    ("c_tail3",       dict(T=1100, B=3,  nk=2,  nv=6, lens=[512, 512, 76])),
    ("c_tail2",       dict(T=2660, B=2,  nk=2,  nv=6, lens=[2560, 100])),
    ("c_tail5",       dict(T=5220, B=5,  nk=2,  nv=6,
                           lens=[1280, 1280, 1280, 1280, 100])),
    ("c_mid_unalign", dict(T=1100, B=3,  nk=2,  nv=6, lens=[512, 76, 512])),  # NOT packable
]


def probe_maps():
    """Which opapi / vendor objects did this process actually map?

    This is the routing check: if the custom vendor package is in use, its
    libcust_opapi.so shows up here. Without it you may be timing the stock
    operator and not know.
    """
    hits = []
    with open("/proc/self/maps") as f:
        for ln in f:
            if "vendors" in ln or "opapi" in ln.lower():
                p = ln.split()[-1]
                if p not in hits:
                    hits.append(p)
    return hits


corr_sum = {}
probed = False
for name, kw in CORR:
    o, fs = run(**kw)
    torch.npu.synchronize()
    if not probed:
        probed = True
        print("MAPS_PROBE:", json.dumps(probe_maps()), flush=True)
    torch.save({"o": o.cpu(), "fs": fs.cpu()}, f"{OUT}/{name}.pt")
    corr_sum[name] = {"o_sum": float(o.float().abs().sum().cpu()),
                      "fs_sum": float(fs.float().abs().sum().cpu())}
print(json.dumps(corr_sum, indent=1))

if os.environ.get("SKIP_PERF") == "1":
    print("DONE", TAG, "(correctness only)")
    sys.exit(0)


def timeit(fn, warmup=8, iters=30):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    ts = []
    for _ in range(iters):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    return ts[len(ts) // 2], ts[0]


# NOTE: this in-process sweep is for coarse screening ONLY. Timing several
# shapes in one process inflates the later ones by up to 2.6x. Any number that
# gets reported must come from probe2.py, one process per shape.
PERF = [
    ("p_tp8_8k",     dict(T=8192, B=1,  nk=2,  nv=6)),
    ("p_tp8_8k_b16", dict(T=8192, B=16, nk=2,  nv=6)),
    ("p_tp1_8k",     dict(T=8192, B=1,  nk=16, nv=48)),
    ("p_tp8_2560",   dict(T=2560, B=1,  nk=2,  nv=6)),
]
perf = {}
for name, kw in PERF:
    q, k, v, beta, state, asl, g = make(**kw)
    fn = lambda: torch_npu.npu_chunk_gated_delta_rule(
        q, k, v, beta=beta, initial_state=state,
        actual_seq_lengths=asl, scale=128 ** -0.5, g=g)
    med, mn = timeit(fn)
    perf[name] = {"median_us": round(med, 1), "min_us": round(mn, 1)}
    print(name, perf[name], flush=True)

with open(f"{OUT}/perf.json", "w") as f:
    json.dump({"tag": TAG, "corr": corr_sum, "perf": perf}, f, indent=1)
print("DONE", TAG)
