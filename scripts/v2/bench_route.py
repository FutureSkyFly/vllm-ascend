"""Correctness through the PUBLIC lora_ops.py entry point (what punica calls),
covering both routes of the hoisted dispatch:
  B <= threshold -> triton custom op
  B >  threshold -> direct torch.ops._C_ascend.*   (the new fast path)
Reference is the stock AscendC op called exactly as stock lora_ops.py calls it.
Requires bitwise equality on both routes.
"""
import importlib
import os
import shutil
import sys

import torch
import torch_npu  # noqa: F401
import vllm_ascend
import vllm_ascend.vllm_ascend_C  # noqa: F401

VA = os.path.dirname(vllm_ascend.__file__)
SRC = os.environ.get("V2_SRC", "/work/v2hoist")
for f in ("lora_ops.py", "lora_ops_triton.py", "lora_ops_triton_kernels.py",
          "lora_cpp_launcher.cpp",
          "lora_cpp_launcher.cpython-312-aarch64-linux-gnu.so"):
    p = os.path.join(SRC, f)
    if os.path.exists(p):
        shutil.copy(p, os.path.join(VA, "lora", f))
print("installed:", SRC, flush=True)
os.environ.setdefault("TRITON_LORA_CPP", "1")
os.environ.setdefault("TRITON_LORA_EXACT", "1")

from vllm_ascend.lora import lora_ops as LO  # noqa: E402

DEV, DT, R, L = "npu", torch.bfloat16, 16, 2
FAILS = []


def check(tag, got, ref):
    eq = torch.equal(got, ref)
    n = 0 if eq else int((got != ref).sum())
    d = 0.0 if eq else float((got.float() - ref.float()).abs().max())
    if not eq:
        FAILS.append(tag)
    print("  %-46s bitexact=%-5s ndiff=%-8d max_abs=%.3e" % (tag, eq, n, d),
          flush=True)


def shrink_case(H, B, NR):
    idx = torch.arange(NR, device=DEV, dtype=torch.int64) % L
    seq = torch.full((NR,), B // NR, device=DEV, dtype=torch.int64)
    seq[-1] += B - (B // NR) * NR
    torch.manual_seed(hash(("s", H, B, NR)) % 2**31)
    x = torch.randn(B, H, device=DEV, dtype=DT) * 0.05
    w = torch.randn(L, 1, R, H, device=DEV, dtype=DT) * 0.05
    y0 = torch.randn(B, R, device=DEV, dtype=torch.float32)
    # reference: exactly what stock vllm_ascend/lora/lora_ops.py issues
    y = y0.clone()
    torch.ops._C_ascend.sgmv_shrink(x, w, idx, seq, y, 0.5)
    torch.npu.synchronize()
    ref = y.clone()
    # through the branch's public entry point
    y = y0.clone()
    LO.sgmv_shrink(x, w, y, None, seq, idx, B, H, NR, 0.5)
    torch.npu.synchronize()
    route = "triton" if B <= int(os.environ.get("TRITON_LORA_SHRINK_MAX_B", "2")) else "ascendc"
    check("shrink H=%-6d B=%-5d NR=%d [%s]" % (H, B, NR, route), y, ref)


def expand_case(Ho, YHO, OFF, B, NR):
    idx = torch.arange(NR, device=DEV, dtype=torch.int64) % L
    seq = torch.full((NR,), B // NR, device=DEV, dtype=torch.int64)
    seq[-1] += B - (B // NR) * NR
    torch.manual_seed(hash(("e", Ho, B, NR)) % 2**31)
    xe = torch.randn(B, R, device=DEV, dtype=torch.float32) * 0.05
    we = torch.randn(L, 1, Ho, R, device=DEV, dtype=DT) * 0.05
    y0 = torch.randn(B, YHO, device=DEV, dtype=DT) * 0.05
    y = y0.clone()
    torch.ops._C_ascend.sgmv_expand(xe, we, idx, seq, y, OFF, Ho)
    torch.npu.synchronize()
    ref = y.clone()
    y = y0.clone()
    LO.sgmv_expand_slice(xe, we, y, None, seq, idx, B, Ho, NR, OFF, Ho, True)
    torch.npu.synchronize()
    route = "triton" if B <= int(os.environ.get("TRITON_LORA_EXPAND_MAX_B", "1")) else "ascendc"
    check("expand Ho=%-6d B=%-5d NR=%d [%s]" % (Ho, B, NR, route), y, ref)


SHR = (5120, 6144, 17408)
EXP = [(6144, 8192, 0), (1024, 8192, 6144), (5120, 5120, 0), (17408, 34816, 17408)]

print("=== both routes through lora_ops.py (public entry) ===", flush=True)
for B in (1, 2, 3, 4, 8, 16, 64, 256):
    for NR in (1, 2):
        if NR > B:
            continue
        for H in SHR:
            shrink_case(H, B, NR)
        for Ho, YHO, OFF in EXP:
            expand_case(Ho, YHO, OFF, B, NR)

print("", flush=True)
print("=== kill switch: TRITON_LORA_DISABLE=1 must be pure stock ===", flush=True)
os.environ["TRITON_LORA_DISABLE"] = "1"
importlib.reload(LO)
for B in (1, 2, 4):
    shrink_case(5120, B, 1)
    expand_case(5120, 5120, 0, B, 1)

print("", flush=True)
print("TOTAL FAILS: %d" % len(FAILS), flush=True)
for f in FAILS[:20]:
    print("  " + f, flush=True)
print("DONE", flush=True)
