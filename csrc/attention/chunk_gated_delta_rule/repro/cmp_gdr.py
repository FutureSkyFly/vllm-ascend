"""Compare two ab_gdr.py runs tensor by tensor.

    GDR_OUT=/tmp/gdr python cmp_gdr.py stock patched

Prints one line per tensor and a final RESULT line. Both output tensors of the
operator are checked: `o` (the attention output) and `fs` (the final state).
Anything short of "PASS all bit-exact" means the patch changed numerics.
"""
import glob
import os
import sys

import torch

ROOT = os.environ.get("GDR_OUT", "./gdr_ab")
a, b = sys.argv[1], sys.argv[2]
bad = 0
seen = 0

for fa in sorted(glob.glob(os.path.join(ROOT, a, "*.pt"))):
    name = os.path.basename(fa)
    fb = os.path.join(ROOT, b, name)
    if not os.path.exists(fb):
        print(f"MISS     {name}")
        bad += 1
        continue
    da, db = torch.load(fa), torch.load(fb)
    for k in ("o", "fs"):
        seen += 1
        ta, tb = da[k], db[k]
        if torch.equal(ta, tb):
            print(f"BITEXACT {name}:{k}")
        else:
            d = (ta.float() - tb.float()).abs()
            rel = float(d.sum() / (ta.float().abs().sum() + 1e-9))
            print(f"DIFF     {name}:{k} max={float(d.max()):.6g} "
                  f"relsum={rel:.6g} nz={int((d > 0).sum())}/{d.numel()}")
            bad += 1

print(f"COMPARED {seen} tensors")
print("RESULT:", "PASS all bit-exact" if bad == 0 else f"FAIL {bad}")
sys.exit(0 if bad == 0 else 1)
