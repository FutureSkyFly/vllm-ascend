"""Insert an env-gated, one-shot logger for actual_seq_lengths into gdn.py.

The question this settles: which path does the fused prefill op actually take
under a live server? The kernel's cross-sequence packing is gated on `b > 1`
AND 64-alignment of every non-final sequence, and every multi-request operator
number in REPRODUCE.md section 1 was measured on synthetic batches that satisfy
both. Timings alone cannot tell you which path ran; this can.

What it found on 2026-09-07 (TP4, --max-num-batched-tokens 8192, out=1,
concurrency 32, three prompt distributions, 1950 logged calls):

  - b takes the values 1, 4, 5, 8, 9, 10, 12, 14, 18; b > 1 in 85% of calls.
    The operator does see real multi-sequence batches.
  - packed = 0 in 100% of calls, including with --random-range-ratio 0.
  - the reason is the chat template: --random-input-len 1024 arrives as 1035
    tokens, so no sequence is ever 64-aligned and the gate never passes.

So the packed path is unreachable through this serving stack as it stands, and
the -54%/-85% rows describe a regime production never enters.

    python3 patch_gdn.py <gdn.py> apply    # writes gdn.py.gdrbak first
    python3 patch_gdn.py <gdn.py> revert   # restores from gdn.py.gdrbak

Logging is off unless GDR_ASL_PROBE is set to a positive call count (and goes to
$GDR_ASL_LOG, default /tmp/asl_probe.log; deleting that file resets the budget,
which is how one server can be probed across several benchmarks), so the
patched file is behaviourally identical to the original in every other run.
"""
import os
import shutil
import sys

PATH, ACTION = sys.argv[1], sys.argv[2]
BAK = PATH + ".gdrbak"

ANCHOR = "        o, final_state = torch_npu.npu_chunk_gated_delta_rule(\n"

PROBE = '''        import os as _os
        _probe = _os.environ.get("GDR_ASL_PROBE")
        if _probe and _os.environ.get("LOCAL_RANK", "0") == "0":
            import builtins as _b
            _p = _os.environ.get("GDR_ASL_LOG", "/tmp/asl_probe.log")
            if not _os.path.exists(_p):
                _b._gdr_asl_n = 0          # deleting the log resets the budget
            _n = getattr(_b, "_gdr_asl_n", 0)
            if _n < int(_probe):
                _l = actual_seq_lengths.tolist()
                _pk = len(_l) > 1 and all(x % 64 == 0 for x in _l[:-1])
                with open(_p, "a") as _f:
                    _f.write("call%d b=%d T=%d packed=%d lens=%s\\n"
                             % (_n, len(_l), sum(_l), int(_pk), _l[:40]))
                _b._gdr_asl_n = _n + 1
'''

if ACTION == "revert":
    if not os.path.exists(BAK):
        sys.exit("no backup at " + BAK)
    shutil.copyfile(BAK, PATH)
    os.remove(BAK)
    print("REVERTED", PATH)
    sys.exit(0)

src = open(PATH, encoding="utf-8").read()
if "_gdr_asl_n" in src:
    sys.exit("already patched")
if src.count(ANCHOR) != 1:
    sys.exit("anchor found %d times, expected 1" % src.count(ANCHOR))
shutil.copyfile(PATH, BAK)
open(PATH, "w", encoding="utf-8", newline="\n").write(src.replace(ANCHOR, PROBE + ANCHOR, 1))
print("PATCHED", PATH, "backup at", BAK)
