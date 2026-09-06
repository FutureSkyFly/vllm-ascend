#!/usr/bin/env python3
"""判别实验：强制 PP 下 batch_queue 深度 = 1。

vllm/config/vllm.py:497 的 max_concurrent_batches 直接返回 pipeline_parallel_size，
于是 PP=2 → batch_queue_size=2 → EngineCore 走 step_with_batch_queue。
把它钉成 1，EngineCore 改走同步的 step()，流水重叠没了但功能保留。

用途：判别死锁是否在 batch_queue 路径；同时是候选规避手段。
"""
import argparse, shutil, sys
from pathlib import Path

ANCHOR = '''    def max_concurrent_batches(self) -> int:
        # PP requires PP-size concurrent batches to fill the pipeline.
        # Async scheduling requires 2 concurrent batches to overlap.
        pp_size = self.parallel_config.pipeline_parallel_size
'''

NEW = '''    def max_concurrent_batches(self) -> int:
        # PP requires PP-size concurrent batches to fill the pipeline.
        # Async scheduling requires 2 concurrent batches to overlap.
        pp_size = self.parallel_config.pipeline_parallel_size
        # --- VLLM_ASCEND_PP_DEPTH1 experiment -------------------------------
        # Force depth 1 under PP to test whether the observed concurrency
        # deadlock lives in the step_with_batch_queue path.
        import os as _os
        if _os.environ.get("VLLM_ASCEND_PP_BATCH_DEPTH1") == "1":
            return 1
        # --------------------------------------------------------------------
'''

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/vllm-workspace/vllm/vllm")
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()
    f = Path(a.root) / "config" / "vllm.py"
    orig = Path(str(f) + ".depth1_orig")
    if a.revert:
        if orig.exists():
            shutil.copy2(orig, f); print("REVERTED", f)
        else:
            print("no backup")
        return 0
    if not f.exists():
        print("FAIL missing", f, file=sys.stderr); return 2
    t = f.read_text(encoding="utf-8")
    if "VLLM_ASCEND_PP_BATCH_DEPTH1" in t:
        print("already applied"); return 0
    if t.count(ANCHOR) != 1:
        print(f"FAIL anchor matched {t.count(ANCHOR)} times", file=sys.stderr); return 1
    if not orig.exists():
        shutil.copy2(f, orig)
    f.write_text(t.replace(ANCHOR, NEW, 1), encoding="utf-8")
    print("APPLIED", f, "(backup", orig.name + ")")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
