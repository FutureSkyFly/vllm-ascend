#!/usr/bin/env python3
"""给 vllm/v1/core/sched/scheduler.py 打一个受环境变量控制的调试补丁。

目的：GLM-5.3-Flash 在 TP4xPP2 下，一条 ~2769 token 的 prompt 会让 schedule()
持续返回 0 个 token（引擎在 core.py:1279 的 sleep 里空转），而同一条请求在
TP8xPP1 下 6.6 秒就返回。要知道请求到底卡在哪一个分支。

打上后设 VLLM_SCHED_DEBUG=1 才会输出；限流 1 秒一条，不影响正常运行。
幂等，保留 .sched_orig 备份，可 --revert。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HEADER_ANCHOR = '''        self.use_pp = self.parallel_config.pipeline_parallel_size > 1
'''

HEADER_NEW = '''        self.use_pp = self.parallel_config.pipeline_parallel_size > 1
        # --- SCHED_DEBUG ---
        import os as _os, time as _time
        self._sd_on = bool(int(_os.getenv("VLLM_SCHED_DEBUG", "0")))
        self._sd_last = 0.0

        def _sd(msg, *a):
            if not self._sd_on:
                return
            now = _time.monotonic()
            if now - self._sd_last < 1.0:
                return
            self._sd_last = now
            logger.info("SCHED_DEBUG " + msg, *a)

        self._sd = _sd
        # --- end SCHED_DEBUG ---
'''

# 1) 等待队列：算完 num_new_tokens 之后
WAIT_ANCHOR = '''                    num_new_tokens = min(num_new_tokens, token_budget)
                    assert num_new_tokens > 0
'''

WAIT_NEW = '''                    num_new_tokens = min(num_new_tokens, token_budget)
                    self._sd(
                        "WAIT req=%s num_tokens=%d computed=%d local_cached=%d "
                        "num_new=%d budget=%d block_size=%s mamba_align=%s",
                        request.request_id, request.num_tokens, num_computed_tokens,
                        num_new_local_computed_tokens, num_new_tokens, token_budget,
                        self.cache_config.block_size, self.need_mamba_block_aligned_split,
                    )
                    assert num_new_tokens > 0
'''

# 2) 等待队列：mamba 对齐切分返回 0 -> break
MAMBA_ANCHOR = '''                if self.need_mamba_block_aligned_split and not load_kv_async:
                    num_new_tokens = self._mamba_block_aligned_split(
                        request,
                        num_new_tokens,
                        num_new_local_computed_tokens,
                        num_external_computed_tokens,
                    )
                    if num_new_tokens == 0:
                        break
'''

MAMBA_NEW = '''                if self.need_mamba_block_aligned_split and not load_kv_async:
                    _sd_before = num_new_tokens
                    num_new_tokens = self._mamba_block_aligned_split(
                        request,
                        num_new_tokens,
                        num_new_local_computed_tokens,
                        num_external_computed_tokens,
                    )
                    if num_new_tokens != _sd_before:
                        self._sd(
                            "WAIT mamba_split %d -> %d (block=%s)",
                            _sd_before, num_new_tokens, self.cache_config.block_size,
                        )
                    if num_new_tokens == 0:
                        self._sd("BREAK mamba_align_zero req=%s", request.request_id)
                        break
'''

# 3) 等待队列：allocate_slots 返回 None -> break
ALLOC_ANCHOR = '''                if new_blocks is None:
                    # The request cannot be scheduled.
'''

ALLOC_NEW = '''                if new_blocks is None:
                    self._sd(
                        "BREAK allocate_slots_none req=%s num_new=%d lookahead=%d",
                        request.request_id, num_new_tokens, effective_lookahead_tokens,
                    )
                    # The request cannot be scheduled.
'''

# 4) running 队列：num_new_tokens == 0 -> continue
RUN_ANCHOR = '''            if num_new_tokens == 0:
                # The request cannot be scheduled because one of the following
'''

RUN_NEW = '''            if num_new_tokens == 0:
                self._sd(
                    "RUN skip zero req=%s computed=%d with_spec=%d placeholders=%d "
                    "budget=%d step=%d eligible=%d",
                    request.request_id, request.num_computed_tokens,
                    request.num_tokens_with_spec, request.num_output_placeholders,
                    token_budget, self.current_step,
                    request.next_decode_eligible_step,
                )
                # The request cannot be scheduled because one of the following
'''

# 5) 每次 schedule 的总结
OUT_ANCHOR = '''        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
'''

OUT_NEW = '''        self._sd(
            "OUT total=%d waiting=%d running=%d new=%d cached=%d",
            total_num_scheduled_tokens, len(self.waiting), len(self.running),
            len(new_reqs_data), getattr(cached_reqs_data, "num_reqs", -1),
        )
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
'''

EDITS = [
    ("header", HEADER_ANCHOR, HEADER_NEW),
    ("waiting num_new_tokens", WAIT_ANCHOR, WAIT_NEW),
    ("mamba aligned split", MAMBA_ANCHOR, MAMBA_NEW),
    ("allocate_slots none", ALLOC_ANCHOR, ALLOC_NEW),
    ("running skip zero", RUN_ANCHOR, RUN_NEW),
    ("schedule output summary", OUT_ANCHOR, OUT_NEW),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()
    f = Path(a.file)
    orig = Path(str(f) + ".sched_orig")

    if a.revert:
        if orig.exists():
            shutil.copy2(orig, f)
            print("REVERTED %s" % f)
            return 0
        print("no backup", file=sys.stderr)
        return 1

    if not f.exists():
        print("FAIL missing %s" % f, file=sys.stderr)
        return 2
    text = f.read_text(encoding="utf-8")
    if not orig.exists():
        shutil.copy2(f, orig)
    rc = 0
    for name, anchor, new in EDITS:
        if new in text:
            print("  already: %s" % name)
            continue
        n = text.count(anchor)
        if n != 1:
            print("FAIL anchor '%s' matched %d times (expected 1)" % (name, n), file=sys.stderr)
            rc = 1
            continue
        text = text.replace(anchor, new, 1)
        print("  applied: %s" % name)
    f.write_text(text, encoding="utf-8")
    print("WROTE %s (backup %s)" % (f, orig.name))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
