#!/usr/bin/env python3
"""给 vllm/v1/core/kv_cache_manager.py 打一个受 VLLM_SCHED_DEBUG 控制的数字探针。

已知：TP4xPP2 下长 prompt 的 allocate_slots 返回 None -> 调度器 break -> 永远排不进。
要知道是两个检查里的哪一个、以及需要多少块 / 有多少空闲块。
幂等，保留 .alloc_orig 备份，可 --revert。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HEAD_ANCHOR = '''class KVCacheManager:
'''

HEAD_NEW = '''import os as _ad_os, time as _ad_time

_AD_ON = bool(int(_ad_os.getenv("VLLM_SCHED_DEBUG", "0")))
_AD_LAST = [0.0]


def _ad(msg, *a):
    if not _AD_ON:
        return
    now = _ad_time.monotonic()
    if now - _AD_LAST[0] < 1.0:
        return
    _AD_LAST[0] = now
    logger.info("ALLOC_DEBUG " + msg, *a)


class KVCacheManager:
'''

FULL_ANCHOR = '''            if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
                return None
'''

FULL_NEW = '''            _ad(
                "fullISL req=%s num_tokens=%s full=%s need=%s free=%s -> %s",
                request.request_id, request.num_tokens, full_num_tokens,
                num_blocks_to_allocate, self.block_pool.get_num_free_blocks(),
                "REJECT" if num_blocks_to_allocate
                > self.block_pool.get_num_free_blocks() else "ok",
            )
            if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
                return None
'''

MAIN_ANCHOR = '''        available_blocks = self.block_pool.get_num_free_blocks() - reserved_blocks
        if num_blocks_to_allocate > available_blocks:
            # Cannot allocate new blocks
            return None
'''

MAIN_NEW = '''        available_blocks = self.block_pool.get_num_free_blocks() - reserved_blocks
        _ad(
            "main req=%s num_tokens=%s need_slot=%s main_model=%s need=%s avail=%s -> %s",
            request.request_id, request.num_tokens, num_tokens_need_slot,
            num_tokens_main_model, num_blocks_to_allocate, available_blocks,
            "REJECT" if num_blocks_to_allocate > available_blocks else "ok",
        )
        if num_blocks_to_allocate > available_blocks:
            # Cannot allocate new blocks
            return None
'''

EDITS = [
    ("module header", HEAD_ANCHOR, HEAD_NEW),
    ("full-ISL check", FULL_ANCHOR, FULL_NEW),
    ("main check", MAIN_ANCHOR, MAIN_NEW),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()
    f = Path(a.file)
    orig = Path(str(f) + ".alloc_orig")
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
            print("FAIL anchor '%s' matched %d (expected 1)" % (name, n), file=sys.stderr)
            rc = 1
            continue
        text = text.replace(anchor, new, 1)
        print("  applied: %s" % name)
    f.write_text(text, encoding="utf-8")
    print("WROTE %s (backup %s)" % (f, orig.name))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
