#!/usr/bin/env python3
"""按 prompt 长度扫描，定位「多长的 prompt 会被调度器静默拒绝」。

背景见 docs/05-pipeline-parallel.md。GLM-5.3-Flash 上，indexer 压缩器状态缓存把自己的
KV 块大小报成 `compress_ratio`（4 个 token），于是长 prompt 会耗尽共享块池，
`allocate_slots` 返回 None，调度器 break，请求**永远排不进去且没有任何报错**：
客户端一直等到超时，引擎侧连一条 `Engine 000` 统计行都不会打。

**关键：每个长度用不同的重复单元，任何两条 prompt 不共享哪怕一个块。**
如果所有 prompt 共享前缀，后面的会命中 prefix cache 只做增量 prefill，
扫出来的阈值是假的（我第一版就是这么骗过自己的）。

用法：
    python3 pp_length_ladder.py                 # 粗扫 100..4000
    python3 pp_length_ladder.py --range fine    # 细扫 2200..3300，卡阈值
    python3 pp_length_ladder.py --range high    # 高区 2k..100k，找配置上限
    python3 pp_length_ladder.py --lengths 1000,5000,20000
    GLM53_URL=http://host:8000 python3 pp_length_ladder.py

判读：
    时间应当随长度平滑增长（1k→100k 约 0.5s→25s）。
    出现**断崖**（前一档 ~1s、下一档直接超时）就是被 allocate_slots 拒了；
    此时「卡后短探针」仍然 OK，说明引擎没死——这正是该缺陷的指纹。
    修法：启动时加 --kv-cache-memory（图模式 10831746048 / eager 11193525248）。
"""
import argparse
import json
import os
import sys
import time
import urllib.request

BASE = os.environ.get("GLM53_URL", "http://127.0.0.1:8000")
URL = BASE + "/v1/chat/completions"
MODEL = os.environ.get("GLM53_MODEL", "glm53")

RANGES = {
    "coarse": (100, 300, 600, 900, 1150, 1400, 1800, 2300, 2800, 3200, 4000),
    "fine": (2200, 2350, 2450, 2550, 2650, 2750, 2850, 2950, 3300),
    "high": (2000, 4000, 5000, 6000, 8000, 12000, 20000, 40000, 100000),
}

T0 = time.time()


def mk(k, want_tokens):
    """每个长度一个独有的重复单元 -> 彻底断开 prefix cache 共享。"""
    unit = "第%d类向量化访存的对齐规则决定了搬运效率。" % k
    return "分析。\n" + unit * max(1, want_tokens // 14)


def ask(prompt, max_tokens, timeout):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
        return True, d["usage"]["prompt_tokens"], time.time() - t0, "OK"
    except Exception as e:  # noqa: BLE001 - 超时/HTTP 错误都要区分出来
        return False, -1, time.time() - t0, type(e).__name__


def preflight():
    """连不上就早退。否则「连接被拒」会被误读成「被调度器静默拒绝」。"""
    try:
        with urllib.request.urlopen(BASE + "/v1/models", timeout=10) as r:
            r.read()
    except Exception as e:  # noqa: BLE001
        print("预检失败：连不上 %s（%s）。先起服务，或设 GLM53_URL。"
              % (BASE, type(e).__name__), file=sys.stderr)
        raise SystemExit(2)


def is_stall(why):
    """只有超时才算「被调度器静默拒绝」；连接类错误是服务没了。"""
    return why in ("TimeoutError", "timeout", "socket.timeout")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--range", choices=sorted(RANGES), default="coarse")
    ap.add_argument("--lengths", help="逗号分隔的 token 数，覆盖 --range")
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--max-tokens", type=int, default=1,
                    help="默认只出 1 个 token —— 这里量的是 prefill 能不能排进去，不是生成速度")
    a = ap.parse_args()

    lengths = ([int(x) for x in a.lengths.split(",")] if a.lengths
               else list(RANGES[a.range]))
    preflight()

    print("=== prompt 长度阶梯（无前缀共享，串行，temp 0）===", flush=True)
    print("=== 断崖 = 被 allocate_slots 静默拒绝；平滑 = 正常 ===", flush=True)

    first_stall = None
    for k, want in enumerate(lengths):
        ok, ntok, dt, why = ask(mk(k + 1, want), a.max_tokens, a.timeout)
        print("[%7.1f] want=%-7d in=%-7s %s t=%7.2fs %s" % (
            time.time() - T0, want, ntok if ntok > 0 else "?",
            "PASS " if ok else "FAIL ", dt, why), flush=True)
        if not ok:
            if not is_stall(why):
                print("  ^ 这不是「被拒」，是连接层错误（%s）——服务没了，别照 stall 解读。"
                      % why, file=sys.stderr)
                return 2
            ok2, _, t2, _ = ask("你好", 4, 60)
            print("[%7.1f]    卡后短探针 %s t=%.1f -> 引擎%s" % (
                time.time() - T0, "OK" if ok2 else "FAIL", t2,
                "健康（是调度器拒了这条，不是崩了）" if ok2 else "真的坏了"), flush=True)
            if first_stall is None:
                first_stall = want

    if first_stall is None:
        print("RESULT=ALL_PASS  最长 %d token 通过" % lengths[-1], flush=True)
        return 0
    print("RESULT=FIRST_STALL_AT=%d  -> 加 --kv-cache-memory 重试" % first_stall, flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
