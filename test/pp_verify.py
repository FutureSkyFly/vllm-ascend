#!/usr/bin/env python3
"""PP 验收：一次跑完正确性 + 并发 + 长度 + 长上下文生成。

对着 `serve/serve-pp.sh` 起的服务跑（TP4×PP2 + 图模式 + prefix cache +
`--kv-cache-memory`）。全过才算 PP 可用。

实测基线（2026-09-06，8×910B4，w8a8 b0829）：

    KV                1,072,407 token（Maximum concurrency 8.18x）
    正确性            17×23 → 391 ✓        capital of France → Paris ✓
    并发              conc=4 × ~3000tok × 300out × temp1.0   4/4  20.0 s
                      conc=8 × ~7000tok × 600out × temp1.0   8/8  55.3 s
    长度              3,010 ✓1.6s  8,008 ✓2.6s  20,006 ✓5.1s
                      60,004 ✓14.5s  100,002 ✓25.5s
    长上下文生成      in=17,633 → out=120，9.5 s
    prefill 吞吐      11,762 tokens/s        Traceback 0 条

不加 `--kv-cache-memory` 时，上面第 2、3 项会**静默停住**（零输出、零报错，
客户端等到超时），根因见 docs/05-pipeline-parallel.md。

用法：
    python3 pp_verify.py
    GLM53_URL=http://host:8000 python3 pp_verify.py
退出码 0 = 全过。
"""
import json
import os
import sys
import threading
import time
import urllib.request

BASE = os.environ.get("GLM53_URL", "http://127.0.0.1:8000")
URL = BASE + "/v1/chat/completions"
MODEL = os.environ.get("GLM53_MODEL", "glm53")

UNIT = "向量化访存的对齐规则决定了搬运效率。"
T0 = time.time()
FAILS = []


def stamp():
    return "%7.1f" % (time.time() - T0)


def ask(prompt, max_tokens, temp, timeout):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temp,
    }
    if temp > 0:
        body["top_p"] = 0.95
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
        return (True, d["usage"]["prompt_tokens"], d["usage"]["completion_tokens"],
                time.time() - t0, d["choices"][0]["message"]["content"])
    except Exception as e:  # noqa: BLE001
        return False, -1, -1, time.time() - t0, type(e).__name__


def concurrent(tag, prompts, max_tokens, temp, timeout):
    res = {}
    lock = threading.Lock()

    def work(i):
        ok, _, out, dt, _ = ask(prompts[i], max_tokens, temp, timeout)
        with lock:
            res[i] = (ok, out, dt)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(len(prompts))]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ok = sum(1 for v in res.values() if v[0])
    print("[%s] %-40s %d/%d wall=%6.1fs | %s" % (
        stamp(), tag, ok, len(prompts), time.time() - t0,
        " ; ".join("%d:%s%d/%.0fs" % (i, "OK" if res[i][0] else "XX", res[i][1], res[i][2])
                   for i in sorted(res))), flush=True)
    if ok != len(prompts):
        FAILS.append(tag)


def preflight():
    """连不上就早退。否则「连接被拒」会被误读成「被调度器静默拒绝」。"""
    try:
        with urllib.request.urlopen(BASE + "/v1/models", timeout=10) as r:
            r.read()
    except Exception as e:  # noqa: BLE001
        print("预检失败：连不上 %s（%s）。先起服务，或设 GLM53_URL。"
              % (BASE, type(e).__name__), file=sys.stderr)
        raise SystemExit(2)


def distinct(i, approx_tokens):
    return "请详细分析第%d段。\n" % i + UNIT * max(1, approx_tokens // 12)


def main() -> int:
    preflight()
    print("=== PP 验收：TP4×PP2 + 图模式 + APC + kv-cache-memory ===", flush=True)

    print("--- 1) 正确性 ---", flush=True)
    for q, want in (("17乘以23等于多少？只回答数字。", "391"),
                    ("What is the capital of France? Answer with one word.", "Paris")):
        ok, _, _, dt, txt = ask(q, 64, 0.0, 180)
        hit = ok and want.lower() in (txt or "").lower()
        print("[%s]   %-45s %s  t=%.1fs  %r" % (
            stamp(), q[:42], "PASS" if hit else "FAIL", dt, (txt or "")[-60:]), flush=True)
        if not hit:
            FAILS.append("correctness:" + want)

    print("--- 2) 并发（不加 --kv-cache-memory 时这两行必挂）---", flush=True)
    concurrent("conc=4 x ~3000tok x 300out temp1.0",
               [distinct(i, 3000) for i in range(4)], 300, 1.0, 600)
    concurrent("conc=8 x ~7000tok x 600out temp1.0",
               [distinct(i + 10, 7000) for i in range(8)], 600, 1.0, 900)

    print("--- 3) 长度阶梯 ---", flush=True)
    for k, want in enumerate((3000, 8000, 20000, 60000, 100000)):
        unit = "第%d类向量化访存的对齐规则决定了搬运效率。" % (k + 200)
        ok, pin, _, dt, _ = ask("分析。\n" + unit * max(1, want // 14), 8, 0.0, 300)
        print("[%s]   want=%-7d in=%-7s %s t=%6.1fs" % (
            stamp(), want, pin if pin > 0 else "?", "PASS " if ok else "FAIL ", dt), flush=True)
        if not ok:
            FAILS.append("len:%d" % want)

    print("--- 4) 长上下文真实生成 ---", flush=True)
    ctx = ("下面是一段重复文本，请在读完后回答：这段文本反复强调的是什么？用一句话回答。\n"
           + UNIT * 1600)
    ok, pin, pout, dt, txt = ask(ctx, 200, 0.0, 600)
    print("[%s]   in=%s out=%s t=%.1fs  %r" % (
        stamp(), pin, pout, dt, (txt or "")[:90]), flush=True)
    if not ok:
        FAILS.append("longctx_gen")

    print("=== 结果 ===", flush=True)
    print("VERIFY=%s  失败项=%s" % ("ALL_PASS" if not FAILS else "FAILED", FAILS or "无"),
          flush=True)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
