#!/usr/bin/env python3
"""三轴分离：prompt 长度 × 输出长度 × 采样温度，判定「到底是哪一根轴」。

这是把「PP 长 prompt 停住」这个现象钉死的那个实验。在此之前我先后相信过
「并发才挂」「冷启动第一批才挂」「一个串行请求热身能预防」——全部是错的，
错因都是对照实验一次动了不止一个变量（见 docs/05-pipeline-parallel.md 的证伪表）。

七个格子全部**串行**（conc=1），每格之后打一次短健康探针，
用来发现「引擎被前一格弄坏了」这种串扰。

实测结果（TP4×PP2，默认 KV 显存，未加 --kv-cache-memory）：

    格  prompt  出    temp   TP4×PP2      TP8×PP1
    A   20      16    0      PASS  7.9s   PASS 14.1s
    B   20      300   0      PASS 38.6s   PASS 40.4s
    C   20      300   1.0    PASS 40.3s   PASS 41.7s
    D   2769    16    0      STALL        PASS  6.6s
    E   2769    16    1.0    STALL        PASS  6.3s
    F   2769    300   0      STALL        PASS 115.4s
    G   2769    300   1.0    STALL        PASS 117.1s

    => 只有 prompt 长度这一根轴动。输出长度、温度/top_p 都不是变量。

用法：
    python3 pp_axis_matrix.py
    GLM53_URL=http://host:8000 python3 pp_axis_matrix.py --timeout 300
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

UNIT = "向量化访存的对齐规则决定了搬运效率。"
SHORT = "你好，1+1等于几？"
LONG = "请详细分析第7段。\n" + UNIT * 250  # ~2769 token

T0 = time.time()


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
        return True, "in=%d out=%d t=%.1f" % (
            d["usage"]["prompt_tokens"], d["usage"]["completion_tokens"], time.time() - t0)
    except Exception as e:  # noqa: BLE001
        return False, "t=%.1f %s" % (time.time() - t0, type(e).__name__)


def preflight():
    """连不上就早退。否则「连接被拒」会被误读成「被调度器静默拒绝」。"""
    try:
        with urllib.request.urlopen(BASE + "/v1/models", timeout=10) as r:
            r.read()
    except Exception as e:  # noqa: BLE001
        print("预检失败：连不上 %s（%s）。先起服务，或设 GLM53_URL。"
              % (BASE, type(e).__name__), file=sys.stderr)
        raise SystemExit(2)


CELLS = (
    ("A 短prompt 出16  temp0", SHORT, 16, 0.0),
    ("B 短prompt 出300 temp0", SHORT, 300, 0.0),
    ("C 短prompt 出300 temp1.0", SHORT, 300, 1.0),
    ("D 长prompt 出16  temp0", LONG, 16, 0.0),
    ("E 长prompt 出16  temp1.0", LONG, 16, 1.0),
    ("F 长prompt 出300 temp0", LONG, 300, 0.0),
    ("G 长prompt 出300 temp1.0", LONG, 300, 1.0),
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=200)
    a = ap.parse_args()

    preflight()
    print("=== 三轴分离：prompt长度 × 输出长度 × 温度（全部串行 conc=1）===", flush=True)
    print("=== 每格后打短健康探针，用来发现引擎被前一格弄坏 ===", flush=True)

    result = {}
    for tag, prompt, mt, temp in CELLS:
        ok, msg = ask(prompt, mt, temp, a.timeout)
        result[tag[0]] = ok
        print("[%7.1f] %-40s %s  %s" % (
            time.time() - T0, tag, "PASS " if ok else "STALL", msg), flush=True)
        ok2, msg2 = ask(SHORT, 8, 0.0, 90)
        print("[%7.1f]     健康 after %s   %s %s" % (
            time.time() - T0, tag[0], "OK" if ok2 else "FAIL", msg2), flush=True)

    line = " ".join("%s=%s" % (k, "P" if v else "S") for k, v in result.items())
    print("SUMMARY " + line, flush=True)

    long_stalled = not all(result[c] for c in "DEFG")
    short_ok = all(result[c] for c in "ABC")
    if long_stalled and short_ok:
        print("VERDICT=LENGTH_AXIS  仅 prompt 长度这一轴 -> 加 --kv-cache-memory", flush=True)
        return 1
    if all(result.values()):
        print("VERDICT=ALL_PASS", flush=True)
        return 0
    print("VERDICT=OTHER  与已知指纹不符，别照搬结论，重新分离变量", flush=True)
    return 2


if __name__ == "__main__":
    sys.exit(main())
