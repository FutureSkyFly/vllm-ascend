#!/bin/bash
# PP 的正确性判据：抓 prompt_logprobs（纯前向，不涉及采样），两次跑出来的 JSON 逐元素比。
#
# 为什么不看输出文本：迷你模型权重是随机的，PP=1 下本来就是乱码；而且贪心续写会在
# 近平局处分叉（实测 top1−top2 只差 0.003418，ulp 级差异就能翻转）。
#
# 对照必须与被测配置的其他特性完全一致，否则会把图模式/APC 的偏差算到 PP 头上：
#   PP2 eager      vs PP1 eager      -> 0.000e+00   ✅
#   PP2 图+APC     vs PP1 图+APC     -> 0.000e+00   ✅
#   PP2 图+APC     vs PP1 **eager**  -> 3.178e-02   ← 三个变量同变，不能归因给 PP
#
# 用法：
#   起 PP=1 服务 -> bash pp_forward_equivalence.sh /tmp/pp1.json
#   起 PP=2 服务（其余参数完全相同）-> bash pp_forward_equivalence.sh /tmp/pp2.json
#   python3 - <<'EOF'
#   import json; a=json.load(open('/tmp/pp1.json')); b=json.load(open('/tmp/pp2.json'))
#   m=max(abs(x-y) for k in a for x,y in zip(a[k]['prompt_logprobs'], b[k]['prompt_logprobs']))
#   print('max abs diff =', m)
#   EOF
#
# 环境变量：PY（python 解释器）、GLM53_URL、GLM53_MODEL
set -u
OUT=${1:?用法: pp_forward_equivalence.sh <输出 json 路径>}
PY=${PY:-python3}
export GLM53_URL="${GLM53_URL:-http://127.0.0.1:8199}"
export GLM53_MODEL="${GLM53_MODEL:-glm53tiny}"

"$PY" - "$OUT" <<'PYEOF'
import json, os, sys, urllib.request

OUT = sys.argv[1]
BASE = os.environ.get("GLM53_URL", "http://127.0.0.1:8199")
MODEL = os.environ.get("GLM53_MODEL", "glm53tiny")
PROMPTS = ["1 2 3 4", "The capital of France is", "def fibonacci(n):", "机器学习是", "a b c d e f g h"]

res = {}
for p in PROMPTS:
    body = {"model": MODEL, "prompt": p, "max_tokens": 1,
            "temperature": 0, "prompt_logprobs": 0, "logprobs": 5}
    req = urllib.request.Request(BASE + "/v1/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.loads(r.read())
    ch = d["choices"][0]
    vals = []
    for entry in (ch.get("prompt_logprobs") or []):
        if not entry:
            continue
        for _tok_id, info in entry.items():
            vals.append(round(float(info["logprob"]), 6))
    top = ch.get("logprobs") or {}
    res[p] = {
        "prompt_logprobs": vals,
        "first_token_top": top.get("top_logprobs", [{}])[0] if top.get("top_logprobs") else {},
    }

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print("WROTE", OUT, "prompts:", len(res))
PYEOF
