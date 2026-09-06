#!/bin/bash
# GLM-5.3-Flash w8a8 on Atlas A2 (8x 910B4-1), TP4 x PP2 + EP.
# Prefix caching + ACL graph 同开；PP 与 MTP 互斥，所以这里没有 --speculative-config。
# 实测记录见 docs/05-pipeline-parallel.md。
#
# 两个前置条件，缺一个都会踩坑：
#
#   1. 先打 patches/pp_support.py。不打的话 PP 根本起不来，而且**没有任何警告**：
#      `SupportsPP` 是 Protocol，`supports_pp()` 假阳性，只有非首 rank 在 _dummy_run 里炸
#      （AttributeError: 'NoneType' object has no attribute 'items'）。
#          python3 patches/pp_support.py --root /vllm-workspace/vllm-ascend/vllm_ascend
#
#   2. 必须显式给 --kv-cache-memory。PP=2 时末段 rank 多带 lm_head，默认只分到 4.85 GiB，
#      块池 690 块 < 准入上限 1035 块，任何超过 ~2.7k token 的 prompt 会被调度器静默拒绝
#      （零输出、零报错，客户端一直等到超时，引擎侧连一条 Engine 000 统计行都不打）。
#      给到 10.09 GiB 后块池 1487 块，实测 100,002 token 的 prompt 正常返回。
#
#      这个值和编译模式绑定，别照抄另一个：
#        图模式（本脚本，无 --enforce-eager）：10831746048 = 10.09 GiB  —— 图捕获另占 0.34 GiB
#        eager（--enforce-eager）           ：11193525248 = 10.42 GiB
#      图模式下误用 eager 的值会在 init_device 阶段直接起不来。换机器或换权重后，
#      以启动日志里 worker.py:758 那行 "to fully utilize NPU free memory" 的建议值为准。
#
# Run INSIDE the container started by docker-run.sh:
#   docker exec glm53s bash -lc "bash /path/to/serve-pp.sh"
#
# NOTE: do not `cd /vllm-workspace` -- the repo checkout there shadows the installed
# vllm package.
MODEL=${MODEL:-/data02/GLM-5.3-Flash-w8a8-b0829}
PORT=${PORT:-8000}
LOG=${LOG:-/tmp/glm53_serve_pp.log}
KV_CACHE_MEMORY=${KV_CACHE_MEMORY:-10831746048}

pkill -f "vllm serve $MODEL" 2>/dev/null
pkill -f "VLLM::" 2>/dev/null          # reparented workers survive the first pkill and hold HBM
sleep 8
pkill -9 -f "VLLM::" 2>/dev/null       # 用 npu-smi 确认显存真的释放了再起下一个
sleep 3

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset ASCEND_LAUNCH_BLOCKING            # incompatible with ACL graph -- vllm-ascend raises
cd /root

nohup setsid vllm serve "$MODEL" \
  --served-model-name glm53 \
  --trust-remote-code \
  --quantization ascend \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 2 \
  --enable-expert-parallel \
  --max-model-len 131072 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.9 \
  --kv-cache-memory="$KV_CACHE_MEMORY" \
  --enable-prefix-caching \
  --prefix-caching-hash-algo xxhash \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 64}' \
  --additional-config '{"enable_cpu_binding": false}' \
  --port "$PORT" > "$LOG" 2>&1 < /dev/null &

echo "launched TP4 x PP2 + graph + APC, kv-cache-memory=$KV_CACHE_MEMORY, log=$LOG"
echo "验收：python3 test/pp_verify.py"
