#!/bin/bash
# GLM-5.3-Flash w8a8 on Atlas A2 (8x 910B4-1), TP8 + EP.
# Prefix caching + ACL graph + MTP speculative decoding, all three on at once.
# Measured working: see docs/02-serving.md.
#
# Run INSIDE the container started by docker-run.sh:
#   docker exec glm53s bash -lc "bash /path/to/serve.sh"
#
# NOTE: do not `cd /vllm-workspace` -- the repo checkout there shadows the installed
# vllm package and you get
#   ImportError: cannot import name 'SamplingParams' from 'vllm' (unknown location)
MODEL=${MODEL:-/data02/GLM-5.3-Flash-w8a8-b0829}
PORT=${PORT:-8000}
LOG=${LOG:-/tmp/glm53_serve.log}

pkill -f "vllm serve $MODEL" 2>/dev/null
pkill -f "VLLM::" 2>/dev/null          # reparented workers survive the first pkill and hold HBM
sleep 8

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset ASCEND_LAUNCH_BLOCKING            # incompatible with ACL graph -- vllm-ascend raises
cd /root

nohup setsid vllm serve "$MODEL" \
  --served-model-name glm53 \
  --trust-remote-code \
  --quantization ascend \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --max-model-len 16384 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.9 \
  `# max-model-len 16384 时不会碰到下面这个天花板，但把它调大就要留意：`\
  `# TP8×PP1 默认块池 1178 块，实测 40,012 token 的 prompt 正常，`\
  `# 100,002 token 会被调度器静默拒绝（need=1186 > free=1178），零输出零报错。`\
  `# 所以 max-model-len 抬到 ~40k 以上时，同样要显式加 --kv-cache-memory`\
  `# （值取启动日志 worker.py:758 的建议）。该路径未实测 [U]。`\
  `# 根因（indexer 状态缓存块大小错账）见 docs/05-pipeline-parallel.md。`\
  --enable-prefix-caching \
  --prefix-caching-hash-algo xxhash \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 64}' \
  --additional-config '{"enable_cpu_binding": false}' \
  --port "$PORT" > "$LOG" 2>&1 < /dev/null &

echo "launched, log=$LOG"
