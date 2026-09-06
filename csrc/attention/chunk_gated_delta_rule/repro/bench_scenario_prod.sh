#!/bin/bash
# Production-shaped scenario: the deployment configuration and the two bench
# workloads as they are actually run, wrapped in a four-arm ABBA so the operator
# can be A/B-ed against the CANN built-in.
#
# The server flags and every bench flag below are the deployment script's,
# unchanged. Four deviations, all of them deliberate and all of them recorded
# here rather than silently applied:
#
#  1. MODEL. Measured against the W8A8 checkpoint with `--quantization ascend`.
#     The deployment script points at the unquantized bf16 checkpoint and passes
#     no --quantization flag. The operator itself is bf16 either way, so its
#     absolute time does not change; what changes is the denominator. W8A8 makes
#     the MoE and the projections faster, which *raises* the operator's share of
#     prefill (~16% measured). On bf16 that share is lower, so the end-to-end
#     gain measured here is an upper bound for a bf16 deployment.
#
#  2. --additional-config key placement. fuse_muls_add and enable_npugraph_ex
#     are nested under ascend_compilation_config. Flat at the top level they are
#     rejected outright by an AscendConfig built with pydantic extra="forbid",
#     and silently ignored by versions without it. Both default to true, so the
#     nesting preserves the intent; check which behaviour your build has.
#
#  3. VLLM_USE_V2_MODEL_RUNNER=0 is exported here and is not in the deployment
#     script. It selects the model runner, which is upstream of the GDN dispatch
#     in vllm_ascend/ops/gdn.py. Whether the fused CANN operator is reached at
#     all depends on that dispatch, so verify routing on your own deployment
#     before assuming these numbers transfer -- see the MAPS_PROBE check in
#     REPRODUCE.md, which is the only way to find out that you measured the
#     built-in operator twice.
#
#  4. The NIC pinning (HCCL_IF_IP, {GLOO,TP,HCCL}_SOCKET_IFNAME) and
#     --host 0.0.0.0 are dropped; this runs single-node on loopback. Cards are a
#     topology-aligned group, {0,1,2,3} or {4,5,6,7} -- a set spanning both, such
#     as 1,2,3,4, fails in HCCL init.
#
# Usage:
#   MODEL=/path/to/model CARDS=4,5,6,7 bash bench_scenario_prod.sh 2>&1 | tee prod.log
set -u

MODEL=${MODEL:?set MODEL to the model path}
CARDS=${CARDS:-4,5,6,7}
PORT=${PORT:-8001}
NAME=${NAME:-Qwen36}
VENDOR=${VENDOR:-gdrcust_transformer}
QUANT=${QUANT:---quantization ascend}   # set QUANT="" for a bf16 checkpoint
OUTDIR=${OUTDIR:-./prod_results}
mkdir -p "$OUTDIR"

source /usr/local/Ascend/ascend-toolkit/set_env.sh
V=$ASCEND_OPP_PATH/vendors
CFG=$V/config.ini
export LD_LIBRARY_PATH=$V/$VENDOR/op_api/lib/:${LD_LIBRARY_PATH:-}

unset VLLM_ASCEND_ENABLE_FUSED_MC2
export ASCEND_RT_VISIBLE_DEVICES=$CARDS
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export OMP_NUM_THREADS=100
export TASK_QUEUE_ENABLE=1
export VLLM_RPC_TIMEOUT=300000
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export VLLM_USE_V2_MODEL_RUNNER=0        # deviation 3

start() {
  nohup vllm serve "$MODEL" --served-model-name $NAME --host 127.0.0.1 --port $PORT \
    --tensor-parallel-size 4 --enable-expert-parallel \
    --max-num-seqs 32 --max-model-len 131072 --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.90 --async-scheduling --trust-remote-code \
    $QUANT \
    --enable-prompt-tokens-details --no-enable-prefix-caching \
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --enable-chunked-prefill --mamba-ssm-cache-dtype bfloat16 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_fused_mc2":0,"enable_cpu_binding":true,"multistream_overlap_shared_expert":true,"ascend_compilation_config":{"fuse_muls_add":true,"enable_npugraph_ex":true}}' \
    > "$OUTDIR/server_$1.log" 2>&1 &
  echo $! > /tmp/prod_serve.pid
  for i in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "READY $1 ${i}0s"; return 0; }
    sleep 10
  done
  echo "TIMEOUT $1"
  return 1
}

stop() {
  kill -9 "$(cat /tmp/prod_serve.pid 2>/dev/null)" 2>/dev/null
  pkill -9 -f "vllm serve" 2>/dev/null
  for r in 1 2 3; do
    pkill -9 -f "VLLM::" 2>/dev/null   # reparented workers keep ~56GB/card otherwise
    sleep 10
    [ "$(npu-smi info | sed -n '/Process id/,$p' | grep -cE 'VLLM|python')" = "0" ] && return 0
  done
  echo "WARNING: cards still busy after stop" >&2
}

bench() {  # $1 tag  $2 prefix-len  $3 input-len  $4 seed
  echo "### $1 prefix=$2 input=$3 seed=$4"
  curl -s "http://127.0.0.1:$PORT/metrics" \
    | grep -E 'prefix_cache_(hits|queries)|prompt_tokens_cached' || true
  vllm bench serve --backend openai-chat --model "$MODEL" \
    --base-url "http://127.0.0.1:$PORT" --endpoint /v1/chat/completions \
    --num-prompts 160 --trust-remote-code --dataset-name random --ignore-eos \
    --seed "$4" --served-model-name $NAME \
    --random-prefix-len "$2" --random-input-len "$3" --random-output-len 256 \
    --random-range-ratio 0 --num-warmups 32 --max-concurrency 32 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --save-result --result-filename "$OUTDIR/$1_p$2.json" 2>&1 \
  | grep -E "Successful requests|Benchmark duration|Request throughput|Output token throughput|Total Token throughput|Mean TTFT|Median TTFT|P99 TTFT|Mean TPOT|Median TPOT|Mean E2EL|Median E2EL"
}

R=0
for arm in base patch patch base; do
  if [ "$arm" = "base" ]; then
    sed -i "s/^load_priority=$VENDOR,/load_priority=/" "$CFG"
  else
    grep -q "^load_priority=$VENDOR," "$CFG" || sed -i "s/^load_priority=/load_priority=$VENDOR,/" "$CFG"
  fi
  R=$((R + 1)); T="${arm}_$R"
  echo "===== ARM $T vendor=$(cat "$CFG") ====="
  start "$T" || { stop; continue; }
  bench "$T" 1229 2867 12345
  bench "$T" 4301 1843 54321
  stop
done
sed -i "s/^load_priority=$VENDOR,/load_priority=/" "$CFG"
echo "PROD_SCENARIO_DONE"
echo
echo "Report base as mean(arm1, arm4) and patch as mean(arm2, arm3), per metric."
echo "Check within-arm spread before reading any delta: on the 4301+1843 load the"
echo "median TTFT spread was 24% and the four values fell monotonically with time"
echo "regardless of arm, which is a warmup trend and not an effect. That metric"
echo "was discarded; duration, E2EL and throughput on the same run were usable."
