#!/bin/bash
# End-to-end A/B with `vllm bench serve`, four arms: base, patch, patch, base.
# Each arm starts a fresh server and benches it twice, giving eight cells.
#
# Two things this guards against, both of which produced wrong numbers here
# before they were added:
#
#  1. Server instances differ. Benching twice inside one instance separates
#     within-instance repeatability (observed <0.05% on median TTFT) from
#     between-instance variation, so you can tell whether an arm-to-arm gap is
#     real. Report pass1-vs-pass1 and pass2-vs-pass2: there is a ~3.5% warmup
#     effect from pass1 to pass2 that is present in both arms.
#
#  2. `pkill -f "vllm serve"` does not stop the workers. VLLM::EngineCore and
#     VLLM::Worker_TPn get reparented and keep ~56GB of HBM per card, so the
#     next arm fails to start -- and the failure does not say it is a memory
#     problem. stop_server kills "VLLM::" as a separate step and the criterion
#     is npu-smi showing no processes, not the process table.
#
# Usage:
#   MODEL=/path/to/Qwen3.6-35B-A3B-w8a8 CARDS=4,5,6,7 bash bench_e2e_abba.sh
set -u

MODEL=${MODEL:?set MODEL to the model path}
CARDS=${CARDS:-0,1,2,3}
PORT=${PORT:-8091}
NAME=${NAME:-q36}
VENDOR=${VENDOR:-gdrcust_transformer}
OUTDIR=${OUTDIR:-./e2e_results}
mkdir -p "$OUTDIR"

source /usr/local/Ascend/ascend-toolkit/set_env.sh
V=$ASCEND_OPP_PATH/vendors
CFG=$V/config.ini
export LD_LIBRARY_PATH=$V/$VENDOR/op_api/lib/:${LD_LIBRARY_PATH:-}

# ASCEND_RT_VISIBLE_DEVICES must be a topology-aligned group: {0,1,2,3} or
# {4,5,6,7}. A set spanning both groups (e.g. 1,2,3,4) fails in HCCL init.
export ASCEND_RT_VISIBLE_DEVICES=$CARDS
export HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=200
export OMP_PROC_BIND=false OMP_NUM_THREADS=1 TASK_QUEUE_ENABLE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_BALANCE_SCHEDULING=1 VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_USE_V2_MODEL_RUNNER=0

start_server() {
  nohup vllm serve "$MODEL" --host 127.0.0.1 --port $PORT --served-model-name $NAME \
    --data-parallel-size 1 --tensor-parallel-size 4 --enable-expert-parallel --seed 1024 \
    --max-num-seqs 64 --max-model-len 16384 --max-num-batched-tokens 8192 \
    --trust-remote-code --gpu-memory-utilization 0.90 --quantization ascend \
    --enable-chunked-prefill --no-enable-prefix-caching \
    --additional-config '{"enable_fused_mc2":0,"multistream_overlap_shared_expert":false}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    > "$OUTDIR/server_$1.log" 2>&1 &
  echo $! > /tmp/e2e_serve.pid
  for i in $(seq 1 90); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "SERVER_READY $1 ${i}0s"; return 0; }
    sleep 10
  done
  echo "SERVER_TIMEOUT $1"
  return 1
}

stop_server() {
  kill -9 "$(cat /tmp/e2e_serve.pid 2>/dev/null)" 2>/dev/null
  pkill -9 -f "vllm serve" 2>/dev/null
  for r in 1 2 3; do
    pkill -9 -f "VLLM::" 2>/dev/null       # reparented workers; see header note 2
    sleep 10
    [ "$(npu-smi info | sed -n '/Process id/,$p' | grep -cE 'VLLM|python')" = "0" ] && return 0
  done
  echo "WARNING: cards still busy after stop_server" >&2
}

runbench() {
  for k in 1 2; do
    echo "### BENCH $1 pass$k  in=1024 out=1 n=128 conc=32"
    vllm bench serve --backend vllm --base-url "http://127.0.0.1:$PORT" --model "$MODEL" \
      --served-model-name $NAME --dataset-name random \
      --random-input-len 1024 --random-output-len 1 --num-prompts 128 --max-concurrency 32 \
      --ignore-eos --percentile-metrics ttft,tpot,itl,e2el \
      --save-result --result-filename "$OUTDIR/$1_p$k.json" 2>&1 \
      | grep -E "Successful requests|Benchmark duration|Request throughput|Total Token throughput|Mean TTFT|Median TTFT|P99 TTFT|Mean E2EL|Median E2EL"
  done
}

R=0
for arm in base patch patch base; do
  if [ "$arm" = "base" ]; then
    sed -i "s/^load_priority=.*/load_priority=/" "$CFG"   # whole line: no trailing comma on a single-vendor install
  else
    grep -q "^load_priority=$VENDOR" "$CFG" || sed -i "s/^load_priority=/load_priority=$VENDOR,/" "$CFG"
  fi
  R=$((R + 1)); TAG="${arm}_$R"
  echo "===== ARM $TAG vendor=$(cat "$CFG") ====="
  start_server "$TAG" || { stop_server; continue; }
  runbench "$TAG"
  stop_server
done
sed -i "s/^load_priority=.*/load_priority=/" "$CFG"   # whole line: no trailing comma on a single-vendor install
echo "DONE"
