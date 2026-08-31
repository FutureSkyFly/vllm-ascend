#!/usr/bin/env bash
set -euo pipefail

# Qwen3.6-35B-A3B (W8A8) with CANN MegaMoe on Atlas A2, 8 x 910B4.
#
# MegaMoe fuses Dispatch + Linear1 + SwiGLU + Linear2 + Combine into a single
# op. It is selected through `enable_fused_mc2` and only for batches whose
# token count is in [mega_moe_min_tokens, mc2_tokens_capacity]; smaller batches
# fall back to the standard A2 MoE path.
#
# Requirements:
#   * cann_ops_transformer installed (provides mega_moe / get_symm_buffer_for_mega_moe)
#   * V1 model runner (VLLM_USE_V2_MODEL_RUNNER=0), data_parallel_size 1
#   * ep_world_size in {2, 4, 8, 16, 32}; here TP8 + EP => 8
#
# Model shape gate for Qwen3.6-35B-A3B (all satisfied):
#   hidden_size            2048   (1024..8192, %512 == 0)
#   moe_intermediate_size   512   ( 512..3072, %512 == 0)
#   num_experts             256   -> 32 experts/rank at EP8 (1..128)
#   num_experts_per_tok       8   (1..16)
#
# Set MEGAMOE=0 to serve the same build with MegaMoe disabled (A/B baseline).

model_path="${MODEL_PATH:-/data1/Qwen3.6-35B-A3B-w8a8}"
host="${HOST:-0.0.0.0}"
port="${PORT:-8077}"
served_model_name="${SERVED_MODEL_NAME:-qwen3.6-35b-a3b-w8a8}"
megamoe="${MEGAMOE:-1}"
max_num_seqs="${MAX_NUM_SEQS:-32}"
max_model_len="${MAX_MODEL_LEN:-32768}"
max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS:-4096}"
mega_moe_min_tokens="${MEGA_MOE_MIN_TOKENS:-512}"

[[ -d "$model_path" ]] || {
    printf 'Model directory does not exist: %s\n' "$model_path" >&2
    exit 1
}
[[ "$port" =~ ^[0-9]+$ ]] || {
    printf 'PORT must be numeric: %s\n' "$port" >&2
    exit 1
}

export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=200
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
# MegaMoe is wired into the V1 model runner only.
export VLLM_USE_V2_MODEL_RUNNER=0

if [[ "$megamoe" == "1" ]]; then
    # enable_fused_mc2=2 is the switch that keeps MegaMoe enabled on main;
    # AscendConfig rewrites it to 1 after recording that MegaMoe is available.
    additional_config="$(printf \
        '{"enable_fused_mc2":2,"mega_moe_min_tokens":%d,"multistream_overlap_shared_expert":false}' \
        "$mega_moe_min_tokens")"
else
    additional_config='{"enable_fused_mc2":0,"multistream_overlap_shared_expert":false}'
fi

exec vllm serve "$model_path" \
    --host "$host" \
    --port "$port" \
    --data-parallel-size 1 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name "$served_model_name" \
    --max-num-seqs "$max_num_seqs" \
    --max-model-len "$max_model_len" \
    --max-num-batched-tokens "$max_num_batched_tokens" \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --quantization ascend \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    --additional-config "$additional_config" \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
