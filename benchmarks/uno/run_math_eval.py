# SPDX-License-Identifier: Apache-2.0
"""Run vLLM Ascend with SGLang UNO's pinned math prompts and timing convention.

Prompt formatting is adapted from SGLang 554f817948c26e8e9c8338b4a33e94a609d6f0fb.
Generation timing excludes startup and includes the complete offline generate call.
Grading is a separate CPU step so the inference runtime need not install math-verify.
"""

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

from benchmarks.uno.sglang_math_data import DATASET_REVISIONS, get_benchmark


def format_prompt(tokenizer, messages, benchmark):
    messages = [dict(message) for message in messages]
    system = next((message for message in messages if message.get("role") == "system"), None)
    if system is None:
        messages.insert(0, {"role": "system", "content": benchmark.instruction})
    elif benchmark.instruction not in system["content"]:
        system["content"] = f"{benchmark.instruction}\n\n{system['content'].strip()}"
    options = dict(add_generation_prompt=True, **benchmark.chat_template_kwargs)
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, **options)
    ids = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, **options)
    return list(ids), str(rendered)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path")
    parser.add_argument("--mode", choices=["ar", "linear", "tree"], required=True)
    parser.add_argument("--benchmark", choices=["gsm8k", "math500", "aime25"], required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-record", type=Path)
    parser.add_argument("--max-running-requests", type=int, required=True)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--context-length", type=int, default=40960)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--config-only", action="store_true")
    args = parser.parse_args()
    if args.mode != "ar" and not args.adapter_path:
        parser.error("--adapter-path is required for UNO")
    if args.mode == "tree" and args.max_running_requests != 1:
        parser.error("Tree requires --max-running-requests 1")
    for name in ("max_running_requests", "max_tokens", "context_length", "max_num_batched_tokens"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.num_samples is None:
        args.num_samples = 10 if args.benchmark == "aime25" else 1
    if args.num_samples <= 0:
        parser.error("--num-samples must be positive")
    if args.output_dir.exists():
        parser.error("--output-dir must be new; preserve previous evidence")
    return args


def main():
    args = parse_args()
    benchmark = get_benchmark(args.benchmark)
    data_path = args.data_root / f"{args.benchmark}.jsonl"
    data = data_path.read_bytes()
    manifests = json.loads(args.data_manifest.read_text())
    data_record = next(entry for entry in manifests if entry["benchmark"] == args.benchmark)
    expected_repo = {"gsm8k": "openai/gsm8k", "math500": "HuggingFaceH4/MATH-500", "aime25": "math-ai/aime25"}[
        args.benchmark
    ]
    assert data_record["repo"] == expected_repo
    assert data_record["revision"] == DATASET_REVISIONS[expected_repo]
    assert hashlib.sha256(data).hexdigest() == data_record["prepared_sha256"]
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    assert len(rows) == benchmark.expected_rows
    rows = rows[: args.limit] if args.limit else rows
    width = {"ar": 1, "linear": 9, "tree": 33}[args.mode]
    sizes = [width * n for n in range(1, args.max_running_requests + 1) if n == 1 or n & (n - 1) == 0]
    if sizes[-1] != width * args.max_running_requests:
        sizes.append(width * args.max_running_requests)
    config = dict(
        model=args.model_path,
        tokenizer=args.model_path,
        dtype="bfloat16",
        seed=args.random_seed,
        tensor_parallel_size=1,
        max_model_len=args.context_length,
        max_num_seqs=args.max_running_requests,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        generation_config="vllm",
        disable_log_stats=False,
    )
    if args.enforce_eager:
        config["enforce_eager"] = True
    else:
        config["compilation_config"] = dict(cudagraph_mode="FULL_DECODE_ONLY", cudagraph_capture_sizes=sizes)
    if args.mode != "ar":
        config["speculative_config"] = dict(method="uno", model=args.adapter_path, num_speculative_tokens=width - 1)
    if args.mode == "tree":
        config["additional_config"] = {"uno_tree": {"draft_width": 16, "candidate_top_k": 32}}
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import EngineArgs

    resolved = EngineArgs(**config).create_engine_config()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    prompts = []
    for index, row in enumerate(rows):
        ids, rendered = format_prompt(tokenizer, row["chat_input"], benchmark)
        assert len(ids) + args.max_tokens + width <= args.context_length, f"Context overflow at row {index}"
        for sample in range(args.num_samples):
            source = row.get("row", index)
            prompts.append(
                dict(
                    id=f"{source}:sample{sample}",
                    source_row=source,
                    sample_index=sample,
                    input_ids=ids,
                    problem=rendered,
                    ground_truth=row["ground_truth"],
                )
            )
    repo = Path(__file__).resolve().parents[2]
    commit_file = repo / "UNO_SOURCE_COMMIT"
    commit = (
        commit_file.read_text().strip()
        if commit_file.exists()
        else subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    )
    runtime_record = None
    if args.runtime_record:
        value = json.loads(args.runtime_record.read_text())
        value = value[0] if isinstance(value, list) else value
        runtime_record = {key: value.get(key) for key in ("Id", "Image", "Name")}
    args.output_dir.mkdir(parents=True)
    sampling = dict(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, max_tokens=args.max_tokens)
    plan = dict(
        mode=args.mode,
        benchmark=args.benchmark,
        config=config,
        resolved_execution=dict(
            async_scheduling=resolved.scheduler_config.async_scheduling,
            enable_prefix_caching=resolved.cache_config.enable_prefix_caching,
            cache_dtype=resolved.cache_config.cache_dtype,
        ),
        sampling=sampling,
        requests=len(prompts),
        dataset=data_record,
        source_commit=commit,
        runtime=runtime_record,
        argv=sys.argv,
        prompt_ids_sha256=hashlib.sha256(json.dumps([p["input_ids"] for p in prompts]).encode()).hexdigest(),
        versions={name: importlib.metadata.version(name) for name in ("vllm", "torch", "torch-npu")},
    )
    (args.output_dir / "plan.json").write_text(json.dumps(plan, indent=2))
    if args.config_only:
        print("MATH_CONFIG_GATE_PASS", args.mode, len(prompts), flush=True)
        return
    engine = LLM(**config)
    try:
        # Match SGLang's complete offline generation interval; initialization
        # above and independent CPU grading below are excluded.
        started = perf_counter()
        outputs = engine.generate(
            [{"prompt_token_ids": p["input_ids"]} for p in prompts], SamplingParams(**sampling), use_tqdm=False
        )
        elapsed = perf_counter() - started
        metrics = engine.get_metrics()
    finally:
        engine.llm_engine.engine_core.shutdown()
    generated = []
    for prompt, output in zip(prompts, outputs, strict=True):
        completion = output.outputs[0]
        generated.append(
            {
                **prompt,
                "output_ids": completion.token_ids,
                "generation": completion.text,
                "num_tokens": len(completion.token_ids),
                "finish_reason": completion.finish_reason,
            }
        )
    with (args.output_dir / "generations.jsonl").open("w", encoding="utf8") as file:
        for row in generated:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    counters = {metric.name: metric.value for metric in metrics if hasattr(metric, "value")}
    total = sum(row["num_tokens"] for row in generated)
    verify = counters.get("vllm:spec_decode_num_drafts", 0)
    summary = dict(
        **plan,
        elapsed_seconds=elapsed,
        output_tokens=total,
        tokens_per_second=total / elapsed,
        tokens_per_second_per_request=total / elapsed / args.max_running_requests,
        tokens_per_forward=total / (2 * verify) if args.mode != "ar" and verify else 1.0 if args.mode == "ar" else None,
        spec_verify_request_iterations=verify,
        counters=counters,
        grading_status="pending",
        limit=args.limit,
        max_tokens=args.max_tokens,
        num_samples=args.num_samples,
        full_dataset=args.limit is None,
        timing="offline generate; startup excluded; no request warmup",
        sampling_semantics="Ascend joint top-k/top-p, identical for AR and UNO",
        tpf_scope="aggregate useful outputs / (2 * request verification iterations); not a speedup",
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "mode",
                    "benchmark",
                    "output_tokens",
                    "tokens_per_second",
                    "tokens_per_forward",
                    "grading_status",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
