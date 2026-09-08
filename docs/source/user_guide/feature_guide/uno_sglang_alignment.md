# UNO Linear / Tree：SGLang 性能口径对齐

本页记录 2026-09-08 的 Ascend 910B 单卡实现与验证。开发分支为 `codex/uno-sglang-alignment`，基于已发布的 `uno-spec-decode` 分支 `eb834d8336871404554e6e676ca5256534d967ca`。旧版草稿图与 eager 草稿的 4.38× 对照见 [原复现记录](uno_reproduction.md)；该数字不是相对 AR 的收益。

## 对齐对象和指标

依据 [SGLang 已合并 UNO PR 37667](https://github.com/sgl-project/sglang/pull/37667) 和 [固定版本的数学评测说明](https://github.com/sgl-project/sglang/blob/554f817948c26e8e9c8338b4a33e94a609d6f0fb/benchmark/uno/README.md)。实现参考已合并提交 `2bb25dc18bc27321fb116bbefcaddaf13c1c1754` 及 adapter 子目录修复 `516cfbd36259e893faee202840f14968c58f8d4f`，没有导入尚未合并的 NPU Linear PR 37906。

| 项目 | Linear | Tree |
|---|---|---|
| 草稿宽度 / 候选 top-k / 验证节点预算（B/K/V） | 8/1/8 | 16/32/32 |
| 并发上限 | 64 | 1 |
| 对照 | 同配置 AR C64 | 同配置 AR C1 |
| 图模式 | FULL_DECODE_ONLY，包含独立草稿图 | FULL_DECODE_ONLY，包含独立草稿图 |

共同参数：BF16、TP=1、上下文 40960、输出上限 32768、temperature=1、top-k=50、top-p=0.95、seed=42。GSM8K 为 1319 题 × 1，MATH500 为 500 题 × 1，AIME25 为 30 题 × 10。使用同一数学指令、reasoning chat template 和 `math_verify` 评分。

计时覆盖整个离线 `generate` 调用，包括 prefill；排除引擎启动和图捕获，不额外执行请求预热。输出吞吐为实际输出 token 总数除以计时秒数，收益为 UNO 输出吞吐除以相应 AR 输出吞吐。`tok/s/request` 是总吞吐除以名义并发上限，不是实测 TPOT。TPF 按两个 UNO target-model forward 计数，不是 speedup；本实现报告 `输出 token / (2 × 请求验证轮数)` 的聚合值。

SGLang PR 的 H200 公布值如下，仅作为目标参考：

| 数据集 | Linear C64 / AR C64 | Tree C1 / AR C1 |
|---|---:|---:|
| GSM8K | 1.1808× | 2.4750× |
| MATH500 | 1.7578× | 2.7772× |
| AIME25 | 2.0273× | 2.7404× |

Ascend 与 H200 的绝对 token/s 不能直接相比。还有两项残余差异必须保留在结论中：本地 AR/UNO 均使用 Ascend joint top-k/top-p，而上述 SGLang UNO 先 top-k 再在保留集合上计算 top-p；本地配对实验均关闭 prefix cache，Tree 关闭 async scheduling。相同参数字符串不表示跨框架采样和调度行为完全相同。

## 实现及已验证范围

Linear 保留 F=8 的独立草稿图。Tree 新增完整的 best-first 构树、祖先 attention mask、目标采样后的分支遍历和接受路径 KV 压紧；验证侧使用 paged BSH FIA-v2，并为每个图尺寸复用一份 workspace。

Tree 当前只支持 Qwen3ForCausalLM（包括 SDAR alias）、TP=1、并发 1、单一 GQA KV cache group、非量化 KV、无滑窗、关闭 prefix cache、standard target sampling。允许 temperature/top-k/top-p，暂不支持 penalties、logprobs、structured outputs、thinking budget 等 logits processors。草稿宽度为 2–16，候选数为 2–128，验证节点预算介于草稿宽度与 32 之间。

已验证：Tree eager、Tree FULL_DECODE_ONLY、优化后的 Tree FULL_DECODE_ONLY 在两个提示、两轮各 32 个 greedy token 上与 AR 逐 token 一致；每种 Tree 模式的 32-token 随机采样请求均完成。这是限定样例的正确性与稳定性检查，不是任意请求历史下的逐位一致保证。

两组 NPU 构树测试覆盖小树和实际 151936 词表、B/K/V=16/32/32，在 temperature=0/0.7/1/1.5 下比较 tokens、parents、depths 和 mask，fused eager、图 replay 均与独立 tensor 参考完全一致。注意力独立测试覆盖前缀 0/126/127/128/256 和变化分支。

构树单项耗时从 tensor eager 的 60.2358 ms 降到 fused eager 的 4.3519 ms；fused graph 为 4.3323 ms。这是构树耗时，不是整模型收益。短模型测试的图内存从 7.94 GiB 降到 0.62 GiB，优化前后输出一致。

数学首轮 pilot（MATH500 前 8 题、C1、完整生成参数）结果：

| 模式 | 实际输出 token | 输出 token/s | TPF | 正确题数 | 达到输出上限 |
|---|---:|---:|---:|---:|---:|
| AR | 26794 | 56.6074 | 1.0000 | 8/8 | 0 |
| Tree | 31261 | 70.3322 | 3.4642 | 8/8 | 0 |

两组输入 token 哈希与采样参数一致，所有请求自然结束；Tree/AR 吞吐比为 **1.2425×（+24.25%）**。Tree 生成了更多 token，因此收益按实际输出吞吐计算，不能仅用两组总耗时之比。这是子集诊断结果，尚未达到或证明 SGLang 全量 MATH500 的 2.7772× 收益；Linear C64 及完整数据评测仍待记录。

## 固定运行环境

| 项目 | 固定值 |
|---|---|
| vLLM commit | `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| vllm-ascend 上游基线 | `748acedfea31b795e507f9c8175585f824328d8c` |
| 数学首轮验证源码快照 | `6a321e48167f4e9d32cbbf07802c66f924681fd5`（本地验证 ref，非远端分支） |
| 镜像 ID | `sha256:b936552e3c49668f1c9be8777d3d879e0bb6b2dc41f6417dedea84ed8f68eb9f` |
| 镜像 RepoDigest | `quay.io/ascend/vllm-ascend@sha256:3d0126a39ad1ec5d431e81b3b3bba0ae2f4d592025eaba45a6a4c644e193efa2` |
| Python / torch / torch_npu / vLLM | 3.12.13 / 2.10.0+cpu / 2.10.0.post4 / 0.27.1+empty |
| 设备 | 910B4-1，物理卡 4，逻辑 npu:0，约 60.96 GiB |
| 模型 | 主机 `/data01/models/uno-qwen3-8B`，容器 `/models/uno-qwen3-8B` |
| adapter | `/models/uno-qwen3-8B/adapter` |
| 当前验证容器 | `uno-align-v6`，ID `55f6ceb8ae7ef64c87410d769d420d5fac3e72e4a9050a70ff42f5ffc7fccba2` |
| 原始结果目录 | `/data01/uno-work/sglang_alignment_20260908/v6` |

复用原镜像、包、模型和两个扩展 `.so`，只更换经过 `git archive`、SHA256 和 LF 校验的源码；ABI 二进制独立保存 manifest。每版源码在独立只读镜像目录中挂载，旧容器保留且停止。任务容器使用 privileged 和 `ASCEND_RT_VISIBLE_DEVICES=4`；原 nonprivileged 容器因同卡挂载冲突出现 DCMI -8020，改动仅限本任务容器权限。卡 0–3 的已有服务及端口 8000 保持运行。

本地复用了原模型目录，未证明其权重 SHA256 与 SGLang 表格的 Qwen/Qwen3-8B 某个固定 revision 相同。严格跨框架复核应固定同一模型文件集，不能只依据模型名称声称权重一致。

## 离线数据准备、bench 和评分

以下均为 Linux Bash 命令。每次使用新的结果目录；在启动任何模型前确认所选卡、该任务 worker 和端口 8018 无占用。离线 bench 不需要启动 HTTP 服务。数据下载和评分依赖应置于独立 CPU 环境，避免改变已验证的 NPU 运行时。

```bash
# 在有 datasets 的 CPU 环境中准备数据，下载固定 revision。
python -m benchmarks.uno.prepare_math_data \
  --output-dir /results/math-data \
  --benchmarks gsm8k math500 aime25
```

将该目录连同 manifest 以 SHA256 校验后挂入推理容器。当前测试已准备 MATH500 和 AIME25；GSM8K 下载尚未完成，不能将其列为已测数据。

```bash
cd /vllm-workspace/vllm-ascend
MODEL=/models/uno-qwen3-8B
DATA=/alignment/math-assets/math-data
MANIFEST=/alignment/math-assets/math-data-manifest.json
RESULT=/alignment/math-results-new
COMMON=(
  --model-path "$MODEL" --benchmark math500
  --data-root "$DATA" --data-manifest "$MANIFEST"
  --context-length 40960 --max-tokens 32768
  --temperature 1 --top-k 50 --top-p 0.95 --random-seed 42
  --max-num-batched-tokens 8192 --gpu-memory-utilization 0.8
)

# 首先添加 --config-only 检查真实 EngineArgs；输出目录必须单独命名。
python -m benchmarks.uno.run_math_eval "${COMMON[@]}" \
  --mode tree --adapter-path "$MODEL/adapter" --max-running-requests 1 \
  --config-only --output-dir "$RESULT/preflight-tree"

# 每组完成并确认该组 worker/设备释放后，再运行下一组。
python -m benchmarks.uno.run_math_eval "${COMMON[@]}" \
  --mode ar --max-running-requests 1 --output-dir "$RESULT/ar-c1"
python -m benchmarks.uno.run_math_eval "${COMMON[@]}" \
  --mode tree --adapter-path "$MODEL/adapter" --max-running-requests 1 \
  --output-dir "$RESULT/tree-c1"
python -m benchmarks.uno.run_math_eval "${COMMON[@]}" \
  --mode ar --max-running-requests 64 --output-dir "$RESULT/ar-c64"
python -m benchmarks.uno.run_math_eval "${COMMON[@]}" \
  --mode linear --adapter-path "$MODEL/adapter" --max-running-requests 64 \
  --output-dir "$RESULT/linear-c64"
```

首轮使用 C1 的 `--limit 8`、C64 的 `--limit 64`，其余保留完整生成参数。这类子集结果必须标为 pilot；完整评估去掉 `--limit`。换数据集时替换 `--benchmark`；AIME25 默认每题 10 次，另外两组默认 1 次。保存的 `plan.json` 包含源码 ID、输入 token 哈希、数据 revision/哈希、包版本、参数和可选容器记录；`generations.jsonl` 包含原始输出及 token IDs。

在 Linux CPU 环境评分。本次验证使用 math-verify 0.9.0、latex2sympy2_extended 1.11.0、antlr4 4.13.2、sympy 1.14.0、mpmath 1.3.0。Windows 的 math-verify timeout 子进程序列化失败可能被 scorer 当成无法解析，不能使用该环境的准确率。

```bash
# 此处目录仅放额外纯 Python 评分依赖，不替换推理环境中的 torch 等包。
PYTHONPATH=/alignment/math-assets/grade-deps-linux:/vllm-workspace/vllm-ascend \
  python -m benchmarks.uno.grade_math_eval "$RESULT/tree-c1"
```

对每组执行评分后，比较 `summary.json` 的 `tokens_per_second` 和 `accuracy`。先核对数据/prompt 哈希、采样参数、输出上限、并发、图模式、设备、模型和版本相同，再计算对应 UNO/AR 比值；同时报告实际输出长度、截断数量和准确率，不能只挑吞吐数字。

## 服务端启动方法

数学收益来自上述离线测量。下面是同配置 HTTP 启动配方，HTTP bench 的 TTFT/TPOT 应另行报告，不能与离线数字混为一组。仅在任务资源空闲时启动：

```bash
MODEL=/models/uno-qwen3-8B
vllm serve "$MODEL" --served-model-name uno-qwen3-8b \
  --host 127.0.0.1 --port 8018 --tokenizer "$MODEL" --dtype bfloat16 \
  --seed 42 --tensor-parallel-size 1 --max-model-len 40960 \
  --max-num-seqs 1 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.8 --no-enable-prefix-caching \
  --generation-config vllm \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[33]}' \
  --speculative-config '{"method":"uno","model":"/models/uno-qwen3-8B/adapter","num_speculative_tokens":32}' \
  --additional-config '{"uno_tree":{"draft_width":16,"candidate_top_k":32}}'
```

Linear C64 将 `max-num-seqs` 改为 64，去掉 `additional-config`，将 `num_speculative_tokens` 改为 8，验证图尺寸改为 `[9,18,36,72,144,288,576]`。AR 去掉 speculative/additional config，图尺寸分别为 C1 的 `[1]` 或 C64 的 `[1,2,4,8,16,32,64]`。无需 `--enable-lora`。

先 `/health`、短正确性请求、重复输出检查，再进行服务端 bench；具体客户端参数格式见 [已有在线 bench 配方](uno_reproduction.md#4-在线-benchvllm-bench-serve)，端口改为 8018，按目标并发和采样参数设置。停止后验证仅本任务的 worker 退出、卡 4 无进程、8018 端口释放，保留全部记录。
