# UNO Linear / Tree：SGLang 性能口径对齐

本页记录 2026-09-08 至 09-09 的 Ascend 910B 单卡实现与验证。开发分支为 `codex/uno-sglang-alignment`，基于已发布的 `uno-spec-decode` 分支 `eb834d8336871404554e6e676ca5256534d967ca`。旧版草稿图与 eager 草稿的 4.38× 对照见 [原复现记录](uno_reproduction.md)；该数字不是相对 AR 的收益。

本轮同卡 MATH500 pilot：Linear C64 / AR C64 为 **1.6559×**，Tree C1 / AR C1 为 **1.6640×**，均为 FULL_DECODE_ONLY。分别只测前 64 / 8 题，完整输出与旧版对应模式一致；尚不能声称全量性能与 SGLang 对齐。[机器可读结果](uno_math500_pilot_20260909.json) 保存完整参数、配对摘要、源码、容器、模型哈希和 tokenizer 对齐证据。

当前整模型验证对应实现 commit `55873225c2dbe001b24b1e6089213e021bbde986`。分支后续文档提交不会改变该测量对象。已有运行环境的源码准备方式见 [原复现记录](uno_reproduction.md#2-准备源码和原有运行时)，将其分支和实现版本替换为：

```bash
git clone --branch codex/uno-sglang-alignment \
  https://github.com/Liuchenbing-2026/vllm-ascend.git "$UNO_SOURCE"
git -C "$UNO_SOURCE" checkout --detach 55873225c2dbe001b24b1e6089213e021bbde986
```

`UNO_SOURCE` 应是新的独立目录；保留原运行时、两个扩展二进制及其哈希，按原记录挂载源码。此命令用于取回实现，不是重新安装运行时。

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

Linear 使用 F=8 的独立草稿图。在验证图的最大请求数范围内，草稿侧逐一捕获每个真实 batch 大小；例如验证图配置 `[9,18,36,72,144,288,576]` 时，草稿捕获请求数 1–64。验证侧仍可按档位填充，草稿侧不添加虚假请求，避免共享 KV 的写入冲突。修复前只捕获请求数 `[1,2,4,8,16,32,64]`，其余 batch 会退回 eager。

Tree 新增完整的 best-first 构树、祖先 attention mask、目标采样后的分支遍历和接受路径 KV 压紧；验证侧使用 paged BSH FIA-v2，并为每个图尺寸复用一份 workspace。分支遍历与全部 GQA 层的 KV 压紧分别使用一个融合 kernel，KV kernel 在写回前先加载所有接受行，以保留源和目标重叠时的正确性。

Tree 当前只支持 Qwen3ForCausalLM（包括 SDAR alias）、TP=1、并发 1、单一 GQA KV cache group、非量化 KV、无滑窗、关闭 prefix cache、standard target sampling。允许 temperature/top-k/top-p，暂不支持 penalties、logprobs、structured outputs、thinking budget 等 logits processors。草稿宽度为 2–16，候选数为 2–128，验证节点预算介于草稿宽度与 32 之间。

已验证：Tree eager、Tree FULL_DECODE_ONLY、优化后的 Tree FULL_DECODE_ONLY 在两个提示、两轮各 32 个 greedy token 上与 AR 逐 token 一致；每种 Tree 模式的 32-token 随机采样请求均完成。这是限定样例的正确性与稳定性检查，不是任意请求历史下的逐位一致保证。

最新候选 `55873225c2dbe001b24b1e6089213e021bbde986` 通过 130 项单元测试和 15 项 NPU 回归测试。三组构树测试覆盖小树、实际 151936 词表的 B/K/V=16/32/32、候选 top-k=128，在 temperature=0/0.7/1/1.5 下比较 tokens、parents、depths 和 mask，fused eager、图 replay 均与独立 tensor 参考完全一致。另有 4 组不同节点数的遍历测试，以及 BF16/FP16、1/4/9/16 接受行的 8 组 KV 测试，覆盖跨页、源目标重叠、填充和全拒绝。注意力独立测试覆盖前缀 0/126/127/128/256 和变化分支。

构树单项耗时从 tensor eager 的 60.2358 ms 降到 fused eager 的 4.3519 ms；fused graph 为 4.3323 ms。这是构树耗时，不是整模型收益。短模型测试的图内存从 7.94 GiB 降到 0.62 GiB，优化前后输出一致。

最新后处理单项测试：72 份 KV cache 的压紧从 24.9221 ms 降到 0.3168 ms，遍历从 8.1853 ms 降到 0.2215 ms（3 次预热、25 次计时、前后 NPU 同步）。这些耗时不等于整模型收益。09-09 在物理卡 0 完成最新融合后处理的整模型复核：Tree FULL_DECODE_ONLY 与同卡 AR FULL_DECODE_ONLY 的两轮、两个提示、各 32 个 greedy token 逐一相同，输出稳定，32-token 随机采样正常。

Linear 扩展图覆盖的整模型检查通过：捕获请求数 1–9，实际运行覆盖 1/2/3/5/9 和重复 batch 切换；48 次草稿 forward、209063936 个 logits 在同一噪声输入、RoPE 位置和 KV 窗口下与 eager 逐值相同。检查使用完整词表，不放宽浮点容差。随后 C64 数学评测成功捕获全部请求数 1–64，完整输出与旧版对应模式一致。

静态检查通过适用的 `format.sh ci` hooks、Ruff 检查与格式检查、logger、symbolic-meta、gitleaks，以及 22 个变更 Python 文件全部函数的长度检查。Windows 上 C++ 格式工具受系统策略阻止，shellcheck 未安装；本次没有 C++ 或仓库 shell 文件变更。已记录跳过项及显式解释器替代检查，不能将此结果称为完整跨平台 CI 全绿。

最新卡 0 配对 pilot（MATH500 前 8 题、C1、完整生成参数）：

| 模式 | 实际输出 token | 计时秒数 | 输出 token/s | TPF | 正确题数 | 达到输出上限 |
|---|---:|---:|---:|---:|---:|---:|
| AR | 26794 | 461.3236 | 58.0807 | 1.0000 | 8/8 | 0 |
| Tree，融合遍历/KV | 31261 | 323.4653 | 96.6440 | 3.4642 | 8/8 | 0 |

Tree/AR 为 **1.6640×（+66.40%）**。输入 token 哈希、采样参数、设备、运行时和源码相同，两组所有请求自然结束。AR 和 Tree 各自的 8 条完整输出 token IDs 还分别与旧版对应模式相同，验证轮数未变；融合后处理没有改变这组随机采样输出。该结果来自子集，不等于已达到 SGLang 全量 MATH500 的 2.7772×。原始文件为 `v10-card0/math500-{ar,tree}-c1-8/{plan.json,generations.jsonl,graded.jsonl,summary.json}`。

数学首轮 pilot（旧卡 4、融合遍历/KV 之前，MATH500 前 8 题、C1、完整生成参数）结果：

| 模式 | 实际输出 token | 输出 token/s | TPF | 正确题数 | 达到输出上限 |
|---|---:|---:|---:|---:|---:|
| AR | 26794 | 56.6074 | 1.0000 | 8/8 | 0 |
| Tree | 31261 | 70.3322 | 3.4642 | 8/8 | 0 |

两组输入 token 哈希与采样参数一致，所有请求自然结束；Tree/AR 吞吐比为 **1.2425×（+24.25%）**。Tree 生成了更多 token，因此收益按实际输出吞吐计算，不能仅用两组总耗时之比。这是融合遍历/KV 之前的子集诊断结果，尚未达到或证明 SGLang 全量 MATH500 的 2.7772× 收益。

最新卡 0 Linear 配对 pilot（MATH500 前 64 题、C64、完整生成参数）：

| 模式 | 实际输出 token | 计时秒数 | 输出 token/s | TPF | 正确题数 | 达到输出上限 |
|---|---:|---:|---:|---:|---:|---:|
| AR | 320660 | 935.0803 | 342.9224 | 1.0000 | 60/64 | 1 |
| Linear，全部真实 batch 草稿图 | 320541 | 564.4993 | 567.8324 | 2.6938 | 61/64 | 0 |

Linear/AR 为 **1.6559×（+65.59%）**。两组配置和输入哈希配对一致；AR 与 Linear 的 64 条完整输出 token IDs 分别与旧版对应模式逐一相同，验证轮数未变。原始文件为 `v10-card0/math500-{ar,linear}-c64-64/{plan.json,generations.jsonl,graded.jsonl,summary.json}`。这组子集收益接近 SGLang 全量 MATH500 的 1.7578×，但 64 题只有一批请求，不能据此宣称全量性能已经对齐。

Linear 修复前 pilot（旧卡 4，MATH500 前 64 题、C64、完整生成参数）：

| 模式 | 实际输出 token | 计时秒数 | 输出 token/s | TPF | 正确题数 | 达到输出上限 |
|---|---:|---:|---:|---:|---:|---:|
| AR | 320660 | 933.4403 | 343.5249 | 1.0000 | 60/64 | 1 |
| Linear，稀疏草稿图 | 320541 | 1293.8707 | 247.7380 | 2.6938 | 61/64 | 0 |

这组旧版 Linear/AR 为 **0.7212×（−27.88%）**，没有性能收益。运行日志和源码确认：请求减少到不在捕获集合的 batch 大小时，草稿会执行 eager。新版通过全部真实 batch 捕获修复了该路径，上方卡 0 配对结果已转为正收益；两代分别与各自同卡 AR 相除，未混用不同卡的分母。

## 固定运行环境

| 项目 | 固定值 |
|---|---|
| vLLM commit | `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| vllm-ascend 上游基线 | `748acedfea31b795e507f9c8175585f824328d8c` |
| 数学首轮验证源码快照 | `6a321e48167f4e9d32cbbf07802c66f924681fd5`（本地验证 ref，非远端分支） |
| 镜像 ID | `sha256:b936552e3c49668f1c9be8777d3d879e0bb6b2dc41f6417dedea84ed8f68eb9f` |
| 镜像 RepoDigest | `quay.io/ascend/vllm-ascend@sha256:3d0126a39ad1ec5d431e81b3b3bba0ae2f4d592025eaba45a6a4c644e193efa2` |
| Python / torch / torch_npu / vLLM | 3.12.13 / 2.10.0+cpu / 2.10.0.post4 / 0.27.1+empty |
| 设备 | 910B4-1，逻辑 npu:0，约 60.96 GiB；首轮物理卡 4，09-09 后续验证物理卡 0 |
| 模型 | 主机 `/data01/models/uno-qwen3-8B`，容器 `/models/uno-qwen3-8B` |
| adapter | `/models/uno-qwen3-8B/adapter` |
| 下载 bundle | `s-sahoo/uno-qwen3-8B@8819e09ac901e7290d8d89d62c98b9f756c602fe` |
| bundle 标注的基础来源 | `IFM/uno-qwen3-8b-base@4ccfeed3fba497e40495fe6dc5c15c89f7f1e2cd` |
| bundle 标注的 adapter 来源 | `s-sahoo/uno-qwen3-8B@79536cf8c70aa48b9badc2532ffef2089947463e3` |
| 首轮验证容器 | `uno-align-v6`，ID `55f6ceb8ae7ef64c87410d769d420d5fac3e72e4a9050a70ff42f5ffc7fccba2`，已停止并保留 |
| 当前源码快照 | `55873225c2dbe001b24b1e6089213e021bbde986` |
| 当前验证容器 | `uno-align-v10-card0`，ID `ccc0fd4f5463744b61c431383ebb79bc7e4cad788a2f99cdbfa1dc868b6d0710` |
| 原始结果目录 | `/data01/uno-work/sglang_alignment_20260908/v6` 和 `v10-card0` |

复用原镜像、包、模型和两个扩展 `.so`，只更换经过 `git archive`、SHA256 和 LF 校验的源码；ABI 二进制独立保存 manifest。每版源码在独立只读镜像目录中挂载，旧容器保留且停止。任务容器使用 privileged；原 nonprivileged 容器因同卡挂载冲突出现 DCMI -8020，改动仅限本任务容器权限。

首轮使用 `ASCEND_RT_VISIBLE_DEVICES=4`，当时卡 0–3 上的既有服务保持运行。09-09 用户通知卡 0–3 已释放后，检查确认无设备进程，将新容器配置为 `ASCEND_RT_VISIBLE_DEVICES=0`。源码、镜像和 ABI 哈希再次核验一致。后续性能必须与卡 0 上新测的 AR 配对，不直接除以卡 4 的旧吞吐；启动命令和资源状态保存在 `v10-card0/launch.json`、`baseline.json`、`container.json`。

09-09 对服务器实际文件重新计算 SHA256，5 个基础权重分片全部与 [Qwen 官方文件页](https://huggingface.co/Qwen/Qwen3-8B/tree/main) 公布值一致，也与下载 bundle 的固定清单一致；adapter 与 bundle 清单一致。完整记录为 `v10-card0/model-file-hashes.json`，包含配置、tokenizer、chat template 等文件的本地哈希。

| 文件 | SHA256 |
|---|---|
| `model-00001-of-00005.safetensors` | `31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f` |
| `model-00002-of-00005.safetensors` | `5991236cea6fe21f3d43cab0f0e84448734fbbe0789816202989f2ddc9d18282` |
| `model-00003-of-00005.safetensors` | `c5185c4794be2d8a9784d5753c9922db38df478ce11f9ed0b415b7304d896836` |
| `model-00004-of-00005.safetensors` | `b5ee7de71fbf17db3d5704e0c8f2bc7d005ca9e1d7ca2aeb19827b0cfcaa917a` |
| `model-00005-of-00005.safetensors` | `20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff` |
| `adapter/adapter_model.safetensors` | `f22fbbc857bce91c6c3a29842932efb355a799c9ab2f1a061ee7917be2f44eb2` |

另用本机缓存的 `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218` 官方 tokenizer，离线构造相同 MATH500 输入：前 8 题和前 64 题的 token IDs 哈希分别为 `2b5879f045881841fba7ce8a3fe783bb3a371c5d875678b41ff93602860f518b`、`336b8cec581a0a2754abee75d111de32fcef220fcaf7c5bb8a5312688877fd4a`，均与服务器实际 bench 记录一致。记录为 `v10-card0/qwen-tokenizer-comparison.json`。

这些检查证明了基础权重和已测子集输入的对应关系；SGLang 原始跑分的完整环境未在本机重跑，采样语义差异仍然存在。本地配对 AR/UNO 使用同一 tokenizer、prompt 哈希和采样实现。

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

先 `/health`、短正确性请求、重复输出检查，再进行服务端 bench；具体客户端参数格式见 [已有在线 bench 配方](uno_reproduction.md#4-在线-benchvllm-bench-serve)，端口改为 8018，按目标并发和采样参数设置。停止后验证仅本任务的 worker 退出、所选卡无进程、8018 端口释放，保留全部记录。
