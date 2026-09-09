# UNO Tree FULL_DECODE_ONLY：构树优化与配对复现

后续已验证的 FP32 LoRA 投影合并、最新保留 commit 和撤回实验见[后续复现记录](uno_tree_lora_20260909.md)。本页保留构树优化当轮的历史证据。

2026-09-09，分支 `codex/uno-tree-performance`，实测源码 commit **`ecb867217a506895c36ca3f86ab4d03a4697c105`**。在 Ascend 910B 单卡、MATH500 前 8 题、并发 1 上，Tree 达到 **99.7272 token/s**，相对本轮同卡 AR 的 **57.6566 token/s** 为 **1.7297×（+72.97%）**。Tree 自身相对上一版提高 **3.19%**。本轮是子集实验，尚未达到或证明 SGLang 全量性能对齐。

[机器可读结果](uno_tree_performance_20260909.json) 包含两组完整摘要、输入哈希、输出一致性、旧版对照、负结果和验证记录。此前 Linear C64 与 Tree C1 的实现和测量见 [原对齐记录](uno_sglang_alignment.md)。

## 改动与验证

原构树 kernel 每次选择节点时扫描所有父节点的全部候选；实际配置需要检查 32×32 个候选。由于每个父节点的孩子已经按分数和 rank 排序，新实现只维护每个父节点的第一个未使用候选，每次选择只扫描 32 项。选中后推进该父节点的 rank，保留原来的分数、深度、rank、token、父节点优先级及祖先 mask。

生产代码仅修改 `vllm_ascend/ops/triton/uno_tree.py`，另在 `test_uno_tree.py` 新增 3 组 NPU 测试。没有引入新的环境变量或主机同步。验证情况：

- 18 项 NPU 回归通过，耗时 92.37 秒。新增 top-k=2/32/128 的分数相同、全负无穷、单点高分情况，图 replay 的 tokens、parents、depths、attention mask 与独立 tensor 参考完全相同；保留原构树、遍历和 KV 压紧测试。
- 整模型 FULL_DECODE_ONLY 通过两个提示、两轮各 32 个 greedy token 的稳定性检查，输出与同卡 AR 逐 token 相同；32-token 随机采样请求完成。短启动检查的上下文为 512，长上下文配置由下面的 MATH 实验验证。
- 两组 MATH 实验均正常评分、8/8 答对、无输出上限截断。AR 与 Tree 各自的 8 条完整输出 token IDs 分别与旧版对应模式完全一致。Tree 的验证轮数仍为 4512。
- 130 项既有 CPU 单元测试的源码和被测 CPU 路径保持不变，沿用上一版通过记录；本轮新增和变更的 NPU 路径以上述 18 项硬件测试验证。
- 适用的 `format.sh ci` hooks、Ruff、logger、symbolic-meta、gitleaks、两个变更 Python 文件的全部函数长度检查通过。Windows 的 C++ 工具受系统策略阻止，shellcheck 未安装；没有 C++ 或仓库 shell 文件变更，不能称为完整跨平台 CI 全绿。

构树独立图 replay 测量如下。每组计时 30 次 replay，取 3 组均值的中位数，组末 NPU 同步；输入和输出在两版间一致。该单项实验使用物理卡 1，整模型测量使用物理卡 0，计时期间没有本任务的并行 NPU 探针。

| 草稿深度 / top-k / 节点数 | 旧 kernel ms | 新 kernel ms | 单项倍数 |
|---|---:|---:|---:|
| 15 / 32 / 32 | 4.108862 | 0.193690 | 21.2136× |
| 7 / 2 / 32 | 0.350687 | 0.190180 | 1.8440× |
| 15 / 128 / 32 | 16.314922 | 0.194361 | 83.9411× |

**21.21× 仅指实际配置下的构树 kernel，不是整模型加速比。**

## 同卡配对性能与 SGLang 参考

共同条件：BF16、TP=1、FULL_DECODE_ONLY、C1、上下文 40960、输出上限 32768、temperature=1、top-k=50、top-p=0.95、seed=42、prefix cache 关闭。Tree 为 B/K/V=16/32/32，验证图尺寸 `[33]`、独立草稿图宽度 16；AR 图尺寸 `[1]`。Tree 的 async scheduling 关闭，AR 按既有实现开启，实际配置保存在摘要中。

| 版本 / 模式 | 输出 token | 计时秒数 | token/s | TPF | 正确题数 | 截断 |
|---|---:|---:|---:|---:|---:|---:|
| 上一版 AR | 26794 | 461.3236 | 58.0807 | 1.0000 | 8/8 | 0 |
| 上一版 Tree | 31261 | 323.4653 | 96.6440 | 3.4642 | 8/8 | 0 |
| 本轮 AR | 26794 | 464.7166 | 57.6566 | 1.0000 | 8/8 | 0 |
| 本轮 Tree | 31261 | 313.4652 | 99.7272 | 3.4642 | 8/8 | 0 |

上一版源码为 `55873225c2dbe001b24b1e6089213e021bbde986`，Tree/AR 为 1.6640×；本轮 Tree/AR 为 1.7297×。本轮 AR 吞吐比上一版低约 0.73%，因此不能把两轮收益比的全部变化都归因于构树优化。相同完整输出下，Tree 吞吐提高 3.19%，用时减少约 10 秒；这是每种配置一次完整子集测量，没有重复试验的置信区间。

计时覆盖离线 `generate`，包含 prefill，排除启动和图捕获，不执行请求预热。输出吞吐是实际输出 token 总数除以秒数；收益是同轮 Tree 吞吐除以 AR 吞吐，不能用两组耗时直接相除。TPF 为输出 token /（2×请求验证轮数），其中两个 target-model forward 都计入，TPF 不等于收益。

[SGLang 已合并 UNO PR 37667](https://github.com/sgl-project/sglang/pull/37667) 公布的单 H200、完整 MATH500 结果为 AR 183.12 token/s、Tree 508.56 token/s，即 **2.7772×**，Tree TPF 为 **3.450**。本地 TPF 3.4642 接近该参考，而收益仍有差距，说明执行开销值得继续重点排查；跨硬件和数据量的对照不能单独证明某个算子是根因。

本地只测 8/500 题，没有完成全量 MATH500、GSM8K 和 AIME25 配对。SGLang 原始环境未在本机重跑，不能直接比较 H200 与 910B 的绝对吞吐。采样语义也仍有差异：Ascend 为 joint top-k/top-p，参考 SGLang UNO 为先 top-k 再在保留集合计算 top-p。两边参数相同不代表行为完全相同。固定评测说明见 [SGLang README](https://github.com/sgl-project/sglang/blob/554f817948c26e8e9c8338b4a33e94a609d6f0fb/benchmark/uno/README.md)。

## 保留的负结果

注意力页长取整并复用图参数的实验快照为 `04c9c2c6f52df6f69ed9359fa3d876e6ca6bb994`。该实验虽通过短正确性检查，但 MATH 前 8 题仅为 93.4322 token/s，低于旧版 96.6440；输出变为 35030 token、TPF 3.4229。因此已撤回注意力改动，未进入本分支。实验快照和日志保存在服务器 `v12` 目录，不是对外发布的实现版本。

实际 adapter 的 LoRA 投影探针发现直接使用 BF16 收缩会在部分投影产生数值差异，最大绝对差为 0.0078125，因此未采用；缓存 FP32 权重及多流重叠也未实现。ND/NZ 布局探针收益不一致，未改变生产布局。这些单项探针都没有作为整模型性能数字。

## 源码与运行环境

| 项目 | 固定值 |
|---|---|
| 分支 | `Liuchenbing-2026/vllm-ascend:codex/uno-tree-performance` |
| 本轮测量源码 | `ecb867217a506895c36ca3f86ab4d03a4697c105` |
| 分支起点 | `9d8ea0a85c01d695f2cef603554185e766067fca`，上一轮文档提交 |
| vllm-ascend 上游基线 | `748acedfea31b795e507f9c8175585f824328d8c` |
| vLLM commit | `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| 服务器 / 设备 | q38，`bms-75847414-002`，物理卡 0，910B4-1 |
| 容器 | `uno-align-v14`，ID `d12a1cdb32d8babbe99e975539e6e676d86c42ecffd4e004110f9d260d8ba313` |
| 镜像 ID | `sha256:b936552e3c49668f1c9be8777d3d879e0bb6b2dc41f6417dedea84ed8f68eb9f` |
| 镜像 digest | `quay.io/ascend/vllm-ascend@sha256:3d0126a39ad1ec5d431e81b3b3bba0ae2f4d592025eaba45a6a4c644e193efa2` |
| Python / torch / torch_npu / vLLM | 3.12.13 / 2.10.0+cpu / 2.10.0.post4 / 0.27.1+empty |
| 主机模型路径 | `/data01/models/uno-qwen3-8B` |
| 容器模型 / adapter | `/models/uno-qwen3-8B` / `/models/uno-qwen3-8B/adapter` |
| 主机结果根目录 | `/data01/uno-work/sglang_alignment_20260908`，容器内 `/alignment` |
| 源码挂载 | `v14/runtime-source` 只读挂载至 `/vllm-workspace/vllm-ascend` |

模型 5 个基础权重文件的实际 SHA256 已与官方 Qwen3-8B 核对；adapter、tokenizer 和基础模型版本见 [模型证据](uno_sglang_alignment.md#固定运行环境)。本轮两组输入 token 哈希均为 `2b5879f045881841fba7ce8a3fe783bb3a371c5d875678b41ff93602860f518b`，与官方 tokenizer 构造的同一子集一致。

在新的源码目录取回已测实现：

```bash
git clone --branch codex/uno-tree-performance \
  https://github.com/Liuchenbing-2026/vllm-ascend.git "$UNO_SOURCE"
git -C "$UNO_SOURCE" checkout --detach ecb867217a506895c36ca3f86ab4d03a4697c105
```

保留已验证运行时，只更新源码；按 [原运行时准备方式](uno_reproduction.md#2-准备源码和原有运行时) 操作，不重新安装 torch、torch_npu 或扩展。两个原有二进制的哈希为 `4b7079cdca151c0a22ff651a0017adadd4c0ad4d11fdbe49d3f55e1ee3817552`（`libvllm_ascend_kernels.so`）和 `e58b455d1ff4c510ca299233d942864843f5f50c3417d5d6693795378a8d49ed`（`vllm_ascend_C.cpython-312-aarch64-linux-gnu.so`），与本轮 staging manifest 一致。

## 离线 bench 复现

以下为容器内 Linux Bash 命令。先确认所选设备、任务 worker 和端口空闲；先静态检查、短正确性请求和重复输出稳定性，再计时。数据准备及 Linux 评分依赖见 [原数据说明](uno_sglang_alignment.md#离线数据准备bench-和评分)。结果目录必须是新的独立目录。两种模式顺序执行，每组结束并确认 worker 释放后再启动下一组。

```bash
cd /vllm-workspace/vllm-ascend
export ASCEND_RT_VISIBLE_DEVICES=0
MODEL=/models/uno-qwen3-8B
RESULT=/alignment/tree-frontier-reproduce-new
COMMON=(
  --model-path "$MODEL" --benchmark math500
  --data-root /alignment/math-assets/math-data
  --data-manifest /alignment/math-assets/math-data-manifest.json
  --context-length 40960 --max-tokens 32768
  --temperature 1 --top-k 50 --top-p 0.95 --random-seed 42
  --max-running-requests 1 --max-num-batched-tokens 8192
  --gpu-memory-utilization 0.8 --limit 8
)
python -m benchmarks.uno.run_math_eval "${COMMON[@]}" \
  --mode tree --adapter-path "$MODEL/adapter" --config-only \
  --output-dir "$RESULT/preflight-tree"
python -m benchmarks.uno.run_math_eval "${COMMON[@]}" \
  --mode tree --adapter-path "$MODEL/adapter" --output-dir "$RESULT/tree"
python -m benchmarks.uno.run_math_eval "${COMMON[@]}" \
  --mode ar --output-dir "$RESULT/ar"
for MODE in tree ar; do
  PYTHONPATH=/alignment/math-assets/grade-deps-linux:/vllm-workspace/vllm-ascend \
    python -m benchmarks.uno.grade_math_eval "$RESULT/$MODE"
done
```

以上 runner 默认选择 FULL_DECODE_ONLY 和对应 Tree/AR 图尺寸。正式全量验证应删除 `--limit 8`，不能将本页子集数字写成全量结论。实际本轮还传入 `--runtime-record /alignment/v14/container.json`，完整命令保存在机器可读摘要的 `argv`；复现时应指向自身容器记录。评分使用 Linux math-verify 0.9.0、latex2sympy2_extended 1.11.0、antlr4 4.13.2、sympy 1.14.0、mpmath 1.3.0。

原始配对结果在 `v14/math500-{tree,ar}-c1-8/{plan.json,generations.jsonl,graded.jsonl,summary.json}`；对应日志、退出码、`paired-results.json`、`smoke-tree-graph.json`、`kernel-validation.log`、`manifest.json`、`abi-manifest.json`、`launch.json` 均保留。传输使用 LF/无 BOM 脚本、远端语法检查和 SHA256 核对；源码来自精确 commit 的 `git archive`。

## HTTP 服务端与客户端

下列命令是同参数服务启动配方；本轮收益来自离线评测，没有新执行 HTTP 压测。端口空闲时在同一运行时内启动：

```bash
MODEL=/models/uno-qwen3-8B
ASCEND_RT_VISIBLE_DEVICES=0 vllm serve "$MODEL" --served-model-name uno-qwen3-8b \
  --host 127.0.0.1 --port 8018 --tokenizer "$MODEL" --dtype bfloat16 \
  --seed 42 --tensor-parallel-size 1 --max-model-len 40960 \
  --max-num-seqs 1 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.8 --no-enable-prefix-caching --generation-config vllm \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[33]}' \
  --speculative-config '{"method":"uno","model":"/models/uno-qwen3-8B/adapter","num_speculative_tokens":32}' \
  --additional-config '{"uno_tree":{"draft_width":16,"candidate_top_k":32}}'
```

AR 去掉 speculative/additional config，capture sizes 改为 `[1]`。先检查 `/health`、短正确性请求和重复输出，再按 [在线客户端 bench 方法](uno_reproduction.md#4-在线-benchvllm-bench-serve) 使用端口 8018、并发 1 和匹配的采样参数；HTTP 的 TTFT/TPOT 和吞吐须独立报告，不能混入离线收益。

本轮评测完成后已停止 `uno-align-v14`，本任务 worker 退出、卡 0/1 释放、端口 8018 无监听；清理记录为 `v14/final-cleanup.log`。当时卡 2 有其他任务进程，未操作。
