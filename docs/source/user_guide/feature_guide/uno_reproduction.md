# UNO FULL_DECODE_ONLY 复现说明

本文记录 2026-09-05 在 Ascend 910B 单卡验证的 UNO 实现、运行环境和 bench 方法。命令均在 **Linux Bash** 中执行。已测性能来自离线 `vllm bench throughput`；服务端和 `vllm bench serve` 命令按同一代码版本整理，尚无对应的 HTTP 性能实测结果。

## 1. 分支与固定版本

| 项目 | 固定值 |
|---|---|
| 提交仓库 | [Liuchenbing-2026/vllm-ascend](https://github.com/Liuchenbing-2026/vllm-ascend) |
| 分支 ID | `uno-spec-decode` |
| UNO 实现 commit | `ed41c51137a147896034c8bd52b5b6435fd1cfec` |
| vllm-ascend 上游基线 | `748acedfea31b795e507f9c8175585f824328d8c` |
| vLLM commit | `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| vLLM 运行时版本 | `0.27.1+empty` |
| Python / PyTorch / torch_npu | `3.12.13` / `2.10.0+cpu` / `2.10.0.post4` |
| NPU | `910B4-1`，物理卡 0，TP=1 |
| 已验证镜像 ID | `sha256:b936552e3c49668f1c9be8777d3d879e0bb6b2dc41f6417dedea84ed8f68eb9f` |
| 镜像 RepoDigest | `quay.io/ascend/vllm-ascend@sha256:3d0126a39ad1ec5d431e81b3b3bba0ae2f4d592025eaba45a6a4c644e193efa2` |
| 模型及 adapter | `s-sahoo/uno-qwen3-8B`，adapter 位于同一模型目录的 `adapter/` |
| 已验证主机模型路径 | `/data01/models/uno-qwen3-8B` |
| 容器模型路径 | `/models/uno-qwen3-8B` |

分支 HEAD 可以继续增加文档提交；复现实现时固定上表的 UNO commit，不随 `main` 更新。上卡源码快照为 `e57e51d0ce7c1e6b2f34767a9aeea90475b0ef48`，它是本地验证对象，不需要从远端获取。已发布的 `ed41c5113` 仅多一处函数参数换行：Python AST 相同，其余 17 个文件的 SHA256 相同。

这次记录未固定 Hugging Face 模型 revision，也未单独记录 CANN/驱动小版本。严格复核应复用已验证模型目录、镜像和主机驱动，补存模型文件哈希及 `npu-smi info`，不能把重新下载的最新权重视为同一基线。

## 2. 准备源码和原有运行时

以下示例针对已有验证环境：`/data01/uno-work/vllm-ascend` 保留上游基线及编译好的两个 `.so`，镜像中的 vLLM 已是上表 commit。只准备独立源码目录并复用原二进制，不重装 vLLM、torch_npu 或 CANN。其他环境需先完成其安装与 ABI 校验，镜像 ID 本身不是裸机安装步骤。

在 Linux 主机执行，目录已存在时停止并选择新的复现目录：

```bash
set -euo pipefail
export UNO_ROOT=/data01/uno-repro-20260905
export UNO_SOURCE="$UNO_ROOT/vllm-ascend"
export UNO_BASE=/data01/uno-work/vllm-ascend
export UNO_CODE=ed41c51137a147896034c8bd52b5b6435fd1cfec
export UNO_IMAGE=sha256:b936552e3c49668f1c9be8777d3d879e0bb6b2dc41f6417dedea84ed8f68eb9f

test ! -e "$UNO_ROOT"
mkdir -p "$UNO_ROOT/results"
git clone --branch uno-spec-decode \
  https://github.com/Liuchenbing-2026/vllm-ascend.git "$UNO_SOURCE"
git -C "$UNO_SOURCE" checkout --detach "$UNO_CODE"
test "$(git -C "$UNO_SOURCE" rev-parse HEAD)" = "$UNO_CODE"
test "$(git -C "$UNO_BASE" rev-parse HEAD)" = 748acedfea31b795e507f9c8175585f824328d8c
git -C "$UNO_SOURCE" diff --exit-code HEAD

# 二进制独立于 Git 源码，明确记录并校验；原目录保持不变。
abi_files=(
  vllm_ascend/libvllm_ascend_kernels.so
  vllm_ascend/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so
)
(
  cd "$UNO_BASE"
  sha256sum "${abi_files[@]}"
) > "$UNO_ROOT/results/abi.sha256"
for path in "${abi_files[@]}"; do
  test -f "$UNO_BASE/$path"
  test ! -e "$UNO_SOURCE/$path"
  cp -p -- "$UNO_BASE/$path" "$UNO_SOURCE/$path"
done
(
  cd "$UNO_SOURCE"
  sha256sum -c "$UNO_ROOT/results/abi.sha256"
)
printf '%s\n' "$UNO_CODE" > "$UNO_ROOT/results/source-commit.txt"
docker image inspect "$UNO_IMAGE" > "$UNO_ROOT/results/image-inspect.json"
(
  cd /data01/models/uno-qwen3-8B
  find -L . -type f -print0 | sort -z | xargs -0 sha256sum
) > "$UNO_ROOT/results/model-files.sha256"

# 先确认卡、推理 worker 和计划端口，不能与已有服务争用资源。
npu-smi info | tee "$UNO_ROOT/results/npu-info.txt"
npu-smi info -t proc-mem -i 0 | tee "$UNO_ROOT/results/device-before.txt"
grep -q 'No process in device' "$UNO_ROOT/results/device-before.txt"
ss -ltnp > "$UNO_ROOT/results/ports-before.txt"
if ss -ltnH 'sport = :8000' | grep -q .; then
  echo 'Port 8000 is occupied'; exit 1
fi
if pgrep -af 'VLLM::|vllm.entrypoints|vllm serve'; then
  echo 'Inspect existing inference workers before continuing'; exit 1
fi
test -z "$(docker ps -aq --filter 'name=^/uno-repro$')"
```

如果从 Windows 转移源码，使用固定 commit 的 `git archive`，并在 Linux 检查归档清单、解压路径和 SHA256；不要用 ZIP 打包运行时，或覆盖已验证目录。下面的容器启动方式是对既有运行环境的复现配方，本次文档整理未重新执行它。

```bash
docker run -d --name uno-repro --network host --shm-size=16g \
  --device /dev/davinci0 --device /dev/davinci_manager \
  --device /dev/hisi_hdc --device /dev/devmm_svm \
  -v "$UNO_SOURCE:/vllm-workspace/vllm-ascend:ro" \
  -v /data01/models/uno-qwen3-8B:/models/uno-qwen3-8B:ro \
  -v "$UNO_ROOT/results:/results" \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -e ASCEND_RT_VISIBLE_DEVICES=0 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_BATCH_INVARIANT=0 -e VLLM_LOGGING_LEVEL=INFO \
  --entrypoint bash "$UNO_IMAGE" -c 'exec sleep infinity'

docker exec -it uno-repro bash
```

进入容器后，先检查包与二进制，再启动模型：

```bash
set -euo pipefail
cd /vllm-workspace/vllm-ascend
test "$(git -C /vllm-workspace/vllm rev-parse HEAD)" = 6e448d0ea9bf3d88d898b65449ca6dc2aec170ac
sha256sum -c /results/abi.sha256
python - <<'PY'
import importlib.metadata as metadata
import sys
import torch
import torch_npu
import vllm
import vllm_ascend
import vllm_ascend.vllm_ascend_C

print(sys.version)
print("torch", torch.__version__, "torch_npu", torch_npu.__version__)
print("vllm", metadata.version("vllm"), vllm.__file__)
print("vllm_ascend", vllm_ascend.__file__)
assert sys.version_info[:3] == (3, 12, 13)
assert torch.__version__ == "2.10.0+cpu"
assert torch_npu.__version__ == "2.10.0.post4"
assert metadata.version("vllm") == "0.27.1+empty"
PY
test -f /models/uno-qwen3-8B/config.json
test -f /models/uno-qwen3-8B/adapter/adapter_config.json
```

出现实际 import/loader 错误时，保留错误及对应 `.so` 的 `ldd`/`readelf` 结果。若旧容器可运行而新目录缺少旧容器具有的文件，应先检查源码覆盖和二进制拷贝，不根据普通启动日志推断 ABI 缺失。

## 3. 服务端启动与正确性请求

在容器终端 A 执行。`DRAFT_EAGER=false` 启用草稿图；改成 `true` 可做对照。两组都保留验证侧 FULL_DECODE_ONLY，**不要用全局 `--enforce-eager` 切换对照**。

```bash
set -euo pipefail
cd /vllm-workspace/vllm-ascend
export DRAFT_EAGER=false
export CASE=graph
export MODEL=/models/uno-qwen3-8B
SPEC="{\"method\":\"uno\",\"model\":\"$MODEL/adapter\",\"num_speculative_tokens\":8,\"enforce_eager\":$DRAFT_EAGER}"
CAPTURE='{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[9,18]}'

vllm serve "$MODEL" \
  --served-model-name uno-qwen3-8b --host 127.0.0.1 --port 8000 \
  --tokenizer "$MODEL" --seed 0 --tensor-parallel-size 1 \
  --max-model-len 512 --max-num-seqs 2 --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.6 --no-enable-prefix-caching \
  --generation-config vllm \
  --compilation-config "$CAPTURE" --speculative-config "$SPEC" \
  2>&1 | tee "/results/serve-$CASE.log"
```

无需 `--enable-lora`、`--lora-modules` 或 `--trust-remote-code`；UNO 自动管理内部 adapter。`--generation-config vllm` 用于在线对照时固定默认采样来源，实际采样参数仍由下面的请求显式指定。

在同一主机的终端 B 执行健康检查和两次短请求，等待模型启动完成后再压测：

```bash
curl --fail --silent --show-error http://127.0.0.1:8000/health
curl --fail --silent --show-error http://127.0.0.1:8000/v1/models
for attempt in 1 2; do
  curl --fail --silent --show-error http://127.0.0.1:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"uno-qwen3-8b","prompt":"The capital of France is","temperature":0,"max_tokens":32,"ignore_eos":true}'
  printf '\n'
done
```

确认无服务错误、两次均完成 32 token，并记录返回内容。普通 BF16 在不同请求历史下可能出现近同分 token 分歧，不能将这一步当作任意批次逐位不变的证明。草稿图组应出现日志：

```text
UNO captured FULL_DECODE_ONLY draft graphs for request counts [1, 2]
```

F=8 时验证图为 9/18 token，草稿图为 8/16 token。`max_num_seqs=2` 是本次测量范围；不要将仅捕获两个请求的结果外推到更大并发。

## 4. 在线 bench：vllm bench serve

在终端 B 进入同一容器的另一个 shell：`docker exec -it uno-repro bash`。服务需保持运行；bench 客户端本身不再加载模型。

```bash
set -euo pipefail
export CASE=graph
test ! -e "/results/online-$CASE.json"
vllm bench serve \
  --backend vllm --host 127.0.0.1 --port 8000 --endpoint /v1/completions \
  --model /models/uno-qwen3-8B --tokenizer /models/uno-qwen3-8B \
  --served-model-name uno-qwen3-8b \
  --dataset-name random --random-input-len 128 --random-output-len 128 \
  --random-range-ratio 0.0 --num-prompts 16 --num-warmups 4 --seed 0 \
  --request-rate inf --max-concurrency 2 \
  --temperature 1.0 --top-p 1.0 --top-k -1 --ignore-eos \
  --save-result --save-detailed --result-dir /results \
  --result-filename "online-$CASE.json" \
  2>&1 | tee "/results/online-$CASE.log"
```

检查请求成功数为 16、输出 token 总数为 2048，记录实际输入 token 总数、输出吞吐、TTFT 和 TPOT。随机文本经 HTTP tokenizer 处理后的实际输入统计以结果为准。

完成 graph 组后，在终端 A 用 Ctrl-C 停止自己启动的服务；在主机核对卡 0、worker 和 8000 端口均已释放。然后将服务端改为 `DRAFT_EAGER=true`、`CASE=eager` 重启；客户端改 `CASE=eager`，其余参数相同。不要同时启动两组服务。两组保持相同的健康检查、短请求和预热顺序，分别保存完整日志、结果 JSON 和启动配置。

这里的在线方案包含 HTTP、服务调度和流式响应开销。它不是下一节离线结果的来源；本次文档提交没有重新启动服务或补跑在线性能。

## 5. 离线 bench：复现已测的 4.38× 对照

先停止 HTTP 服务并确认资源释放。以下命令在容器内执行，bench 自己创建引擎，无需运行 `vllm serve`。

先检查固定长度数据集。此版本的 `--random-range-ratio 0.0` 才是固定长度；`1.0` 会引入长度波动。

```bash
python - <<'PY'
from vllm.benchmarks.datasets import RandomDataset
from vllm.tokenizers import get_tokenizer

tokenizer = get_tokenizer("/models/uno-qwen3-8B")
rows = RandomDataset(random_seed=0).sample(
    tokenizer=tokenizer, num_requests=16,
    input_len=128, output_len=128, range_ratio=0.0,
)
lengths = {(row.prompt_len, row.expected_output_len) for row in rows}
assert len(rows) == 16 and lengths == {(128, 128)}, lengths
print("BENCH_INPUT_LENGTHS_VERIFIED", len(rows), lengths)
PY
```

先执行 eager 草稿组。完成后检查本组 worker 已退出、卡无进程，再以 `CASE=graph`、`DRAFT_EAGER=false` 执行同一段命令。完整运行环境、权重和源码保持不变。

```bash
set -euo pipefail
cd /vllm-workspace/vllm-ascend
export CASE=eager
export DRAFT_EAGER=true
export MODEL=/models/uno-qwen3-8B
SPEC="{\"method\":\"uno\",\"model\":\"$MODEL/adapter\",\"num_speculative_tokens\":8,\"enforce_eager\":$DRAFT_EAGER}"
CAPTURE='{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[9,18]}'
test ! -e "/results/offline-$CASE.json"

vllm bench throughput --backend vllm --model "$MODEL" \
  --tokenizer "$MODEL" --dataset-name random \
  --random-input-len 128 --random-output-len 128 --random-range-ratio 0.0 \
  --num-prompts 16 --num-warmups 4 --seed 0 \
  --max-model-len 512 --max-num-seqs 2 --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.6 --no-enable-prefix-caching \
  --tensor-parallel-size 1 --compilation-config "$CAPTURE" \
  --speculative-config "$SPEC" --output-json "/results/offline-$CASE.json" \
  2>&1 | tee "/results/offline-$CASE.log"
```

在固定的 vLLM commit 中，该离线入口使用 `temperature=1.0`、`top_p=1.0`、`ignore_eos=True`，预热不计入计时。原生 JSON 的 `tokens_per_second` **包含输入和输出**；本表按 `2048 / elapsed_time` 统一计算输出吞吐。

| 执行方式 | elapsed_time（秒） | 输入 / 输出 token | 原生总 token/s | 输出 token/s |
|---|---:|---:|---:|---:|
| FULL 验证图 + eager 草稿 | 89.48132496513426 | 2048 / 2048 | 45.77491450418259 | 22.887457252091295 |
| FULL 验证图 + 草稿图 | 20.416630663909018 | 2048 / 2048 | 200.62076193799206 | 100.31038096899603 |

吞吐比为 **4.3827665×**。这是一次 128 输入 / 128 输出、16 条计时请求、4 条预热、并发上限 2 的对照；没有测相对 AR、多卡、长上下文或更大并发的性能。启动和图捕获时间不在计时内。

原始离线结果文件名为 `uno_eager_draft_fixed_bench-candidate-560c74bc.json` 与 `uno_full_fixed_bench-candidate-b8a4e21c.json`，两者的生产源码逐文件相同。计算结果时同时检查两组日志各有 `Total num prompt tokens: 2048` 和 `Total num output tokens: 2048`，不能只从两者合计 4096 推断输出长度。

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("/results")
results = {mode: json.loads((root / f"offline-{mode}.json").read_text())
           for mode in ("eager", "graph")}
for mode, result in results.items():
    assert result["num_requests"] == 16
    assert result["total_num_tokens"] == 4096
    lines = (root / f"offline-{mode}.log").read_text().splitlines()
    for label in ("Total num prompt tokens:", "Total num output tokens:"):
        assert any(label in line and int(line.split(label)[1].strip()) == 2048
                   for line in lines), (mode, label)
    print(mode, "output_token/s", 2048 / result["elapsed_time"])
print("speedup", results["eager"]["elapsed_time"] / results["graph"]["elapsed_time"])
PY
```

## 6. 正确性回归与验收范围

在资源空闲的容器中，按下列顺序运行。batch-invariant 单测会修改模块状态，需独立 pytest 进程。

```bash
cd /vllm-workspace/vllm-ascend
python -m pytest -q --tb=short -p no:cacheprovider \
  tests/ut/spec_decode/test_uno_config.py \
  tests/ut/spec_decode/test_uno_proposer.py \
  tests/ut/lora/test_punica_npu_dense_route.py
python -m pytest -q --tb=short -p no:cacheprovider tests/ut/test_batch_invariant.py

# 与上卡验证相同的测试函数，显式替换成离线模型路径。
python - <<'PY'
from tests.e2e.pull_request.one_card.spec_decode import test_uno as test

if __name__ == "__main__":
    method = next(iter(test.UNO))
    test.UNO[method] = {
        "main": "/models/uno-qwen3-8B",
        "adapter": "/models/uno-qwen3-8B/adapter",
    }
    test.test_uno_full_decode_graph_matches_eager_logits_across_batch_changes(method)
    print("E2E_FULL_PASS")
PY
```

已保存的结果为 90 项 UNO/LoRA UT + 19 项 batch-invariant UT 通过，另有 6 类 NPU reduction 检查通过。最终 e2e 在同一次草稿调用中固定输入噪声、位置、attention metadata 和 KV 窗口，以 `torch.equal` 比较图 replay 与 eager 的完整 logits：24 次前向、43,757,568 个值全等，覆盖实际请求数 1/2 及 1→2 恢复。测试还检查五条响应各生成 32 token、clean-token 接受率 >95%、diffusion 提议接受率 >0。

不保证任意请求历史下普通 BF16 的逐位输出一致；此前 France 提示曾出现 Germany/Spain 近同分分歧。`VLLM_BATCH_INVARIANT=1` 的注意力不支持 FULL_DECODE_ONLY，严格批次不变需求使用 eager/PIECEWISE。上述回归是相关测试范围，不是仓库全量测试通过的声明。

## 7. 结束与保留结果

退出容器 shell，并在主机停止本次创建的容器，保留容器及日志以便检查：

```bash
docker stop -t 30 uno-repro
docker inspect uno-repro > "$UNO_ROOT/results/container-final.json"
npu-smi info -t proc-mem -i 0 | tee "$UNO_ROOT/results/device-after.txt"
grep -q 'No process in device' "$UNO_ROOT/results/device-after.txt"
ss -ltnp > "$UNO_ROOT/results/ports-after.txt"
if ss -ltnH 'sport = :8000' | grep -q .; then
  echo 'Port 8000 is still occupied'; exit 1
fi
if pgrep -af 'VLLM::|vllm.entrypoints|vllm serve'; then
  echo 'Inference workers remain; inspect ownership before stopping anything'; exit 1
fi
```

保留源码 commit、镜像 inspect、驱动/NPU 信息、模型及 ABI 哈希、两组完整命令、启动日志、bench JSON、正确性输出和清理后的资源记录。再次运行时使用新结果目录，避免混合不同配置的结果。
