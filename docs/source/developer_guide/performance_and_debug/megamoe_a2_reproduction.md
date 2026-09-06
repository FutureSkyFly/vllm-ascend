# MegaMoe A2 收益复现与测试方法

本文对应 2026-09-06 已验收的基线。命令分为 CPU 检查、CANN 算子对照、整模型开关对照，必须按顺序通过正确性和稳定性检查后再测性能。本次补充文档只做静态校验，没有重新启动模型或占用设备。

## 1. 环境、源码和辅助文件

| 项目 | 验收配置 |
| --- | --- |
| 硬件 | 单机 8×Ascend 910B4-1，单卡 64 GB，AIV 核数 40 |
| 服务容器 | `mm_q36`；镜像 `vllm-ascend:pr14171-v026-runtime-20260821` |
| 镜像 ID | `sha256:e6519803e088d655590cfe3b2ef9b429c1c4509e790f85b1fae4a37acd94ba75` |
| 推理软件 | Python 3.12.13、vLLM 0.27.1+empty、torch 2.10.0、torch-npu 2.10.0.post4、CANN 9.1.0 |
| AISBench | `mm_ais` 容器，`ais_bench_benchmark==3.1.20260630`、transformers 5.12.1、mmengine-lite 0.10.7 |
| 模型 | `/data1/Qwen3.6-35B-A3B-w8a8`；hidden=2048、intermediate=512、experts=256、top-k=8 |
| vLLM 功能源码 | `bccbcbe1df9fdc4957a425694f88a1b239d24f7f`；后续文档提交不改变该源码 |
| CANN V2 功能源码 | `77aa65981595df488f2c0522ae2601801e25ac89`，含批量搬运 |
| CANN V4 功能源码 | `867dfdf3c6806cb90814df30aa4ccdec54be4112`，在 V2 上增加专家波次 |
| CANN 参考基线 | `5f33f1e23d41fe0a047d01b7c7ae278777cac1f5`；也可复用保留的已验收原始 vendor 包 |

源码补丁分别应用到 README 指定的精确基线，不直接套到最新 main/master。Linux 源码归档须来自 `git archive <精确提交>`；传输后核对 SHA256、归档文件列表及顶层布局，先解压到独立目录。沿用已验证容器中的运行时和 ABI 文件，只替换源码；复用的未跟踪 `.so`、自定义算子目录和扩展缓存应单列哈希清单，不混入源码归档，也不通过重装运行时来“复现”。

提交 ZIP 的 `reproduce/` 随附本次实际执行过的测试脚本、两个算子 case JSON、AIS 配置和完整的 32 条固定输入；`reproduce/manifest.json` 给出来源与 SHA256。这些辅助文件随提交包交付，不属于生产运行时代码。将提交包完整解压后设置 `REPRO`；仅复制本 Markdown 无法提供这些输入。

以下代码块均为 **Linux Bash**，不要直接粘贴到 PowerShell。先在 `mm_q36` 容器内设置路径，替换示例中的源码和提交包目录：

```bash
set -euo pipefail
export VA=/data1/megamoe_submit/vllm-ascend
export CANN_SRC=/data1/megamoe_submit/ops-transformer-v4
export REPRO=/data1/megamoe_submission_20260906/reproduce
export MODEL_PATH=/data1/Qwen3.6-35B-A3B-w8a8
export WORK=/data1/megamoe_repro_$(date -u +%Y%m%dT%H%M%SZ)
test -d "$VA" && test -d "$CANN_SRC" && test -d "$REPRO"
test -f "$MODEL_PATH/config.json"
test ! -e "$WORK"
mkdir "$WORK"
python3 - <<'PY'
from pathlib import Path
import hashlib, json, os
r = Path(os.environ['REPRO'])
for row in json.loads((r / 'manifest.json').read_text()):
    assert hashlib.sha256((r / row['path']).read_bytes()).hexdigest() == row['sha256'], row['path']
print('REPRODUCTION_INPUTS_SHA256_PASS')
PY
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
```

`WORK` 为本轮新目录；后续另一个终端使用同一个值，不要重新生成。先记录源码提交/归档哈希、模型 `config.json` 和权重文件清单、容器及镜像 ID、完整命令/环境、NPU 分配。`npu-smi info`、`ss -ltnp` 和 worker 进程检查必须确认 0–7 卡以及 8077/29541 端口可用。原环境的 7 号卡已恢复业务服务，重测前需要另行安排空闲窗口；本文不自动停业务服务。

## 2. CPU 单测和静态检查

在保留 ABI 文件的独立 vLLM 源码副本中执行。这里的空设备可见性和 `_lazy_init` 拦截用于保证此阶段不初始化 NPU：

```bash
cd "$VA"
export PYTHONPATH="$VA:${PYTHONPATH:-}"
ASCEND_RT_VISIBLE_DEVICES='' python3 - <<'PY'
from pathlib import Path
from unittest.mock import patch
import os, sys
import pytest, torch, torch_npu
r = Path(os.environ['VA'])
files = ['a2/test_megamoe.py', 'test_megamoe_deferred_reduction.py',
         'test_fused_moe.py', 'test_prepare_finalize.py']
assert not torch.npu.is_initialized()
with patch.object(torch_npu.npu, '_lazy_init', side_effect=AssertionError('CPU_ONLY')), \
     patch.object(torch.accelerator, 'is_available', return_value=False):
    rc = pytest.main(['-q', *[str(r / 'tests/ut/ops' / f) for f in files],
                      '--confcutdir=' + str(r / 'tests/ut/ops')])
assert not torch.npu.is_initialized()
print('NPU_INITIALIZED', torch.npu.is_initialized())
sys.exit(rc)
PY
```

通过标准：`109 passed`、退出码 0、`NPU_INITIALIZED False`。新增文档提交不改变测试数量。覆盖静态归约契约、不同通信路径共享一个编译图、非均匀分片/补齐、关闭优化回退、dummy 缓存复用与修改隔离。

在安装有指定版本检查器的开发环境执行 `ruff==0.14.0` 的 check/format（7 个改动 Python 文件）、`clang-format==18.1.8 --dry-run --Werror --style=file`（A2 INT8 头文件）及 `git diff --check`。地址检查命令如下，参数为 V4 提交中的头文件：

```bash
python3 "$REPRO/../verify_dispatch_partition.py" \
  "$CANN_SRC/mc2/mega_moe/op_kernel/arch22/mega_moe_kernel_a2_int8.hpp"
```

通过标准：96 组配置、226560 个地址，`status=PASS`。这只验证整数地址分区和通知顺序，不能替代 NPU 上的并发与数值测试。

## 3. CANN 构建和运行包隔离

分别在 V2、V4 的独立源码目录构建，不覆盖已验收源码或运行包。以下展示 V4；V2 使用提交 `77aa6598`，目录独立，并将 vendor 名改成 `mmgainv2`：

```bash
cd "$CANN_SRC"
export PATH=/usr/local/Ascend/cann-9.1.0/tools/bisheng_compiler/bin:$PATH
export CMAKE_BUILD_PARALLEL_LEVEL=32
unset OMP_NUM_THREADS
bash build.sh --pkg --soc=ascend910b --ops=mega_moe --vendor_name=mmgainv4 -j32 \
  > "$WORK/build-v4.log" 2>&1
export PKG="$CANN_SRC/build_out/cann-ops-transformer-mmgainv4_linux-aarch64.run"
test -f "$PKG"
sha256sum "$PKG" > "$WORK/package-v4.sha256"
bash "$PKG" --tar tf > "$WORK/package-v4.files"
python3 - <<'PY'
from pathlib import Path, PurePosixPath
import os
names = [s for s in (Path(os.environ['WORK']) / 'package-v4.files').read_text().splitlines()
         if not s.startswith('Makeself logfile:')]
assert names and all(s.startswith('./') and '..' not in PurePosixPath(s).parts for s in names)
assert './packages/vendors/mmgainv4_transformer/op_api/lib/libcust_opapi.so' in names
PY
sha256sum -c "$WORK/package-v4.sha256"
test ! -e "$WORK/package-v4.staging" && test ! -e "$WORK/package-v4"
bash "$PKG" --noexec --extract="$WORK/package-v4.staging"
python3 - <<'PY'
from pathlib import Path
import hashlib, json, os
w = Path(os.environ['WORK']).resolve()
d = w / 'package-v4.staging'
assert d.resolve().parent == w
rows = []
for p in d.rglob('*'):
    assert p.resolve().is_relative_to(d.resolve()), str(p)
    if p.is_file():
        rows.append(dict(path=p.relative_to(d).as_posix(), sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(w / 'package-v4-manifest.json').write_text(json.dumps(rows, indent=2))
d.rename(w / 'package-v4')
PY
export VENDOR_V4="$WORK/package-v4/packages/vendors/mmgainv4_transformer"
ldd "$VENDOR_V4/op_api/lib/libcust_opapi.so" > "$WORK/ldd-v4.log"
ldd "$VENDOR_V4/op_impl/ai_core/tbe/op_tiling/liboptiling.so" >> "$WORK/ldd-v4.log"
ldd "$VENDOR_V4/op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so" >> "$WORK/ldd-v4.log"
if grep -q 'not found' "$WORK/ldd-v4.log"; then cat "$WORK/ldd-v4.log"; exit 1; fi
```

V2、参考基线也需完成相同包校验和 loader 检查后设置绝对路径 `VENDOR_V2`、`VENDOR_REF`。本次测得 V4 内核 SHA256 为 `9dfdef9b7fe58ddcd30e33856d8731f8ba3f675be2d718e43a9a743495da270d`；重新构建的二进制可能不同，应记录新哈希并重新验证，不能冒用原构建结果。

## 4. 算子正确性、异步稳定性和 ABBA 性能

执行对象是完整 `cann_ops_transformer.ops.mega_moe.mega_moe`，包含 Dispatch、两次 Linear、SwiGLU 和 Combine；不是单独测某次 memcpy 或通信子阶段。

测试输入：hidden=2048、intermediate=512、256 专家、top-k=8、EP8，每 rank 32 专家；BF16 激活、INT8 权重、NZ 格式 29，随机种子为 1024+rank。主 case 为每 rank 64/512 token 均匀路由、512 token 偏斜路由，另含 4 个 dummy 行。辅助文件 `cases_ep8.json` 和脚本定义精确生成规则。

以下在 `mm_q36` 中依次执行，所有结果写入本轮 `WORK`。每次 torchrun 前后检查端口和 worker，不能与模型服务并行：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=200
unset HCCL_DETERMINISTIC
export HCCL_CONNECT_TIMEOUT=120
export HCCL_EXEC_TIMEOUT=120
export TORCH_EXTENSIONS_DIR=/data1/megamoe_gain_20260905/torch_extensions
export MAX_JOBS=8
# 算子性能使用单一 vendor；不引入模型侧 bootstrap。
export PYTHONPATH="$CANN_SRC/torch_extension"
BASE_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
run_op() {
  local vendor=$1 mode=$2 label=$3
  test ! -e "$WORK/$label"
  mkdir "$WORK/$label"
  ASCEND_CUSTOM_OPP_PATH="$vendor" LD_LIBRARY_PATH="$vendor/op_api/lib:$BASE_LD_LIBRARY_PATH" \
  timeout --signal=TERM --kill-after=25s 480s \
    torchrun --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29541 \
    "$REPRO/megamoe_bench_verified.py" "$REPRO/cases_ep8.json" \
    "$mode" "$WORK/$label" "$WORK/op-reference" > "$WORK/$label/run.log" 2>&1
}
run_op "$VENDOR_REF" record op-reference
run_op "$VENDOR_V2" compare v2-correct
run_op "$VENDOR_V4" compare v4-correct
```

通过标准：8 个 rank 的 3 个 case 均 `correctness=PASS`，输入指纹与参考相同，全部输出 `torch.equal`，每例 5 次重复一致；非有限值、全零或退化稀疏输出均判失败。逐 rank JSON 和退出码必须全部检查，不能只看 rank 0。

另做模型 import 顺序与 buffer 契约验证。`cases_ep8_buffer164.json` 含 `buffer164_sync`、`buffer164_async40`，开启真实模型的 4 行 sentinel，`sym_inter=1024` 是该组对称缓冲参数，FFN intermediate 仍为 512。这里 **HCCL_BUFFSIZE=164**；不要与主性能/服务的 200 混用。

```bash
export HCCL_BUFFSIZE=164
export PYTHONPATH="$VA:$CANN_SRC/torch_extension"
mkdir "$WORK/buffer-reference"
ASCEND_CUSTOM_OPP_PATH="$VENDOR_REF" LD_LIBRARY_PATH="$VENDOR_REF/op_api/lib:$BASE_LD_LIBRARY_PATH" \
  torchrun --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29541 \
  "$REPRO/megamoe_bench_buffer_contract.py" "$REPRO/cases_ep8_buffer164.json" \
  record "$WORK/buffer-reference" "$WORK/buffer-reference"
# 保留原始 harness 字节；仅在本轮 wrapper 副本中重定位两个路径。
cp "$REPRO/megamoe_bench_buffer_contract.py" "$WORK/"
python3 - <<'PY'
from pathlib import Path
import os
r, w = Path(os.environ['REPRO']), Path(os.environ['WORK'])
s = (r / 'megamoe_bench_vendor_order_checked.py').read_text()
old = "ROOT = Path('/data1/megamoe_gain_20260905')"
assert s.count(old) == 1
s = s.replace(old, "ROOT = Path(os.environ['WORK'])")
old_vendor = "vllm_vendor = ROOT / 'vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer'"
assert s.count(old_vendor) == 1
s = s.replace(old_vendor, "vllm_vendor = Path(os.environ['VA']) / 'vllm_ascend/_cann_ops_custom/vendors/custom_transformer'")
(w / 'vendor_order_check.py').write_text(s)
PY
mkdir "$WORK/buffer-v4"
export ASCEND_CUSTOM_OPP_PATH="$VENDOR_V4:$VA/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
export MMGAIN_EXPECTED_VENDOR="$VENDOR_V4"
export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_PROCESS_LOG_PATH="$WORK/buffer-v4/cann_logs"
mkdir "$ASCEND_PROCESS_LOG_PATH"
LD_LIBRARY_PATH="$VENDOR_V4/op_api/lib:$BASE_LD_LIBRARY_PATH" \
  torchrun --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29541 \
  "$WORK/vendor_order_check.py" "$REPRO/cases_ep8_buffer164.json" \
  compare "$WORK/buffer-v4" "$WORK/buffer-reference"
```

通过标准：另 16 个 rank/case 精确一致、每例 5 次重复，async case 每 rank 排队 40 次后一致；`vendor_postflight_rank*.json` 中 API/tiling/proto 的实际加载路径属于目标 V4，CANN 日志里的 MegaMoe bin path 及哈希对应本轮目标内核。三种主 case 的 24 条加这 16 条，共 40 个 rank/case。仅设置环境变量或看到 import 成功不代表实际执行了 V4 内核。

通过以上检查后，在关闭 CANN 调试日志的同等环境中测 ABBA：

```bash
unset ASCEND_GLOBAL_LOG_LEVEL ASCEND_SLOG_PRINT_TO_STDOUT ASCEND_PROCESS_LOG_PATH
export HCCL_BUFFSIZE=200
export PYTHONPATH="$CANN_SRC/torch_extension"
run_op "$VENDOR_V2" perf A1
run_op "$VENDOR_V4" perf B1
run_op "$VENDOR_V4" perf B2
run_op "$VENDOR_V2" perf A2
```

每个 case：先重复正确性检查，10 次预热，30 次计时；每次 barrier 和 NPU synchronize 后计时一次调用，再 synchronize。记录的是主机 wall time，含该调用完成等待，不是 profiler 中单个 kernel 时间。

汇总规则：每个迭代先取 8 rank 的最大 `wall_us[i]`，再对 30 个最大值取中位数；A1/A2 中位数求均值为 V2，B1/B2 为 V4；耗时下降为 `(V2−V4)/V2×100%`。同时要求两轮 V4 均低于两轮 V2。历史结果为：64 均匀 926.028→801.638 µs；512 均匀 1682.620→1425.380 µs；512 偏斜 4730.581→4609.792 µs。

## 5. 整模型启动：只切换 MEGAMOE

算子测试退出、0–7 卡空闲且 8077 无监听后，在 `mm_q36` 容器的服务终端设置以下环境。四轮使用同一份提交源码、模型、V4 vendor、运行时和缓存策略；按 **R1=1、R2=0、R3=1、R4=0** 顺序，每轮单独启动、验收、停止。

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ASCEND_CUSTOM_OPP_PATH="$VENDOR_V4:$VA/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
export LD_LIBRARY_PATH="$VENDOR_V4/op_api/lib:$BASE_LD_LIBRARY_PATH"
export PYTHONPATH="$VA:$CANN_SRC/torch_extension"
export TORCH_EXTENSIONS_DIR=/data1/megamoe_gain_20260905/torch_extensions
export VLLM_CACHE_ROOT=/data1/megamoe_gain_20260905/vllm_cache
export MAX_JOBS=8
export HCCL_DETERMINISTIC=true
export MEGAMOE_SKIP_COMPILED=0
export SHARED_EXPERT_OVERLAP=0
export PREFIX_CACHING=0
export PORT=8077
export MAX_NUM_SEQS=32
export MAX_MODEL_LEN=32768
export MAX_NUM_BATCHED_TOKENS=4096
export MEGA_MOE_MIN_TOKENS=512
export MEGAMOE=1
export ROUND=R1
test ! -e "$WORK/$ROUND"
mkdir "$WORK/$ROUND"
bash "$VA/examples/online_serving/serve_qwen36_35b_a3b_w8a8_megamoe_a2.sh" \
  > "$WORK/$ROUND/server.log" 2>&1
```

该命令以前台方式运行。在另一终端等待 `/health` 返回 200，再做后续检查。脚本固定 TP8+EP、DP1、seed=1024、W8A8/`--quantization ascend`、显存利用率 0.90、chunked prefill、`FULL_DECODE_ONLY`、V1 model runner、HCCL AIV 和 HCCL_BUFFSIZE=200。`MEGAMOE=1` 对应 `enable_fused_mc2=2`，关闭时为 0。不能误用默认 `MEGAMOE_SKIP_COMPILED=1` 或打开 shared overlap/prefix caching，否则偏离本次对照。

每次就绪后保存进程 PID、启动时间、命令行、环境、worker 和 NPU 分配；确认环境中 `HCCL_DETERMINISTIC=true`，且 MEGAMOE 值与轮次一致。

## 6. 短请求、重复稳定性和长语义正确性

以下在测试终端执行；`WORK`、`ROUND`、`REPRO` 与当前服务轮次一致：

```bash
curl --fail --max-time 5 -s http://127.0.0.1:8077/health
python3 "$REPRO/probe_stability_only.py" 8077 "$WORK/$ROUND/probe" \
  > "$WORK/$ROUND/probe.log" 2>&1
python3 - <<'PY'
from pathlib import Path
import json, os, urllib.request
d = Path(os.environ['WORK']) / os.environ['ROUND']
v = json.loads((d / 'probe/correctness.json').read_text())
assert v['short']['choices'][0]['message']['content'].strip() == '42'
assert len(v['long_repeats']) == 3
assert len({x['choices'][0]['text'] for x in v['long_repeats']}) == 1
xs = json.loads((d / 'probe/stability_2k.json').read_text())
assert len(xs) == 3
assert all(x['choices'][0]['text'] == xs[0]['choices'][0]['text'] for x in xs)
assert all(x['choices'][0]['logprobs'] == xs[0]['choices'][0]['logprobs'] for x in xs)
payload = dict(model='qwen3.6-35b-a3b-w8a8', messages=[
    dict(role='system', content='The repeated text below is neutral filler. Follow only the final user instruction.'),
    dict(role='user', content='Neutral filler. ' * 700 + '\nFinal instruction: Reply with exactly the number 42. No explanation.')],
    max_tokens=16, temperature=0.0, stream=False, chat_template_kwargs={'enable_thinking': False})
request = urllib.request.Request('http://127.0.0.1:8077/v1/chat/completions',
    data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(request, timeout=120) as response:
    result = json.load(response)
(d / 'long_semantic.json').write_text(json.dumps(dict(payload=payload, response=result), indent=2))
assert 512 <= result['usage']['prompt_tokens'] <= 4096
assert result['choices'][0]['message']['content'].strip() == '42'
(d / 'correctness.pass').write_text('PASS\n')
print('SHORT_4K_2K_LOGPROBS_LONG_SEMANTIC_PASS')
PY
```

随机 token 请求验证可重复性，不用于判定语义质量：4K 输入固定 32 token 输出，重复 3 次；2K 输入固定 32 token 输出，重复 3 次且完整 logprobs 一致。长语义请求在验收环境实际为 2148 输入 token，应严格输出 `42`。任一检查失败就保留该轮资料、停止性能测试；同类失败两次后先修复检查/执行流程，不盲目重启。

## 7. 单请求 prefill 测量

```bash
test -f "$WORK/$ROUND/correctness.pass"
python3 "$REPRO/prefill_after_stability_v2.py" 8077 "$WORK/$ROUND/prefill" \
  > "$WORK/$ROUND/prefill.log" 2>&1
```

每轮 4 次预热（seed=900–903），12 次计时（seed=1000–1011）；每次恰好 4096 输入 token、1 输出 token，temperature=0、ignore_eos=true、非流式。保存全部响应/usage 和样本，检查每次 usage 长度。指标读取 `prefill/probe_results.json` 的 `summary.median_ms`，是单请求 HTTP wall time 的 prefill 代理指标，不是 AISBench 并发 TTFT，也不是单个 kernel 耗时。

## 8. AISBench：固定数据、4K/1K/C16/32

使用提交包内的数据，不重新随机生成。`test.jsonl` 共 32 条合成文本，`train.jsonl` 为空，使用 GSM8K 数据加载格式和 ZeroRetriever；此处测试性能，不将其称为 GSM8K 准确率评测。文本经 chat 模板后约 4K 输入，不能把它称为逐条恰好 4096 token。数据 SHA256：`78cd8ff755f8ebdf5a982d0b9cb90e5216b00416c32c1cb6e3c31fe20d734e59`。

为本轮创建配置副本，仅改模型路径和数据位置，保留其他负载参数：

```bash
test -f "$WORK/$ROUND/correctness.pass"
python3 - <<'PY'
from pathlib import Path
import os
r, w = Path(os.environ['REPRO']), Path(os.environ['WORK'])
config = w / 'ais_config.py'
old_config = '/data1/megamoe_gain_20260905/aisbench/config_4k_1k_c16.py'
s = (r / 'aisbench/config_4k_1k_c16.py').read_text()
s = s.replace('/data1/Qwen3.6-35B-A3B-w8a8', os.environ['MODEL_PATH'])
s = s.replace('/data1/megamoe_gain_20260905/aisbench/dataset', str(r / 'aisbench/dataset'))
if config.exists():
    assert config.read_text() == s  # 四轮配置必须一致
else:
    config.write_text(s)
v = (r / 'aisbench/validate_config.py').read_text().replace(old_config, str(config))
(w / 'validate_ais_config.py').write_text(v)
PY
```

从宿主机执行下面两条（`mm_ais` 应能访问相同的 `/data1` 路径并连到 127.0.0.1:8077；`WORK`、`ROUND` 需设置为当前轮次值）。校验器使用 AISBench 自带配置检查器和真实数据加载器：

```bash
docker exec mm_ais python3 "$WORK/validate_ais_config.py"
docker exec mm_ais ais_bench "$WORK/ais_config.py" \
  --mode perf --num-prompts 32 --num-warmups 1 --work-dir "$WORK/$ROUND/aisbench"
```

负载固定为 `VLLMCustomAPIChatStream`、并发 `batch_size=16`、`request_rate=0`、`retry=2`、`max_out_len=1024`、temperature=0、ignore_eos=true。每轮 1 个 AIS warmup，然后 32 个计入统计的请求；核对最终成功数 32/32、长度与错误/重试日志。四轮合计 128/128；不能只选成功或更快的轮次。

结果保存在 `aisbench/<timestamp>/performances/mmgain-vllm-chat/gsm8k.json` 及同目录 CSV，保存原始文件和哈希。指标取输出 token 吞吐、平均 TTFT、平均 E2EL；不要混用总 token 吞吐或把单请求 prefill 与并发 TTFT 拼成同一指标。

## 9. 轮次结束、结果计算和判定

每轮完成后在服务终端 Ctrl+C 正常退出；按记录的 PID、启动时间、进程归属确认服务和 worker 已停止，并复查 8077、29541 与 0–7 卡占用。恢复借用的业务资源时按保留的 Deployment/副本/模板基线恢复，不根据过期 PID 批量杀进程。

然后开始下一轮（R2: MEGAMOE=0，R3: 1，R4: 0），重复第 5–8 节。保持 V4 vendor、源码、运行时、模型、缓存目录、batch 和服务选项不变，每轮均重新完成正确性/稳定性验收，不复用上一轮的 pass 文件。

汇总算法：先分别计算 R1/R3 与 R2/R4 的指标均值。prefill 指标先取各轮 12 个样本的中位数，再按模式求均值；吞吐提升为 `(on/off−1)×100%`，时延下降为 `(off−on)/off×100%`。同时检查两轮 on 的 prefill 都低于两轮 off、吞吐都高于两轮 off，并保存 R1↔R2、R3↔R4 两组配对变化。

| 指标 | off 两轮均值 | on 两轮均值 | 本次验收变化 |
| --- | ---: | ---: | ---: |
| 单请求 prefill p50 | 241.184751 ms | 235.037432 ms | −2.5488% |
| AIS 输出吞吐 | 655.3849 token/s | 662.2442 token/s | +1.0466% |
| AIS 平均 TTFT | 1964.40 ms | 1905.45 ms | −3.0009% |
| AIS 平均 E2EL | 24877.1 ms | 24620.8 ms | −1.0303% |

这些是历史验收观察值，不是新环境必须精确达到的固定阈值，也不是统计显著性或所有负载都有收益的保证。整模型收益对应整套优化的开关对照；算子的 15.29% 是 V4 相对 V2 的增量，不能标成整模型收益，也不能归因于单个 Python/CANN 提交。
