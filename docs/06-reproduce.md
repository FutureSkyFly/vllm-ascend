# 06 —— 复现手册

从零把 GLM-5.3-Flash w8a8 在 Atlas A2 上拉起来，并复现本仓库记录的每一个结论。
**这一页只写"怎么做"和"应该看到什么"；为什么这么做在 docs/01–05。**

本手册对应本仓库 commit **`03f29c8`**（`glm5.3-flash-w8a8` 分支）。

---

## 1. 环境（下面全是实测读数，不是推荐配置）

| 项 | 值 | 怎么查 |
|---|---|---|
| 硬件 | Atlas A2，8 × Ascend 910B4-1，每卡 65536 MB（可用约 60.96 GiB） | `npu-smi info` |
| 宿主 OS / 内核 | Ubuntu 22.04 LTS / 5.15.0-25-generic | `uname -r` |
| 驱动 / 固件 | Software 26.0.rc1 / Firmware 7.7.0.9.220 | `npu-smi info -t board -i 0` |
| 镜像 | `quay.io/atlas-ci/vllm-atlas-temp:glm-5.3-flash-0902-1-910b-openeuler-33646519920-1-arm64-temp` | `docker inspect glm53s --format '{{.Config.Image}}'` |
| 镜像摘要 | `sha256:5f949be392518e55ccd96c8086959049f9dec30a1b9da85de1d52bd3de30f103` | `docker inspect glm53s --format '{{.Image}}'` |
| CANN toolkit | 9.1.0（innerversion `V100R001C11SPC001B243`） | `cat /usr/local/Ascend/ascend-toolkit/latest/*/ascend_toolkit_install.info` |
| Python / torch / torch_npu | 3.12.13 / 2.10.0 / 2.10.0.post2 | `python3 -c "import torch,torch_npu;print(torch.__version__,torch_npu.__version__)"` |
| transformers | 5.16.0.dev0（editable） | `python3 -c "import transformers;print(transformers.__version__)"` |
| **vLLM** | **0.23.0 @ `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`** | `git -C /vllm-workspace/vllm rev-parse HEAD` |
| **vllm-ascend** | **`db701c1fd31de40ab028628ee2580ae68c7fd3c0`**（2026-09-02） | `git -C /vllm-workspace/vllm-ascend rev-parse HEAD` |
| 权重 | `/data02/GLM-5.3-Flash-w8a8-b0829`，311 GB | `du -sh` |

换环境时**这两个 commit 最要紧**：本仓库所有行号（`glm5_next.py:339`、`core.py:1279`、
`scheduler.py:781` 等）都是对着它们数的。

---

## 2. 前置

### 2.1 权重

用 `quantize/run_quant.sh` 从 `ZhipuAI/GLM-5.3-Flash` 量化，或直接用已有产物。
**必须用 `glm5_next_quant_0829` 分支**——`0830` 是分叉不是新版，它早于 ViT 导出修复，
会静默丢掉全部 347 个 `model.visual.*` 张量。核对：

```bash
python3 quantize/inspect_artifact.py /data02/GLM-5.3-Flash-w8a8-b0829
```

### 2.2 起容器

```bash
bash serve/docker-run.sh
```

**唯一的硬前提是 `--ulimit memlock=-1`。** docker 默认 max-locked-memory 是 64 KB，
`--privileged` 不会抬高它；缺了它，一开 prefix caching 就会在图捕获阶段死掉，
而驱动报的是 `operation not permitted when a stream is capturing`——那是烟雾弹，
真正失败的是 host 侧 pinned 分配（`aclrtMallocHostWithCfg`，207001）。

脚本跑完会自己打印判据：

```
memlock: unlimited   (must be: unlimited)
xxhash <版本号>
```

`xxhash` 镜像里没有，脚本会装；不装的话 `--prefix-caching-hash-algo xxhash` 直接 500。

---

## 3. 路线 A：TP8 × PP1（生产配置）

prefix caching + ACL graph + MTP 投机解码三个特性同开。**不需要改 vllm-ascend 代码。**

```bash
docker exec glm53s bash -lc "bash /data02/<你的路径>/serve/serve.sh"
```

就绪判据是日志里的 `Application startup complete`。
**不要用 `/health` 判就绪**——上一个卡死的实例也会返回 200。

```bash
docker exec glm53s bash -lc "grep -q 'Application startup complete' /tmp/glm53_serve.log && echo ready"
```

验收：

```bash
docker exec glm53s bash -lc "python3 /data02/<你的路径>/test/smoke_test.py"
docker exec glm53s bash -lc "python3 /data02/<你的路径>/test/prefix_cache_test.py"
```

应当看到（首次提交时的实测值，`serve.sh` 的 `--max-model-len 16384` 配置下）：

```
KV cache        14.11 GiB / 337,547 tokens
图捕获          58 s / 0.42 GiB
prefix cache    5001-token prompt 1.89 s -> 0.83 s，命中 3840 = 6.00 × 640
MTP 接受率      73.0%（84/115 草稿 token，num_speculative_tokens=1）
正确性          17×23 -> 391
解码            1.01 s/tok（eager） -> 40.5 ms/tok（图 + MTP）
```

> `serve.sh` 的 `--max-model-len` 是 16384。想抬到 40k 以上，要先读第 4.2 节的
> KV 块记账问题——PP=1 同样有这个天花板（100,002 token 会被静默拒绝）。

---

## 4. 路线 B：TP4 × PP2（流水并行）

**两个前置缺一不可**，缺任何一个都会以"没有报错"的方式失败。

### 4.1 打补丁（必需）

```bash
docker exec glm53s bash -lc \
  "python3 /data02/<你的路径>/patches/pp_support.py --root /vllm-workspace/vllm-ascend/vllm_ascend"
```

预期输出：

```
  applied: patch_pp_mtp: resolve MTPModelTypes lazily
  applied: import make_empty_intermediate_tensors_factory
  applied: AscendGlm5NextModel.make_empty_intermediate_tensors
  applied: forward(): tolerate absent residual
  applied: forward(): never send None
  applied: AscendGlm5NextForCausalLM: forward the attribute
  applied: multimodal wrapper: forward the attribute
```

幂等（重复执行打印 `already applied`），回滚 `--revert`。

**不打会怎样**：服务照常接受 `--pipeline-parallel-size 2`，零警告，
只有非首 rank 在 `_dummy_run` 里炸 `AttributeError: 'NoneType' object has no attribute 'items'`，
而 `Worker_PP0` 一切正常甚至会报出 KV cache 大小——这个非对称就是指纹。

### 4.2 起服务（必须带 `--kv-cache-memory`）

```bash
docker exec glm53s bash -lc "bash /data02/<你的路径>/serve/serve-pp.sh"
```

脚本里已经写好了：

```
--kv-cache-memory=10831746048     # 图模式，10.09 GiB/rank
```

**这个值和编译模式绑定**：图模式 `10831746048`（10.09 GiB）、eager `11193525248`（10.42 GiB），
差的 0.34 GiB 就是图捕获占用。图模式下误用 eager 的值会在 `init_device` 阶段直接起不来。
换机器或换权重后，从启动日志里 `worker.py:758` 那行 `to fully utilize NPU free memory`
的建议值重新取。

**不给这个参数会怎样**：任何超过约 2758 token 的 prompt 会被调度器**永久拒绝**，
零输出、零报错、连一条 `Engine 000` 统计行都不打，客户端一直等到超时；
中止之后引擎自己恢复，下一条短请求 2.9 s 正常返回。根因见 docs/05。

就绪后日志里应有：

```
Setting GLM-5 logical attention block size to 1152 tokens ...
GPU KV cache size: 1,072,407 tokens
Maximum concurrency for 131,072 tokens per request: 8.18x
```

### 4.3 验收

```bash
docker exec glm53s bash -lc "python3 /data02/<你的路径>/test/pp_verify.py"
```

退出码 0 = 全过。预期（本次实测）：

```
1) 正确性        17×23 -> 391 PASS      capital of France -> Paris PASS
2) 并发          conc=4 × ~3000tok × 300out × temp1.0   4/4  20.0 s
                 conc=8 × ~7000tok × 600out × temp1.0   8/8  55.3 s
3) 长度阶梯      3,010 PASS 1.6s   8,008 PASS 2.6s   20,006 PASS 5.1s
                 60,004 PASS 14.5s  100,002 PASS 25.5s
4) 长上下文生成  in=17,633 out=120  9.5 s
VERIFY=ALL_PASS  失败项=无
```

引擎日志里 `Traceback` 应为 0 条，prefill 吞吐约 11,762 tokens/s。

### 4.4 前向数值等价（可选，但这是"PP 没改坏结果"的唯一硬判据）

用迷你模型即可，不占 8 卡：

```bash
python3 test/make_tiny_model.py /data02/glm53_tiny/model2
# 起 PP=1 服务（端口 8199）
PY=python3 GLM53_URL=http://127.0.0.1:8199 GLM53_MODEL=glm53tiny \
  bash test/pp_forward_equivalence.sh /tmp/pp1.json
# 换成 PP=2，其余参数完全一致，再来一次
PY=python3 GLM53_URL=http://127.0.0.1:8199 GLM53_MODEL=glm53tiny \
  bash test/pp_forward_equivalence.sh /tmp/pp2.json
```

比对：

```bash
python3 - <<'EOF'
import json
a=json.load(open('/tmp/pp1.json')); b=json.load(open('/tmp/pp2.json'))
print('max abs diff =', max(abs(x-y) for k in a
      for x,y in zip(a[k]['prompt_logprobs'], b[k]['prompt_logprobs'])))
EOF
```

预期 `0.000e+00`。**对照必须同特性**：拿 PP2 图模式去比 PP1 eager 会得到 3.178e-02，
那是图模式/APC 与 eager 的差，不是 PP 的。**不要用贪心续写文本判**，随机权重下近平局
（top1−top2 只差 0.003418）会被 ulp 级差异翻转。

### 4.5 排查脚本（复现"静默停住"用）

故意**不加** `--kv-cache-memory` 起服务，然后：

```bash
python3 test/pp_length_ladder.py --range fine   # 卡阈值：2660 过 / 2758 停
python3 test/pp_axis_matrix.py                  # 判定自变量：VERDICT=LENGTH_AXIS
```

想看调度器内部为什么拒，装两个诊断探针（**用完必须回滚**）：

```bash
docker exec glm53s bash -lc "
  python3 /path/patches/sched_debug.py --file /vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py
  python3 /path/patches/alloc_debug.py --file /vllm-workspace/vllm/vllm/v1/core/kv_cache_manager.py"
# 起服务时 export VLLM_SCHED_DEBUG=1
```

会打出：

```
SCHED_DEBUG BREAK allocate_slots_none req=... num_new=2304 lookahead=0
ALLOC_DEBUG fullISL num_tokens=2660 need=671 free=690 -> ok
ALLOC_DEBUG fullISL num_tokens=2758 need=696 free=690 -> REJECT
```

用完回滚：两个脚本都加 `--revert`。判据是 `grep -c SCHED_DEBUG <文件>` 返回 0。

---

## 5. 脚本一览

| 脚本 | 干什么 | 什么时候用 |
|---|---|---|
| `serve/docker-run.sh` | 起容器（`--ulimit memlock=-1` + xxhash） | 一次 |
| `serve/serve.sh` | TP8×PP1 生产启动（APC + 图 + MTP） | 路线 A |
| `serve/serve-pp.sh` | TP4×PP2 启动（**必须带 `--kv-cache-memory`**） | 路线 B |
| `patches/pp_support.py` | PP 必需的七处改动，幂等可 revert | 路线 B，起服务前 |
| `test/smoke_test.py` | 正确性 + prefix cache 命中 + MTP 接受率 | 路线 A 验收 |
| `test/prefix_cache_test.py` | 单独测 prefix cache | 路线 A |
| `test/pp_verify.py` | PP 验收：正确性 + 并发 + 长度 + 长上下文 | 路线 B 验收 |
| `test/pp_length_ladder.py` | 按 prompt 长度扫描找断崖 | 排查 |
| `test/pp_axis_matrix.py` | 三轴分离，判定停住的自变量 | 排查 |
| `test/pp_forward_equivalence.sh` | PP2 vs PP1 前向逐位对照 | 正确性判据 |
| `test/make_tiny_model.py` | 造结构完整的迷你模型（2 卡即可） | 不占 8 卡验通路 |
| `patches/sched_debug.py` / `alloc_debug.py` | 调度器/分配器诊断探针 | 排查，**用完回滚** |
| `test/mm_test.py` | 图像请求（当前必失败，见 docs/04） | 多模态 |

所有测试脚本都认 `GLM53_URL` / `GLM53_MODEL` 环境变量。

---

## 6. 故障指纹速查

| 你看到的 | 真正的原因 | 去哪看 |
|---|---|---|
| 图捕获阶段死，驱动说 `operation not permitted when a stream is capturing` | **不是**捕获模式问题，是 `--ulimit memlock=-1` 没给 | docs/02 |
| `ModuleNotFoundError: xxhash` / 请求 500 | 镜像没带 xxhash | docs/02 |
| `--pipeline-parallel-size 2` 零警告，只有非首 rank 在 `_dummy_run` 炸 | 没打 `patches/pp_support.py` | docs/05 |
| 长 prompt 永远不返回，**零输出零报错**，`Engine 000` 一条都没有，中止后引擎正常 | 没给 `--kv-cache-memory`；block 记账错账 | docs/05 |
| 图模式起服务时 `init_device` 报 free memory 不够 | 用了 eager 的 `--kv-cache-memory` 值 | 本页 4.2 |
| 停服后再起，报显存不足 | 上一批 worker 被 reparent 成孤儿还占着 HBM。多杀一轮 `VLLM::`，**以 `npu-smi` 为判据**，不要以 `docker ps` 为判据 | `serve*.sh` 注释 |
| 图像请求答非所问 | 厂商 chat template 拒图，见 docs/04；多模态默认关闭 | docs/04 |
| 用 `/health` 判就绪，却探到了旧进程 | 卡死的实例也答 200。**只认日志里的 `Application startup complete`** | 本页 3 |

---

## 7. 已知不在本仓库范围内

- **精度评测**（AIME 2026 等）没有随包：脚本和结果都不在本仓库，需要单独提供。
- **多模态图像通路**从未产出过正确答案，默认关闭（docs/04）。
- PP=1 加 `--kv-cache-memory` 能否吃下 100k **未实测**（PP=1 默认配置实测 100k 被拒）。
- 上游是否接受把 `AscendIndexerKPoolStateSpec` 的 `block_size` 改成 1152 —— 本仓库只给了绕过手段。
