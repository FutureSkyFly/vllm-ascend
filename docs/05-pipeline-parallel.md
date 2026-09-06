# 05 —— 流水并行（PP）适配

证据强度：**[F]** 实测或读到代码并引用；**[I]** 推断；**[U]** 未验证。

## 结论

**PP 可用，但必须显式加大 KV 显存。** 两个彼此独立的问题，都已定位：

1. **起不来** —— 已修。`patches/pp_support.py` 七处改动。PP=2 能起、能算对，
   前向数值与 PP=1 **逐位相同**（`prompt_logprobs` 偏差 0.000e+00，eager 与图模式各做一次同配置对照）。
2. **起来之后，prompt 超过 ~2.7k token 就永远排不进调度** —— 已定位到具体一行，
   **加一个启动参数即可绕过**：

```
--kv-cache-memory=10831746048        # 图模式：10.09 GiB/rank
--kv-cache-memory=11193525248        # eager  ：10.42 GiB/rank
# 两者差的 0.34 GiB 就是图捕获占用。值来自启动日志 worker.py:758 那行的建议，
# 换机器/换权重后要重新取；图模式下误用 eager 的值会在 init_device 直接起不来。
```

| PP=2 (TP4×PP2) | KV 显存 | 块池 | 实测能吃多长的 prompt |
|---|---|---|---|
| eager，默认 `gpu-memory-utilization 0.9` | 4.85 GiB | 690 块 | **2,758 token 就被静默拒绝** |
| eager + `--kv-cache-memory=11193525248` | 10.42 GiB | **1487 块** | **100,002 token ✅（25.5 s）** |
| 图模式 + `--kv-cache-memory=10831746048` | 10.09 GiB | 未测（该轮已回滚探针） | **100,002 token ✅（25.5 s）**，KV 1,072,407 token |

根因是 indexer 压缩器状态缓存把自己的 KV 块大小报成了 `compress_ratio`（4 个 token），
详见下文「根因」一节。**PP=1 也有同一个缺陷**，只是天花板高（40k 通过、100k 被拒），
同一个参数同样适用。

**PP 与 MTP 不能同时开** —— 这是厂商刻意设的限制，不是 bug（见下）。

| 组合 | 状态 |
|---|---|
| PP=2 起服务 / 权重加载 / 数值正确性 | ✅ 与 PP=1 逐位相同（eager 与图模式各自对照） |
| PP=2 + 默认 KV 显存 + 长 prompt | ❌ 调度器永远排不进，零输出、零报错 |
| **PP=2 + `--kv-cache-memory`** | ✅ **100k token prompt 正常；并发 8 × 7k prompt × 600 out 也正常** |
| PP=2 + MTP | ❌ 被 `platform.py:443` 显式拒绝，仅 PD 分离的 P 节点放行 |

## 为什么不打补丁会失败，且**没有任何警告**

`AscendGlm5NextForCausalLM` 声明了 `SupportsPP`（`glm5_next.py:2437`），却**从未定义
`make_empty_intermediate_tensors`**（全文 2532 行 grep 零命中）。

关键在于 `SupportsPP` 是 **Protocol**，方法体是 `...`，被继承后就是一个
**真实存在、返回 `None` 的方法**（`interfaces.py:628-634`）。于是：

```
supports_pp() → True
  → ModelConfig 接受 --pipeline-parallel-size>1   (config/model.py:1201-1208)
  → 零警告
  → 非首 rank 在 _dummy_run 里炸
```

实测的错误（迷你模型 TP1×PP2，`/data02/glm53_tiny/logs/pp_nopatch.log`）[F]：

```
File "/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py", line 3601, in _dummy_run
    {k: v[:intermediate_tokens] for k, v in self.intermediate_tensors.items()}
AttributeError: 'NoneType' object has no attribute 'items'
```

**只有 `Worker_PP1` 死；`Worker_PP0` 一切正常**，甚至报出了
`Available KV cache memory: 15.27 GiB`。这个非对称正是「非首 rank 才需要接收 buffer」的指纹。

多模态包装类还叠了一层：`AscendGlm5NextForConditionalGeneration.__init__` 调用
`super(Glm4vForConditionalGeneration, self).__init__()`（`glm5_next_multimodal.py:665`），
**故意跳过**父类 `__init__`，因此也跳过了那行
`self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors`
（`glm4_1v.py:1655-1657`）。

## 光补上方法不够 —— GLM-5.3 特有的两个坑

### ① PP 边界的张量是**三维**的

超连接（`mhc=True`, `mhc_num_residual_streams=4`）让残差流在层间是 4 倍宽的：

| 位置 | 形状 | 代码 |
|---|---|---|
| layer 0 展开 | `[T, 4096]` → `[T, 4, 4096]` | `glm5_next.py:2213-2215` (`_expand_mhc_residual_streams`) |
| **中间各层（PP 边界落在这里）** | **`[T, 4, 4096]`** | `hc_post` 返回 `[T,n,d]` |
| 最后一层坍缩 | `.mean(dim=1)` → `[T, 4096]` | `glm5_next.py:2237-2239` |

而 `make_empty_intermediate_tensors_factory` 只会分配**二维** `[batch, hidden_size]`
（`vllm/model_executor/models/utils.py:700-713`）—— 接收端 buffer 形状对不上。

补丁按 `config.mhc` 分支装配：开 → 自定义工厂分配 `[batch, n, hidden]`；关 → 用标准工厂。

### ② mHC 路径下 `residual` 恒为 `None`

`glm5_next.py:2240`：`return hidden_states, None`

但模型无条件把它塞进 `IntermediateTensors`（`:2316`），接收端又无条件读 `["residual"]`（`:2309`）。
PP 传输会遍历这个 dict 逐个 `isend`，塞 `None` 直接炸。

补丁：发送端有 residual 才放键，接收端用 `.tensors.get("residual")`。

## PP + MTP：被刻意禁止

`vllm_ascend/platform.py:430-447` [F]：

```python
@classmethod
def _validate_pd_pp_mtp_config(cls, vllm_config) -> None:
    if not cls._is_mtp_speculative_config(speculative_config): return
    if parallel_config.pipeline_parallel_size <= 1: return
    if kv_transfer_config is not None and kv_transfer_config.kv_role == "kv_producer": return
    raise ValueError(
        "PP+MTP is only supported on PD-disaggregated P nodes "
        "(kv_role='kv_producer'). D nodes must use "
        "pipeline_parallel_size=1 and may combine data parallelism with MTP."
    )
```

原因写在 `patch/platform/patch_pp_mtp.py` 的 docstring 里：PP 的 batch_queue 会让 EngineCore
在消费旧输出之前就调度新批次，drafter 从 `post_step` 更新 `request.spec_token_ids` 时
会读到**新调度步**的 Request 状态。P 节点不做 decode，所以没有这个竞争。

**部署取舍**：单节点混合部署里，PP 和 MTP 二选一。实测 MTP 在 TP8 上接受率 73.0%，
而 PP 的价值主要在跨节点扩展 —— 单机 8 卡没有理由用 PP 换掉 MTP。

### 附带修掉的一个死代码 bug

`patch_pp_mtp.py` 本该让 MTP drafter 以 PP=1 通过校验，但它在**打补丁时**就把
`MTPModelTypes` 快照进闭包了：

```python
mtp_model_types = set(get_args(MTPModelTypes))   # 导入时求值
```

而导入顺序是（`patch/platform/__init__.py`）：

| 行 | 模块 | 动作 |
|---|---|---|
| **23** | `patch_pp_mtp` | 快照 `MTPModelTypes`（此时只有 vLLM 原生 18 项） |
| **53** | `patch_speculative_config` | 才把 `"glm5_next_mtp"` 追加进去（`:148-152`） |

所以 `is_mtp_drafter` 对 GLM-5.3-Flash **恒为 False，整个补丁是死代码**。
实测错误：`NotImplementedError: Pipeline parallelism is not supported for this model.`
且堆栈停在 `patch_pp_mtp.py:70`（fall-through 分支），不是 `:69`（打过补丁的分支）—— 这就是判据 [F]。

`patches/pp_support.py` 把它改成调用时惰性读取，不再依赖导入顺序。
修好之后才暴露出上面那条真正的 PD 限制。

## 正确性判据：为什么不能看输出文本

迷你模型的权重是随机的（从真 checkpoint 的 safetensors **header** 反解张量名/形状/dtype 造的），
**PP=1 下也是乱码**，所以输出文本不能用来判正确性。

判据用 **`prompt_logprobs`**（纯前向，不涉及采样）：

| 对照 | 5 个 prompt 的最大逐元素偏差 |
|---|---|
| PP2 eager vs PP1 eager | **0.000e+00** |
| PP2 图+APC vs PP1 图+APC | **0.000e+00** |
| PP2 图+APC vs PP1 **eager** | 3.178e-02 ← 三个变量同时变，**不能归因给 PP** |

第三行是个陷阱：如果拿它当判据，会误判 PP 破坏了数值。加上同配置的 PP=1 对照才看清
偏差全部来自「图模式/APC vs eager」。

贪心续写会分叉（5 条里 1 条在共同前缀 7 个 token 后分开），原因是随机权重下
**top1−top2 只差 0.003418**，近平局被 ulp 级差异翻转。**别拿贪心续写当 PP 的判据。**

## 实测数据（迷你模型，TP1×PP2，2 卡）

```
层切分       8 层 → PP0: 0-3, PP1: 4-7
权重         PP0 4.28 GB / PP1 5.83 GB   (PP0 含 embed+ViT，PP1 含 lm_head)
KV           670,666 tokens
图捕获       41 s
逻辑块大小   4352 tokens   ← TP1；与 TP8=640 / TP4=1152 / TP2=2176 的倍增规律吻合
prefix cache 2601-token prompt：冷 5.24 s → 热 1.11 s（4.71×）
```

`Setting GLM-5 logical attention block size to 4352 tokens`（`patch_mamba_config.py:154`）
说明 block size 推导在 PP 下仍走全局路径，不同 PP rank 不会算出不同的值。

## 已确认**不是**问题的（省得再查）

| 项 | 事实 | 出处 |
|---|---|---|
| vllm-ascend 的 PP 传输 | 真实现：NPU worker 有 IntermediateTensors 的异步 isend/irecv，ACL 图捕获的 `weak_ref_tensors` 支持 IntermediateTensors | [F] 代码 + 实测 |
| 混合 KV / APC 的 block size | 在 EngineCore 里对所有 worker 的合并 spec **全局算一次**再按 rank 投影，不同 PP rank 不可能算出不同值 | `kv_cache_utils.py:1994-2011, 1914-1953` [F] |
| w8a8 权重加载 | 113k 条 quant description 是全局字典精确查键，无连续性假设；`is_pp_missing_parameter` 在所有路径（含专家路径）正确丢弃非本地层 | [F] |
| 各 attention 层的 kv_cache | 已经是 `[torch.tensor([]) for _ in range(pipeline_parallel_size)]`，本来就 PP 感知 | `glm5_next.py:288,345` [F] |
| vLLM 0.23.0 核心 | 没有 PP-vs-prefix-caching / chunked-prefill / hybrid-mamba-KV / 多模态 的任何 raise | [F] |

## 已知但未修

**视觉塔在每个 PP rank 都会构建**（`glm5_next_multimodal.py:679-686`，无 `is_first_rank` 保护）。
实测两个 rank 都打印了 `Using ... for vit attention` [F]。
403M 参数 × 2 字节 ≈ **每 rank 浪费 0.8 GiB**，但不影响正确性。
加保护会牵动权重加载路径（`AutoWeightsLoader` 对 `PPMissingLayer` 的处理），建议单独提交。

## 真权重 TP4×PP2：能起来、算得对，但**长 prompt 会停住**

### 起来了，而且算对了 [F]

```
rank 拓扑     Worker_PP0_TP0..TP3 + Worker_PP1_TP0..TP3   （TP4 × PP2 × EP 三层都建起）
权重/rank     PP0 36.3955 GB / PP1 39.6380 GB   （PP1 多 lm_head）
KV            8.09 GiB / 514,606 tokens
逻辑块大小    1152 tokens        ← 正是 TP4 应有的值，说明 PP 下仍走全局推导
图捕获        63 s（PIECEWISE + FULL decode 都过）
正确性        17×23 → 391 ✓ ；capital of France → Paris ✓
```

### 但**prompt 一长就永远排不进调度** [F]

**唯一的自变量是 prompt 长度。** 同一台机器、同一份权重、同一条请求：

| 配置 | prompt 2310 token | prompt 2814 token |
|---|---|---|
| **TP8 × PP1** | PASS 0.9 s | **PASS 1.1 s**（4004 token 也只要 1.4 s） |
| **TP4 × PP2** eager + APC | PASS 1.0 s | **STALL，零输出，客户端超时** |
| TP4 × PP2 eager **关 APC** | PASS 0.8 s | STALL |
| TP4 × PP2 **图模式** + APC | PASS 0.9 s | STALL |
| TP4 × PP2 eager **预算 16384** | PASS 1.0 s | STALL |

是**断崖**不是曲线：2310 之前时间平坦在 0.5–1.0 s，2814 直接 >150 s 零输出。
阈值对 prefix cache、图模式、`max-num-batched-tokens` 三者**完全不敏感**，只随 PP 变。

### 停住时引擎在干什么 [F]

```
EngineCore  35 次采样里 25 次落在 core.py:1279
            → if not model_executed and self.scheduler.has_requests(): time.sleep(0.001)
            走到这一行在控制流上就要求 model_executed == False，
            而 is_ec_consumer 在无 EC 配置时恒为 True（core.py:200-203），
            所以 model_executed 就是 scheduler_output.total_num_scheduled_tokens > 0
            ⇒ **调度器有请求，但每一步都调度出 0 个 token**

8 个 worker  全部 idle 在 shm_broadcast.dequeue —— 从未收到过这个 batch
NPU          AICore 0%
日志         整个停住窗口内 `Engine 000` 统计行**一条都没有**（统计行随输出打，
             零行 = 零输出）；`Added request` 有 4 条，说明请求确实进了引擎
```

**不是死锁。** 客户端超时中止后，引擎立刻恢复，下一条短请求 2.9 s 正常返回。

### 三轴分离实验（每格之后打短健康探针）[F]

| 格 | prompt | 出 | temp | TP4×PP2 | TP8×PP1 |
|---|---|---|---|---|---|
| A | 20 | 16 | 0 | PASS 7.9 s | PASS 14.1 s |
| B | 20 | 300 | 0 | PASS 38.6 s | PASS 40.4 s |
| C | 20 | 300 | 1.0 | PASS 40.3 s | PASS 41.7 s |
| D | **2769** | 16 | 0 | **STALL** | PASS 6.6 s |
| E | **2769** | 16 | 1.0 | **STALL** | PASS 6.3 s |
| F | **2769** | 300 | 0 | **STALL** | PASS 115.4 s |
| G | **2769** | 300 | 1.0 | **STALL** | PASS 117.1 s |

输出长度、温度/top-p 都不是变量；每格后的短探针一律 OK，说明引擎状态没被前一格弄坏。

## 端到端验证（生产同款配置）[F]

`TP4 × PP2 + 图模式 + prefix cache + --kv-cache-memory=10831746048`，一次干净启动跑完：

```
KV                1,072,407 token（Maximum concurrency 8.18x）

1) 正确性          17×23 → 391 ✓          capital of France → Paris ✓
2) 原始并发复现    conc=4 × ~3000tok × 300out × temp1.0   4/4  20.0 s   ← 此前必挂
                   conc=8 × ~7000tok × 600out × temp1.0   8/8  55.3 s   ← 此前从没跑到过
3) 长度阶梯        3,010 ✓1.6s   8,008 ✓2.6s   20,006 ✓5.1s
                   60,004 ✓14.5s   100,002 ✓25.5s
4) 长上下文生成    in=17,633 → out=120，9.5 s，答案切题
   prefill 吞吐    11,762 tokens/s

Traceback 0 条
```

验收脚本 `test/pp_verify.py`（退出码 0 = 全过）。

配套的排查脚本：

| 脚本 | 用途 |
|---|---|
| `test/pp_verify.py` | 验收：正确性 + 并发 + 长度 + 长上下文生成，一次跑完 |
| `test/pp_length_ladder.py` | 按 prompt 长度扫描，找断崖位置（`--range coarse\|fine\|high`） |
| `test/pp_axis_matrix.py` | 三轴分离（prompt 长度 × 输出长度 × 温度），判定是哪一根轴 |
| `test/pp_forward_equivalence.sh` | PP2 vs PP1 前向逐位对照（`prompt_logprobs`） |

## 根因：indexer 状态缓存把自己的块大小报成了 `compress_ratio`

### 证据链 [F]

给 `Scheduler.schedule()` 和 `KVCacheManager.allocate_slots()` 各打了一个受
`VLLM_SCHED_DEBUG` 控制的探针（`patches/sched_debug.py`、`patches/alloc_debug.py`，幂等可 revert），
在会卡的那台服务上直接读出：

```
SCHED_DEBUG BREAK allocate_slots_none req=... num_new=2304 lookahead=0
ALLOC_DEBUG fullISL num_tokens=2660 full=2660 need=671 free=690 -> ok
ALLOC_DEBUG fullISL num_tokens=2758 full=2758 need=696 free=690 -> REJECT
```

`allocate_slots` 返回 `None` → 调度器在 `scheduler.py:781` `break` → 请求永远留在等待队列 →
`schedule()` 每步返回 0 个 token → 引擎在 `core.py:1279` 以 1 ms 为周期空转。**全程零报错。**

关掉 full-ISL 准入门（`--no-scheduler-reserve-full-isl`）后，第二个检查点把数字暴露得更干净：

```
ALLOC_DEBUG main need_slot=1152 need=292 avail=690 -> ok
ALLOC_DEBUG main need_slot=2304 need=581 avail=690 -> ok      差 289
ALLOC_DEBUG main need_slot=3456 need=870 avail=690 -> REJECT  差 289
```

**1152 个 token 要 289 块 ⇒ 每块 4 个 token。**

### 代码位置

`vllm_ascend/models/glm5_next.py:339-362`，`AscendIndexerKPoolCompressorState`：

```python
self.compress_ratio = compress_ratio
self.sliding_window = compress_ratio
self.block_size     = compress_ratio        # ← 4
...
def get_kv_cache_spec(self, vllm_config):
    return AscendIndexerKPoolStateSpec(
        block_size=self.block_size,          # 以 token 计的 KV 块大小 = 4
        ...
    )
```

同一个文件里，真正的 indexer K cache（`:295-302`）用的是
`block_size=self.cache_config.block_size`（1152），**是对的**；出问题的只有这个**压缩器状态**组。

`KVCacheCoordinator.get_num_blocks_to_allocate`（`kv_cache_coordinator.py:162-184`）
把各组的 `cdiv(num_tokens, block_size)` **逐组相加**，而所有组共用同一个物理块池
（池子按 1152 token 的 MLA/KDA 大页尺寸切）。于是这一个组独自要走 `num_tokens/4` 块
——**在准入上限以内是这个斜率**，上限之上会被压平（见下）。

### 为什么表现成「PP 特有」

`need` 会被准入上限（`apply_admission_cap`）压平在 **约 1035 块**。所以能不能跑，取决于
**块池大小和这个上限谁大**：

| 配置 | 末段 rank 的 KV 显存 | 块池 | vs 上限 1035 | 实测能吃多长的 prompt |
|---|---|---|---|---|
| TP8 × PP1 | 9.13 GiB（各 rank 对称） | **1178 块** | 池 > 上限 | 40,012 token ✅（11.9 s）；100,002 token ❌ |
| TP4 × PP2 | **4.85 GiB**（PP1 段多带 lm_head，PP0 段有 8.09 GiB） | **690 块** | **池 < 上限** | **2,758 token 就 ❌** |

PP 并没有引入这个错账，它只是把块池砍掉一半，**掉到准入上限以下**，于是阈值从
「10 万 token」直接塌到「2.7k token」。

**注意：PP=1 也有同一个缺陷**，只是天花板高。一个宣称 `max-model-len 131072` 的服务，
在 100k token 的 prompt 上会以完全相同的方式静默停住（`need=1186 > free=1178`）。

## 被证伪的假设（含我自己走过的弯路）

| 假设 | 判据 | 结论 |
|---|---|---|
| **并发**（"并发 4 才挂，串行没事"） | conc=1 单条长 prompt 同样 stall | **证伪**。原始"串行通过"的对照全部用的是短输出/短 prompt |
| **冷启动第一批**（"一个串行请求热身就能预防"） | 该对照同时把 prompt 结构从「独立前缀」换成「2800 token 公共前缀」，后 3 条几乎全命中 prefix cache（日志里 67.1%）。同参数重测即 stall | **证伪，且是我自己制造的混淆** |
| PP 的 batch_queue（`step_with_batch_queue`） | 环境变量强制 `max_concurrent_batches=1`，照样 stall | 证伪 |
| ACL 图模式 | `--enforce-eager` 与图模式阈值完全相同 | 证伪 |
| prefix caching | `--no-enable-prefix-caching` 阈值不变 | 证伪 |
| `max-num-batched-tokens` | 4096 → 16384 阈值不变 | 证伪 |
| 温度 / top-p 采样算子 | temp 0 与 temp 1.0+top_p 0.95 同结果 | 证伪 |
| 输出长度 | 出 16 与出 300 同结果 | 证伪 |
| 请求没到引擎 | `--enable-log-requests` 显示 4 条 `Added request`（打在 `await add_request_async` 之后） | 证伪 |
| 补丁少发 `residual` 键导致 PP 收发错位 | `irecv_tensor_dict` 按**收到的元数据**重建 dict（`parallel_state.py:1092`），不按预分配 buffer 的键 | 代码层证伪 |
| `indexer_kpool_topk_pytorch` / `_sparse_attention_pytorch` 里的 16-query Python 循环 | 全树 grep 无调用点，是死代码 | 证伪 |
| 上游 `next_decode_eligible_step` 的 PP 节流 | 只在 `use_v2_model_runner` 下设置，我们跑 V1 runner | 证伪 |
| `patch_balance_schedule` 的调度器 | `VLLM_ASCEND_BALANCE_SCHEDULING` 默认 0，`schedule()` 只是转调父类 | 证伪 |

### 我自己被推翻的三条读数（写下来免得重犯）

1. **「worker 79–91% CPU = 在自旋等死锁」** —— 错。`shm_broadcast.dequeue` 本来就是忙等，那是正常空闲态。
2. **「EngineCore 阻塞在 `queue.get`，引擎真空闲」** —— 那是**中止之后**的状态。停住期间它在 `core.py:1279` 空转。
3. **「请求没进引擎，卡在前端」** —— 错。`Added request` 四条俱全。

## 结论：PP 可用，但必须带 `--kv-cache-memory`

补丁解决的是「PP 起不来」；KV 显存参数解决的是「起来之后长 prompt 排不进」。两件事互相独立。

**部署建议**：单机 8 卡仍然没有理由用 PP —— 它会换掉 MTP（实测接受率 73.0%），
而 PP 的价值在跨节点扩展。真要用 PP，务必：

```bash
--kv-cache-memory=10831746048     # 图模式；eager 用 11193525248
                                  # 否则默认配置下 prompt 一过 2.7k token 就静默停住
```

**并且应当把根因报给上游**：`AscendIndexerKPoolCompressorState.block_size = compress_ratio`
让这一个缓存组按「4 个 token 一块」向共享块池要块，`--kv-cache-memory` 只是把池子撑大到
准入上限之上，并没有修掉错账本身。

## 未验证 [U]

- 上游是否接受「`AscendIndexerKPoolStateSpec` 的 `block_size` 应当是 1152 而不是 4」这个修法
  （本轮只验证了绕过手段，没有改这一行重测）。
- PP=1 + `--kv-cache-memory` 是否也能吃下 100k（PP=1 默认配置实测 100k 被拒，未回测加参数后的情况）。
- PP > 2（PP=4/8 的层切分与显存分布）。
- PP 下的多模态图像通路（文本通路已验；图像通路在 TP8 下本来就没跑通过）。
- 迷你模型 TP1×PP2 是否同样有这个阈值（此前所有迷你模型测试的 prompt 都很短）。
