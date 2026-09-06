# 补丁目录

| 文件 | 用途 |
|---|---|
| `pp_support.py` | **PP 必需**。七处改动，不打 PP 根本起不来（`SupportsPP` 是 Protocol，非首 rank 会在 `_dummy_run` 里炸）。注意它只解决「起不来」；起来之后**还必须加 `--kv-cache-memory`**，否则 prompt 一过 ~2758 token 会被调度器静默拒绝。见 `docs/05-pipeline-parallel.md`。 |
| `pp_batch_depth1.py` | 判别实验用，**已证伪**（强制 batch_queue 深度 1 后照样停住）。不要用。 |
| `sched_debug.py` / `alloc_debug.py` | 诊断探针，`VLLM_SCHED_DEBUG=1` 才输出，幂等可 revert。 |
| 下文的补丁 A / 补丁 B | **不需要**，留作记录。 |

## 补丁 A / 补丁 B —— 不需要，留作记录

排查图捕获失败时，我按驱动报错的字面意思

```
Not_Supported(EE1016): ... The current thread is in the capture state and the current
operation cannot be performed ... This operation is supported only in the RELAXED mode.
rtMemcpy execution failed, reason=operation not permitted when a stream is capturing...
```

判断成「在 ACL graph 捕获期间分配 pinned host 内存是非法的」，改了 vllm-ascend 两处：

### 补丁 A —— `vllm_ascend/worker/block_table.py:283-288`

```python
def commit_block_table(self, num_reqs: int) -> None:
    self.block_table.gpu[:num_reqs].copy_(
        self.block_table.cpu[:num_reqs].clone().pin_memory(),   # 每次调用都新分配 pinned
        non_blocking=True,
    )
```
改成 `self.block_table.copy_to_gpu(num_reqs)`
（`CpuGpuBuffer.cpu` 构造时就已经是 pinned，`.clone().pin_memory()` 是多余的）。

### 补丁 B —— `vllm_ascend/worker/utils.py:15-18`

```python
def copy_snapshot_to_gpu(buffer: CpuGpuBuffer) -> torch.Tensor:
    cpu_snapshot = buffer.cpu.clone().pin_memory()
    return buffer.gpu.copy_(cpu_snapshot, non_blocking=True)
```
改成用一个挂在 buffer 上的持久 pinned 暂存区（保留 snapshot 语义，只分配一次）。
该 helper 有 15 处调用，含 `_dummy_run` 与 `_pad_query_start_loc_for_fia`。

---

## 为什么说不需要

做了单变量 A/B：

| 补丁 | memlock | 默认 PIECEWISE 捕获 |
|---|---|---|
| 打上 | unlimited | ✓ 40 s |
| **还原** | unlimited | **✓ 9 s** |
| 打上 | 64 KB (默认) | ✗ |
| 还原 | 64 KB (默认) | ✗ |

**决定性的只有 `--ulimit memlock=-1`。** 那条 EE1016 是驱动在
`aclrtMallocHostWithCfg` 失败时一并吐出的次要信息，真正的失败是撞了容器 64 KB 的
max-locked-memory rlimit。

## 补丁 B 仍有一个理论上的论点（未验证 [U]）

在**被捕获的图**里，host 源地址必须稳定 —— 图记录的是地址，重放时若该 pinned 缓冲区
已被回收，读到的就是垃圾。`.clone().pin_memory()` 每次给一个新地址，理论上不适合捕获路径。

但：CachingHostAllocator 会把释放推迟到拷贝完成，实际是否会出问题**没有证据**，
而且实测未打补丁一切正常（正确性、APC 命中、MTP 接受率都对）。
**在拿到反例之前不要上这两个补丁。**

## sched_debug.py / alloc_debug.py —— 调度器与 KV 分配器的诊断探针

两个都是**幂等、可 revert、受环境变量 `VLLM_SCHED_DEBUG=1` 控制**的临时探针，
默认不输出、不影响性能。用来回答「请求为什么排不进调度」这一类问题。

```bash
python3 sched_debug.py --file /vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py
python3 alloc_debug.py --file /vllm-workspace/vllm/vllm/v1/core/kv_cache_manager.py
# 起服务时 export VLLM_SCHED_DEBUG=1
# 用完务必回滚：
python3 sched_debug.py --file ... --revert
python3 alloc_debug.py --file ... --revert
```

打出来的行：

```
SCHED_DEBUG WAIT   req=... num_tokens=.. computed=.. num_new=.. budget=.. block_size=..
SCHED_DEBUG BREAK  mamba_align_zero | allocate_slots_none
SCHED_DEBUG RUN    skip zero ...
SCHED_DEBUG OUT    total=.. waiting=.. running=..
ALLOC_DEBUG fullISL num_tokens=.. need=.. free=.. -> ok|REJECT
ALLOC_DEBUG main    need_slot=.. need=.. avail=.. -> ok|REJECT
```

**这两个探针就是本次定位到 GLM-5 indexer 状态缓存块大小错账的工具**，
过程见 `docs/05-pipeline-parallel.md`。它们不是交付内容，仅作排查用。
