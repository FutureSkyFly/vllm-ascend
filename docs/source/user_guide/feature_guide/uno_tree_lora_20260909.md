# UNO Tree：FP32 LoRA 投影合并与复现

2026-09-09，分支 `codex/uno-tree-performance`。保留版本为 **`001edc61ec821dde195eaafe39a7fa8befcfc510`**；硬件实测快照为 `5e00bd1f2b8ce769500bc5ea744666b4346b990d`。两个 commit 的完整 Git tree 相同：`93aee615405e20dd6526791ffd76c6c3ce3519fa`。本页记录已验证的阶段性基线，尚未完成 SGLang 全量性能对齐。

## 改动与结果

Tree 草稿把同一层的 LoRA A 投影合并，并缓存 FP32 A 与块对角 B，减少小矩阵乘法的次数。adapter 重载时原地刷新缓存，保持图回放地址稳定；base seed 行仍屏蔽 LoRA。Linear 和 AR 的计算路径保持原状。构树优化的前一轮证据保留在[构树记录](uno_tree_performance_20260909.md)。

| 模式 | 输出 token | token/s | TPF | 正确题数 | 截断 |
|---|---:|---:|---:|---:|---:|
| 同卡 AR | 26794 | 58.6028 | 1.0000 | 8/8 | 0 |
| Tree | 31261 | 111.7383 | 3.4642 | 8/8 | 0 |

Tree/AR 为 **1.9067×**。两种模式各自的全部输出 token IDs 均与此前对应模式一致。这是 MATH500 前 8 题、每种模式一次测量；Tree 测量期间卡 1 曾运行有界算子探针。正式配对必须停止所有探针后重新运行全量，不能将本页数字当作受控全量结论。

共同条件为 BF16、TP1、C1、FULL_DECODE_ONLY、上下文 40960、输出上限 32768、temperature=1、top-k=50、top-p=0.95、seed=42，关闭 prefix cache。Tree B/K/V=16/32/32，验证图 `[33]`、草稿图 16；AR 图 `[1]`。Tree 同步调度，AR 异步调度。计时包含整个离线 `generate` 及 prefill，排除启动和图捕获，没有请求预热。吞吐按实际输出 token/秒计算；TPF 按输出 token/(2×Tree 验证轮数)计算，不能代替收益比。

[SGLang 已合并参考](https://github.com/sgl-project/sglang/pull/37667)为 H200 完整 MATH500：AR183.12、Tree508.56 token/s，收益2.7772×、TPF3.450。本地尚未达到该收益口径。其 UNO 路径先 top-k 再 top-p，而[AR FlashInfer 路径](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/sampler.py)显式使用 joint；本实现两种模式均保留 joint，因此相同参数并不代表完全相同的采样行为。

## 验证与撤回记录

- 153 项 CPU 回归、21 项 NPU 回归通过；NPU 92.39秒。覆盖 adapter 重载、固定图地址、base 行隔离、图回放及原 Punica 路径。
- 完整模型两轮、两个提示、各32个 greedy token 稳定，并与 AR 逐 token 相同；随机采样短请求和上述长 MATH 子集均完成。
- 适用 `format.sh ci`、Ruff、logger、symbolic-meta、gitleaks 和函数长度检查通过。Windows C++ hook受系统策略阻止、shellcheck未安装；不称为完整跨平台CI全绿。
- 前一轮 replay 事件栅栏和 BF16 LoRA 收缩均未证实整模型收益，已撤回。
- 稀疏掩码候选虽通过短检查，但长 MATH 仅2/8正确，已撤回，根因尚未确认。后续独立缓冲区和真实祖先关系检查通过，不能据此忽略模型回归。该探针曾在 frontier=capacity 后计时，没有活跃树位置，相关单项提速不作为有效收益证据。

未验证的异步候选保存在私有实验快照中，不包含在本页保留代码 commit。

## 分支、服务器与运行时

```bash
UNO_SOURCE=/data01/uno-work/vllm-ascend-packed-reproduce
git clone --branch codex/uno-tree-performance \
  https://github.com/Liuchenbing-2026/vllm-ascend.git "$UNO_SOURCE"
git -C "$UNO_SOURCE" checkout --detach 001edc61ec821dde195eaafe39a7fa8befcfc510
```

服务器 q38 / `bms-75847414-002`，实测物理卡0，910B4-1。实测容器 `uno-align-v16`，容器 ID `d6d03b198bb2838bf5af173a472dfe5db728c319d83a8fb0709ab28e7fc68776`。结果根目录 `/data01/uno-work/sglang_alignment_20260908/v16`，容器内 `/alignment/v16`。源码以只读方式挂载到 `/vllm-workspace/vllm-ascend`。

沿用已验证镜像 `sha256:b936552e3c49668f1c9be8777d3d879e0bb6b2dc41f6417dedea84ed8f68eb9f`，vLLM commit `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`，Ascend上游基线 `748acedfea31b795e507f9c8175585f824328d8c`。Python3.12.13、torch2.10.0+cpu、torch_npu2.10.0.post4、vLLM0.27.1+empty。仅替换源码，保留运行时和二进制；完整容器启动命令、ABI SHA256及源文件清单见[机器可读证据](uno_tree_lora_20260909.json)与[原运行时准备方式](uno_reproduction.md#2-准备源码和原有运行时)。

模型主机路径 `/data01/models/uno-qwen3-8B`，容器模型 `/models/uno-qwen3-8B`，adapter位于其 `adapter` 子目录。模型、tokenizer、数据 revision 和文件哈希沿用[模型证据](uno_sglang_alignment.md#固定运行环境)，两组输入 IDs SHA256为 `2b5879f045881841fba7ce8a3fe783bb3a371c5d875678b41ff93602860f518b`。

## Bench 与服务端

复现 Linux 离线 bench 时使用[既有完整命令](uno_tree_performance_20260909.md#离线-bench-复现)，将代码固定为本页 commit，结果目录改为新的 `/alignment/tree-packed-lora-reproduce-new`，运行时记录指向自身容器。保持 Tree/AR 同卡、顺序执行，先检查端口、worker和设备，再进行短正确性与重复稳定性检查。删除 `--limit 8` 才是完整 MATH500。评分继续使用 Linux math-verify 0.9.0，完整实测 argv、计时和评分摘要包含在机器可读证据中。

HTTP 服务端的完整 `vllm serve` 与客户端命令见[服务端配方](uno_tree_performance_20260909.md#http-服务端与客户端)，端口8018、FULL_DECODE_ONLY及模型参数不变。必须确认设备空闲再启动。本轮没有执行HTTP压测，离线收益不能当作HTTP吞吐。

原始记录为 `v16/math500-{tree,ar}-c1-8/{plan.json,generations.jsonl,graded.jsonl,summary.json}`，另保留启动/稳定性结果、test日志、退出码、镜像/容器/ABI/源码清单及失败候选。所有Linux脚本使用LF、UTF-8无BOM，传输后校验SHA256和语法；源码归档来自精确commit的 `git archive`。
