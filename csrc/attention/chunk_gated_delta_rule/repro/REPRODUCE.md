# Reproducing the arch22 chunk_gated_delta_rule results

Everything below was measured on 910B4-1 with the kernel source in this commit.
The harness in this directory is the harness that produced the numbers; nothing
was re-typed for the writeup.

## 1. What is being claimed

Bit-exactness, first: 36/36 output tensors compare equal under `torch.equal`
across the 18 shapes in `ab_gdr.py`. Both operator outputs are checked, `o` and
the final state `fs`. If this does not pass, nothing else matters.

Operator level, one process per shape, one card, one session, round-level ABBA,
median of 30, at Qwen3.6-35B-A3B GDN shapes (Dk=Dv=128, global Nk=16 / Nv=32):

| Config | Shape | Base (us) | Patch (us) | Delta | Within-arm spread |
|---|---|---|---|---|---|
| TP8 (Nk=2, Nv=4) | T=8192 B=1 | 1457.5 | 1225.6 | -15.9% | 0.5% / 0.6% |
| TP8 | T=8192 B=16 | 2162.4 | 988.6 | -54.3% | 1.6% / 1.1% |
| TP8 | T=2560 B=40 | 2511.4 | 371.6 | -85.2% | 0.5% / 2.4% |
| TP4 (Nk=4, Nv=8) | T=8192 B=1 | 1922.9 | 1661.8 | -13.6% | 0.3% / 2.0% |
| TP4 | T=8192 B=16 | 2743.6 | 1494.6 | -45.5% | 2.1% / 0.1% |
| TP2 (Nk=8, Nv=16) | T=8192 B=1 | 2618.3 | 2362.6 | -9.8% | 0.5% / 0.5% |
| TP1 (Nk=16, Nv=32) | T=8192 B=1 | 4740.9 | 4332.5 | -8.6% | 0.4% / 0.5% |

The single-batch rows carry roughly +-1pp. The gain grows with how many
sequences prefill in the same step, which is what changes 4 and 5 in the commit
message address.

End to end, Qwen3.6-35B-A3B-w8a8, TP4, `vllm bench serve`, four-arm ABBA with
two passes per server instance; random 1024-token prompts, 128 requests,
concurrency 32:

| Metric | Base | Patch | Delta |
|---|---|---|---|
| Median TTFT | 1444.4 ms | 1390.4 ms | -3.7% |
| Mean TTFT | 1401.7 ms | 1342.3 ms | -4.2% |
| P99 TTFT | 1826.7 ms | 1768.2 ms | -3.2% |
| Request throughput | 20.52 req/s | 21.66 req/s | +5.6% |
| Benchmark duration | 6.24 s | 5.91 s | -5.3% |

All eight median-TTFT cells are non-overlapping: patch max 1391.95 < base min
1436.26. Within a server instance the two passes differ by less than 0.05%.

A production-shaped run (4096- and 6144-token prompts, 256 output tokens, 160
requests, concurrency 32, `--max-num-batched-tokens 8192`) gains about 1%:
throughput +0.7 to +1.1%, end-to-end latency -1.0 to -1.2%, median TTFT -1.7%
on the shorter-prompt load with four non-overlapping cells. Two structural
reasons, both outside the operator: 256 output tokens put 89% of end-to-end
latency in decode, which runs the untouched recurrent operator; and prompts
that size against an 8192-token budget admit only one or two sequences per
prefill step, the operator's weakest regime.

## 2. Environment

| | |
|---|---|
| Device | 910B4-1 (20 AI Cube / 40 AI Vector cores, 192KB UB, 512KB L1) |
| CANN | 9.1.0 |
| torch_npu | 2.10.0.post4 (2.9.0.post4 does **not** have `npu_chunk_gated_delta_rule`) |
| vllm | 0.27.1, installed from a source distribution -- **not a git checkout, so there is no commit id** |
| vllm-ascend (runtime) | `5debbe58d1f0ed09621c0d427781de806ec04013`, plus 13 uncommitted local modifications |
| Kernel source | `ops-transformer`, branch `9.1.0`, `0684247` |

The operator is new in CANN 9.1.0. It is absent from 8.0.0, 8.3.RC1, 8.5.1,
9.0.0 and 9.0.1, verified three independent ways (`libopapi.so` symbols, the
`aclnnop` headers, and the opp kernel directory).

Two caveats on that table, both of which matter if you are trying to match the
numbers exactly rather than reproduce the direction:

The runtime vllm-ascend tree had 13 uncommitted modifications, so
`5debbe5` alone will not reconstruct it. One of them is a local switch added to
`vllm_ascend/ops/gdn.py` to force the Triton fallback for comparison; the rest
predate this work (megamoe, dsa_v1, sampler, cv_linear, model_runner_v1,
ascend_config, utils).

And the measurements did not go through this branch's build. See next section.

## 3. Two delivery paths, one kernel

The kernel reaches the device two different ways:

- **This branch** vendors the source into `csrc/attention/chunk_gated_delta_rule/`
  and builds it as part of vllm-ascend, which is what PR #12607 set up.
- **The measurements** built the same source into a CANN custom opp vendor
  package and installed it under `$ASCEND_OPP_PATH/vendors/`, where
  `torch_npu.npu_chunk_gated_delta_rule` picks it up through `load_priority`.

The second path was used because it toggles in about a second, which is what
makes a four-arm ABBA with package swaps practical at all. It also means the
runtime vllm-ascend version is irrelevant to the kernel under test: `5debbe5`
does not contain PR #12607's vendored kernel, it calls into CANN.

The two paths hand the compiler byte-identical source. That is the claim the
checksum chain below exists to support, and it is worth re-deriving rather than
taking on trust:

```
git show HEAD:csrc/attention/chunk_gated_delta_rule/op_kernel/arch22/chunk_gated_delta_rule_matmul_basic.h | md5sum
  -> 806d1f38...

pkg.run --noexec --extract=/tmp/x
find /tmp/x -name chunk_gated_delta_rule_matmul_basic.h | xargs md5sum
  -> 806d1f38...        # same bytes, and this is the package that was measured
```

Do this check. An earlier round of this work uploaded a patch but not a
newly-added file, so a stale copy survived on the build host and four
consecutive builds silently carried a variant that had supposedly been reverted
-- it compiled, it was bit-exact, the numbers looked good, and nothing anywhere
reported an error. Source-tree state is not evidence; extract the artifact and
hash it.

While you are extracting, `diff -rq` the base and patch packages. Here they
differed in exactly three files -- the header, its `.o`, and its `.json` --
which is what makes the A/B a single-variable experiment. That by-product is
worth more than the checksum.

## 4. Build and install the custom opp package

```bash
git clone -b 9.1.0 https://gitcode.com/cann/ops-transformer.git
cd ops-transformer
# copy the five files from csrc/attention/chunk_gated_delta_rule/op_kernel/arch22/
# in this branch over attention/chunk_gated_delta_rule/op_kernel/
# (this repo has no arch22/ subdirectory and no csrc/ prefix)

bash build.sh --pkg --soc=ascend910b --vendor_name=gdrcust --ops=chunk_gated_delta_rule -j64
./cann-ops-transformer-gdrcust_linux-aarch64.run --quiet
export LD_LIBRARY_PATH=$ASCEND_OPP_PATH/vendors/gdrcust_transformer/op_api/lib/:$LD_LIBRARY_PATH
```

The build takes about 2.5 minutes. Installing puts `gdrcust_transformer` first
in `$ASCEND_OPP_PATH/vendors/config.ini`'s `load_priority`.

Check that no other vendor in that list also provides this operator, or your
"base" arm is not the CANN built-in:

```bash
for d in $ASCEND_OPP_PATH/vendors/*/; do
  echo "$d: $(ls $d/op_impl/ai_core/tbe/kernel/ascend910b/ 2>/dev/null | tr '\n' ' ')"
done
```

## 5. Confirm the routing before trusting any number

```bash
GDR_OUT=/tmp/gdr GDR_DEV=0 SKIP_PERF=1 python ab_gdr.py check
```

`MAPS_PROBE` in the output lists what the process actually mapped. With the
package active it contains the vendor's `libcust_opapi.so`. Without that line
you are timing the built-in operator and will measure a difference of zero,
correctly.

## 6. Correctness

```bash
sed -i 's/^load_priority=gdrcust_transformer,/load_priority=/' $ASCEND_OPP_PATH/vendors/config.ini
GDR_OUT=/tmp/gdr GDR_DEV=0 SKIP_PERF=1 python ab_gdr.py base

./pkg.run --quiet
export LD_LIBRARY_PATH=$ASCEND_OPP_PATH/vendors/gdrcust_transformer/op_api/lib/:$LD_LIBRARY_PATH
GDR_OUT=/tmp/gdr GDR_DEV=0 SKIP_PERF=1 python ab_gdr.py patch

GDR_OUT=/tmp/gdr python cmp_gdr.py base patch
# expected: COMPARED 36 tensors / RESULT: PASS all bit-exact
```

Bit-exactness over a fixed shape set is a strong check on the arithmetic and a
weak one on ordering. It cannot rule out a timing-dependent race, which is why
change 3 keeps an explicit Fixpipe->MTE2 handshake rather than relying on the
Fixpipe unit flag: the variant that leans on the unit flag is also bit-exact on
all 18 shapes and about 1-2% faster, and it was still rejected.

## 7. Operator level

```bash
GDR_DEV=0 bash bench_op_abba.sh | tee op_abba.log
```

Report base as `mean(A1, A2)` and patch as `mean(B1, B2)` per shape, and report
the within-arm spread alongside. A delta smaller than the spread is not a
result -- on the T=2560 B=40 shape the two arms of one sweep came out +4.6% and
-2.7%, opposite signs, which is the correct way to discover you have no signal.

## 8. End to end

```bash
MODEL=/path/to/Qwen3.6-35B-A3B-w8a8 CARDS=4,5,6,7 bash bench_e2e_abba.sh | tee e2e_abba.log
```

Compare pass1 against pass1 and pass2 against pass2: there is a consistent
~3.5% warmup improvement from the first bench to the second within one server
instance, present in both arms.

## 8b. Production-shaped scenario

`bench_scenario_prod.sh` carries the deployment configuration verbatim -- the
server flags and both bench workloads (prefix 1229 + input 2867, and prefix 4301
+ input 1843, 256 output tokens, 160 requests, concurrency 32) -- wrapped in the
same four-arm ABBA.

```bash
MODEL=/path/to/model CARDS=4,5,6,7 bash bench_scenario_prod.sh | tee prod.log
```

Its header lists the four deviations from the deployment script, of which one
matters for interpreting the result: the runs here used the W8A8 checkpoint,
while the deployment uses the unquantized bf16 one. The operator is bf16 either
way and its absolute time is unchanged; W8A8 shrinks everything around it, which
raises the operator's share of prefill. The ~1% end-to-end gain this scenario
shows is therefore an upper bound for a bf16 deployment.

## 9. Things that produced wrong numbers here

**One shape per process.** Timing several shapes in one process inflated a
later small shape by 2.6x (626us read as 1631us). The inflation is not uniform:
the patched build, with smaller workspaces, was affected less, so a
same-process sweep exaggerates the win in the flattering direction.

**Cross-session absolute times are not comparable.** In two sessions the same
package read 1186.7us and 1196.0us for the same shape; base timings for TP4/TP2/
TP1 sat 3-4% apart between sessions. Only same-card same-session pairs mean
anything. A gain that "shrank" between writeups turned out to be a baseline
that moved, not a patch that regressed.

**TPOT is not a control under chunked prefill.** It is tempting to treat TPOT
as untouched -- decode runs a different operator -- and subtract it as a noise
floor. With `--enable-chunked-prefill` a scheduler step mixes prefill chunks
with decode tokens, so a decoding request's inter-token latency includes prefill
work done in the same step. TPOT improves genuinely when prefill gets faster.
Normalising by it deletes the result. There is no clean control in these runs;
the four-arm ABBA plus non-overlapping cells is the evidence.

**Killing the server does not free the cards.** Covered in the header of
`bench_e2e_abba.sh`. The criterion is `npu-smi`, not the process table.

**An orchestration script launched through `docker exec` outlives its parent.**
Killing the host-side waiter and the vllm processes left the inner loop running;
starting a second run then had two instances toggling the same global
`config.ini` and racing on the same port, each one's teardown killing the
other's server. None of that reports an error, it just corrupts the data. Count
instances (`pgrep -fc`), do not assume.

**`--additional-config` key placement is version-dependent.** `fuse_muls_add`
and `enable_npugraph_ex` belong inside `ascend_compilation_config`. At the top
level, an `AscendConfig` built with pydantic `extra="forbid"` refuses to start;
older versions without that setting accept the flat form and silently ignore
the keys, so the options are off and nothing says so.

## 10. Rollback

Remove `gdrcust_transformer` from `load_priority` in
`$ASCEND_OPP_PATH/vendors/config.ini`. The built-in operator is used again on
the next process start; nothing else is modified.
