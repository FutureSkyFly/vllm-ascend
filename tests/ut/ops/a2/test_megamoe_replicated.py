# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend import utils
from vllm_ascend.ascend_config import AscendConfig
from vllm_ascend.ops.fused_moe import moe_comm_method as comm
from vllm_ascend.ops.fused_moe.prepare_finalize import PrepareAndFinalizeWithReplicatedMegaMoe
from vllm_ascend.quantization.quant_type import QuantType


@pytest.mark.parametrize("rows", [1, 511, 512, 2048, 4095, 4096])
def test_full_input_and_local_partial_are_preserved_without_collectives(rows):
    prepare = object.__new__(PrepareAndFinalizeWithReplicatedMegaMoe)
    x = torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 4).to(torch.bfloat16)
    logits = torch.ones(rows, 8)
    result = prepare.prepare(x, logits)
    assert result.hidden_states is x and result.router_logits is logits
    assert result.mc2_mask is None and result.padded_hidden_states_shape == x.shape
    partial = x * 0.5
    assert prepare.finalize_tp_partial(partial, x.shape) is partial
    ids = torch.arange(rows)
    assert prepare.pad_and_split_input_ids(ids) is ids
    with pytest.raises(RuntimeError, match="deferred"):
        prepare.finalize(partial, True, x.shape)
    with pytest.raises(ValueError, match="full real-token"):
        prepare.finalize_tp_partial(partial[: rows // 2], x.shape)


@pytest.mark.parametrize(
    "replace,quant,dtype",
    [
        (True, QuantType.NONE, torch.bfloat16),
        (False, QuantType.W8A8, torch.bfloat16),
        (False, QuantType.NONE, torch.float16),
    ],
)
def test_prepare_rejects_shards_or_quantization(replace, quant, dtype):
    prepare = object.__new__(PrepareAndFinalizeWithReplicatedMegaMoe)
    with pytest.raises(ValueError, match="unsharded BF16"):
        prepare.prepare(torch.ones(2, 4, dtype=dtype), torch.ones(2, 8), replace, quant)


def test_replicated_capacity_is_explicit_and_default_bound_stays_unchanged():
    assert utils.get_cann_megamoe_buffer_params(2048, 2, 256, 8) == (2080, 128, 32, 33280)
    with pytest.raises(ValueError):
        utils.get_cann_megamoe_buffer_params(4096, 2, 256, 8)
    assert utils.get_cann_megamoe_buffer_params(4096, 2, 256, 8, replicated_input=True) == (4128, 128, 32, 33024)
    for capacity, ep, experts, topk in [
        (4097, 2, 256, 8),
        (0, 2, 256, 8),
        (4096, 4, 256, 8),
        (4096, 2, 128, 8),
        (4096, 2, 256, 4),
    ]:
        with pytest.raises(ValueError):
            utils.get_cann_megamoe_buffer_params(capacity, ep, experts, topk, replicated_input=True)


@pytest.mark.parametrize("tokens", [511, 2048, 4095, 4096])
def test_group_query_and_operator_allocation_use_identical_full_capacity(tokens, monkeypatch):
    import vllm.config

    config = SimpleNamespace(mega_moe_replicated_input=True)
    vc = SimpleNamespace(
        model_config=SimpleNamespace(
            get_num_experts=lambda: 256,
            hf_text_config=SimpleNamespace(num_experts_per_tok=8, hidden_size=2048),
        ),
        parallel_config=SimpleNamespace(world_size_across_dp=2, pipeline_parallel_size=1, tensor_parallel_size=2),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=tokens),
    )
    calls = []

    def query(*args, **kwargs):
        calls.append((args, kwargs))
        return 400

    monkeypatch.setattr(vllm.config, "get_current_vllm_config", lambda: vc)
    monkeypatch.setattr(utils, "get_ascend_config", lambda: config)
    monkeypatch.setattr(utils, "_load_cann_megamoe_ccl_buffer_size", lambda: query)
    monkeypatch.setattr(utils, "_warn_if_megamoe_shape_is_unfavourable", lambda *args: None)
    assert utils.calculate_cann_megamoe_hccl_buffer_size() == 400
    impl = object.__new__(comm.FusedMC2CommImpl)
    impl.token_dispatcher = object.__new__(comm.TokenDispatcherWithMC2)
    impl.token_dispatcher.global_bs = 0
    impl.token_dispatcher.ep_world_size = 2
    impl.token_dispatcher.max_num_tokens_per_rank = (tokens + 1) // 2
    impl.moe_config = SimpleNamespace(
        experts_per_token=8, num_experts=256, hidden_dim=2048, intermediate_size_per_partition=512
    )
    impl.get_symm_buffer_for_mega_moe = query
    monkeypatch.setattr(comm, "get_ascend_config", lambda: config)
    monkeypatch.setattr(comm, "get_mc2_group", lambda: SimpleNamespace(device_group="test_group"))
    monkeypatch.setattr(comm, "_is_a2_megamoe_enabled", lambda _: True)
    impl._init_mega_moe_symm_buffer()
    group_args, group_kwargs = calls[0]
    op_args, op_kwargs = calls[1]
    expected_capacity = (tokens + 1) // 2 * 2 + 32
    assert group_args[:4] == (2, 256, expected_capacity, 8)
    assert op_args == ("test_group", 256, expected_capacity, 8)
    for key, value in {
        "max_recv_token_num": 0,
        "dispatch_quant_mode": 0,
        "dispatch_quant_out_dtype": None,
        "comm_alg": "replicated_input",
    }.items():
        assert group_kwargs[key] == op_kwargs[key] == value


def test_config_default_does_not_change_existing_models():
    AscendConfig._validate_megamoe_replicated_input(SimpleNamespace(mega_moe_replicated_input=False), None)


@pytest.mark.parametrize(
    "invalid", [None, "tp", "dp", "sp", "quant", "dtype", "hidden", "capacity", "lora", "eplb", "disabled"]
)
def test_config_rejects_incompatible_replication_contract(invalid, monkeypatch):
    from vllm_ascend import ascend_config

    cfg = SimpleNamespace(
        mega_moe_replicated_input=True,
        enable_fused_mc2=1,
        enable_sp_by_pass=False,
        eplb_config=SimpleNamespace(dynamic_eplb=False),
    )
    pc = SimpleNamespace(
        enable_expert_parallel=True,
        tensor_parallel_size=2,
        data_parallel_size=1,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
    )
    hf = SimpleNamespace(hidden_size=2048, moe_intermediate_size=512, num_experts_per_tok=8)
    mc = SimpleNamespace(dtype=torch.bfloat16, hf_text_config=hf, get_num_experts=lambda: 256)
    vc = SimpleNamespace(
        parallel_config=pc,
        model_config=mc,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4096),
        quant_config=None,
        lora_config=None,
        use_v2_model_runner=False,
    )
    monkeypatch.setattr(utils, "get_ascend_device_type", lambda: utils.AscendDeviceType.A2)
    monkeypatch.setattr(ascend_config, "is_mega_moe_supported", lambda: True)
    if invalid == "tp":
        pc.tensor_parallel_size = 4
    elif invalid == "dp":
        pc.data_parallel_size = 2
    elif invalid == "sp":
        cfg.enable_sp_by_pass = True
    elif invalid == "quant":
        vc.quant_config = object()
    elif invalid == "dtype":
        mc.dtype = torch.float16
    elif invalid == "hidden":
        hf.hidden_size = 4096
    elif invalid == "capacity":
        vc.scheduler_config.max_num_batched_tokens = 4097
    elif invalid == "lora":
        vc.lora_config = object()
    elif invalid == "eplb":
        cfg.eplb_config.dynamic_eplb = True
    elif invalid == "disabled":
        cfg.enable_fused_mc2 = 0
    if invalid is None:
        AscendConfig._validate_megamoe_replicated_input(cfg, vc)
    else:
        with pytest.raises(ValueError, match="mega_moe_replicated_input requires"):
            AscendConfig._validate_megamoe_replicated_input(cfg, vc)


def test_only_fused_mc2_selects_replicated_prepare(monkeypatch):
    monkeypatch.setattr(comm, "get_ascend_config", lambda: SimpleNamespace(mega_moe_replicated_input=True))
    monkeypatch.setattr(comm, "PrepareAndFinalizeWithReplicatedMegaMoe", lambda cfg: "replicated")
    monkeypatch.setattr(comm, "PrepareAndFinalizeWithMC2", lambda cfg: "ordinary_mc2")
    fused = object.__new__(comm.FusedMC2CommImpl)
    ordinary = object.__new__(comm.MC2CommImpl)
    fused.moe_config = ordinary.moe_config = None
    assert fused._get_prepare_finalize() == "replicated"
    assert ordinary._get_prepare_finalize() == "ordinary_mc2"
