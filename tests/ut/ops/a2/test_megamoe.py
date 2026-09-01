from types import SimpleNamespace

import pytest
import torch

from vllm_ascend import ascend_forward_context as afc
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe.moe_comm_method import _append_cann_megamoe_dummy_tokens
from vllm_ascend.utils import _warn_if_megamoe_shape_is_unfavourable, get_cann_megamoe_buffer_params


def test_dummy_routes_cover_all_experts_across_ep_ranks():
    routed_experts = []
    for ep_rank_id in range(4):
        hidden_states = torch.zeros((2, 4), dtype=torch.bfloat16)
        topk_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
        topk_weights = torch.full((2, 2), 0.5, dtype=torch.float32)
        active_mask = torch.tensor([1, 0], dtype=torch.int8)

        hidden_states, topk_ids, topk_weights, active_mask, original_num_tokens = _append_cann_megamoe_dummy_tokens(
            hidden_states,
            topk_ids,
            topk_weights,
            active_mask,
            num_experts=8,
            ep_rank_id=ep_rank_id,
            ep_world_size=4,
        )

        assert original_num_tokens == 2
        assert torch.equal(hidden_states[-1], torch.ones(4, dtype=torch.bfloat16))
        assert torch.equal(topk_weights[-1], torch.full((2,), 0.5))
        assert active_mask.tolist() == [1, 0, 1]
        routed_experts.extend(topk_ids[-1].tolist())

    assert sorted(routed_experts) == list(range(8))


def test_receive_bound_uses_documented_worst_case():
    assert get_cann_megamoe_buffer_params(480, 32, 256, 8) == (512, 8, 32, 131072)


def _make_a2_config(
    *,
    quantize: str,
    use_v2_model_runner: bool = False,
    hidden_size: int = 4096,
    moe_intermediate_size: int = 1536,
):
    model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            num_experts_per_tok=8,
            quantize=quantize,
        ),
        get_hidden_size=lambda: hidden_size,
        get_num_experts=lambda: 256,
    )
    return SimpleNamespace(
        model_config=model_config,
        quant_config=None,
        lora_config=None,
        use_v2_model_runner=use_v2_model_runner,
        parallel_config=SimpleNamespace(
            enable_expert_parallel=True,
            world_size_across_dp=8,
            pipeline_parallel_size=1,
            data_parallel_size=1,
        ),
    )


@pytest.mark.parametrize("quantize", ["w8a8_dynamic", "w4a8_dynamic"])
def test_a2_mode_1_selects_megamoe_for_supported_v1_config(monkeypatch, quantize):
    monkeypatch.setattr(afc, "is_mega_moe_supported", lambda: True)
    monkeypatch.setattr(afc, "is_moe_model", lambda _: True)
    monkeypatch.setattr(afc, "get_mc2_tokens_capacity", lambda: 4096)
    monkeypatch.setattr(afc, "get_ascend_device_type", lambda: afc.AscendDeviceType.A2)
    monkeypatch.setattr(afc, "get_ep_group", lambda: SimpleNamespace(world_size=8))
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(
            enable_fused_mc2=1,
            mega_moe_min_tokens=512,
            eplb_config=SimpleNamespace(dynamic_eplb=False),
        ),
    )

    assert afc.select_moe_comm_method(512, _make_a2_config(quantize=quantize)) == MoECommType.FUSED_MC2


@pytest.mark.parametrize(
    ("is_draft_model", "use_v2_model_runner"),
    [(True, False), (False, True)],
)
def test_a2_megamoe_falls_back_for_unvalidated_runner_paths(monkeypatch, is_draft_model, use_v2_model_runner):
    monkeypatch.setattr(afc, "is_mega_moe_supported", lambda: True)
    monkeypatch.setattr(afc, "is_moe_model", lambda _: True)
    monkeypatch.setattr(afc, "get_mc2_tokens_capacity", lambda: 4096)
    monkeypatch.setattr(afc, "get_ascend_device_type", lambda: afc.AscendDeviceType.A2)
    monkeypatch.setattr(afc, "get_ep_group", lambda: SimpleNamespace(world_size=8))
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(
            enable_fused_mc2=1,
            mega_moe_min_tokens=512,
            eplb_config=SimpleNamespace(dynamic_eplb=False),
        ),
    )

    result = afc.select_moe_comm_method(
        512,
        _make_a2_config(quantize="w4a8_dynamic", use_v2_model_runner=use_v2_model_runner),
        is_draft_model=is_draft_model,
    )
    assert result == MoECommType.ALLGATHER


def _patch_a2_megamoe_env(monkeypatch):
    monkeypatch.setattr(afc, "is_mega_moe_supported", lambda: True)
    monkeypatch.setattr(afc, "is_moe_model", lambda _: True)
    monkeypatch.setattr(afc, "get_mc2_tokens_capacity", lambda: 4096)
    monkeypatch.setattr(afc, "get_ascend_device_type", lambda: afc.AscendDeviceType.A2)
    monkeypatch.setattr(afc, "get_ep_group", lambda: SimpleNamespace(world_size=8))
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(
            enable_fused_mc2=1,
            mega_moe_min_tokens=512,
            eplb_config=SimpleNamespace(dynamic_eplb=False),
        ),
    )


@pytest.mark.parametrize(
    ("moe_intermediate_size", "expected"),
    [
        (512, MoECommType.FUSED_MC2),  # Qwen3.5/3.6-35B-A3B, documented lower bound
        (3072, MoECommType.FUSED_MC2),  # documented upper bound
        (256, MoECommType.ALLGATHER),  # below the documented range
        (3584, MoECommType.ALLGATHER),  # above the documented range
        (768, MoECommType.ALLGATHER),  # not a multiple of 512
    ],
)
def test_a2_megamoe_intermediate_hidden_range(monkeypatch, moe_intermediate_size, expected):
    _patch_a2_megamoe_env(monkeypatch)
    config = _make_a2_config(
        quantize="w8a8_dynamic",
        hidden_size=2048,
        moe_intermediate_size=moe_intermediate_size,
    )
    assert afc.select_moe_comm_method(512, config) == expected


@pytest.mark.parametrize(
    ("ep_world_size", "experts_per_rank", "warns"),
    [
        (8, 32, True),  # Qwen3.6-35B-A3B on one A2 node: measured ~3.7x slower than AllGather
        (8, 16, True),  # EP below the threshold upstream requires for the MC2 family
        (16, 32, True),  # too many experts per rank: ~14us fixed cost each, per layer
        (16, 24, False),  # both thresholds satisfied
        (32, 4, False),
    ],
)
def test_warns_on_unfavourable_megamoe_shape(caplog, ep_world_size, experts_per_rank, warns):
    from vllm_ascend import utils as ascend_utils

    # warning_once is lru_cached upstream; clear it so each parametrisation is
    # independent, but do not assume the attribute exists.
    getattr(ascend_utils.logger.warning_once, "cache_clear", lambda: None)()
    with caplog.at_level("WARNING"):
        _warn_if_megamoe_shape_is_unfavourable(ep_world_size, experts_per_rank)
    assert ("likely to be SLOWER" in caplog.text) is warns
