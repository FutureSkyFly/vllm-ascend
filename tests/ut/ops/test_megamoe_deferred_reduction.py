# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe import fused_moe as fused_moe_module
from vllm_ascend.ops.fused_moe import prepare_finalize as module
from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
from vllm_ascend.ops.fused_moe.prepare_finalize import PrepareAndFinalizeWithMC2
from vllm_ascend.ops.fused_moe.shared_experts import SharedExpertParallelMode


def make_finalize(rank, world, num_tokens):
    obj = PrepareAndFinalizeWithMC2.__new__(PrepareAndFinalizeWithMC2)
    obj.tp_rank, obj.tp_size = rank, world
    obj.num_tokens, obj.replace_allreduce = num_tokens, False
    return obj


@pytest.mark.parametrize("num_tokens,padded_tokens", [(1, 8), (13, 16), (17, 17), (512, 512)])
def test_token_partials_reconstruct_without_collective(num_tokens, padded_tokens, monkeypatch):
    collective = MagicMock(side_effect=AssertionError("ROUTED_ALL_GATHER_MUST_BE_DEFERRED"))
    monkeypatch.setattr(module.dist, "all_gather", collective)
    full = torch.arange(padded_tokens * 4, dtype=torch.float32).reshape(padded_tokens, 4)
    shards = torch.tensor_split(full, 8, dim=0)
    partials = [
        make_finalize(rank, 8, num_tokens).finalize_tp_partial(shard, full.shape) for rank, shard in enumerate(shards)
    ]
    torch.testing.assert_close(torch.stack(partials).sum(0), full[:num_tokens], rtol=0, atol=0)
    assert all(p.data_ptr() != full.data_ptr() for p in partials)
    collective.assert_not_called()


def test_deferred_reduction_rejects_sequence_shards():
    finalize = make_finalize(0, 8, 2)
    finalize.replace_allreduce = True
    with pytest.raises(AssertionError):
        finalize.finalize_tp_partial(torch.ones(2, 4), torch.Size([16, 4]))


def test_compiled_switch_uses_one_reduction_of_the_sum(monkeypatch):
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    runner = AscendMoERunner.__new__(AscendMoERunner)
    torch.nn.Module.__init__(runner)
    runner._a2_defer_tp_reduction = True
    runner.moe_config = SimpleNamespace(is_sequence_parallel=False)
    runner.ascend_shared_experts = SimpleNamespace(parallel_mode=lambda: SharedExpertParallelMode.TENSOR_PARALLEL)
    context = SimpleNamespace(moe_comm_type=MoECommType.FUSED_MC2)
    monkeypatch.setattr(fused_moe_module, "_EXTRA_CTX", context)
    reductions = []

    @torch.library.custom_op("megamoe_deferred_test::forward_shared", mutates_args=())
    def opaque_moe(shared: torch.Tensor, routed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # The leading dimension simulates all eight TP ranks in this CPU test.
        if context.moe_comm_type == MoECommType.ALLGATHER:
            return shared.clone(), routed.clone()
        full = routed.sum(0)
        shards = torch.tensor_split(full, 8, dim=0)
        partials = [
            make_finalize(rank, 8, full.shape[0]).finalize_tp_partial(shard, full.shape)
            for rank, shard in enumerate(shards)
        ]
        return shared.clone(), torch.stack(partials)

    @opaque_moe.register_fake
    def fake_moe(shared, routed):
        return torch.empty_like(shared), torch.empty_like(routed)

    @torch.library.custom_op("megamoe_deferred_test::all_reduce", mutates_args=())
    def all_reduce(value: torch.Tensor) -> torch.Tensor:
        reductions.append(value.numel())
        return value.sum(0, keepdim=True).expand_as(value).clone()

    @all_reduce.register_fake
    def fake_reduce(value):
        return torch.empty_like(value)

    monkeypatch.setattr(fused_moe_module, "tensor_model_parallel_all_reduce", all_reduce)
    graphs = []

    def backend(graph, example_inputs):
        graphs.append(graph)
        return graph.forward

    @torch.compile(backend=backend, fullgraph=True)
    def forward(shared, routed):
        shared, routed = opaque_moe(shared, routed)
        reduced = runner._fused_output_is_reduced
        shared = runner._maybe_reduce_shared_expert_output(shared, reduced)
        return runner._maybe_reduce_final_output(shared + routed, None, reduced)

    shared = torch.arange(8 * 16 * 4, dtype=torch.float32).reshape(8, 16, 4)
    routed = shared + 2
    expected = (shared.sum(0) + routed.sum(0)).expand_as(shared)
    for comm in (MoECommType.FUSED_MC2, MoECommType.ALLGATHER, MoECommType.MC2, MoECommType.FUSED_MC2):
        context.moe_comm_type = comm
        reductions.clear()
        torch.testing.assert_close(forward(shared, routed), expected, rtol=0, atol=0)
        assert reductions == [shared.numel()]
    assert len(graphs) == 1


def test_disabled_flag_keeps_existing_reduction_contract(monkeypatch):
    runner = AscendMoERunner.__new__(AscendMoERunner)
    torch.nn.Module.__init__(runner)
    runner._a2_defer_tp_reduction = False
    runner.moe_config = SimpleNamespace(is_sequence_parallel=False)
    context = SimpleNamespace(moe_comm_type=MoECommType.FUSED_MC2)
    monkeypatch.setattr(fused_moe_module, "_EXTRA_CTX", context)
    assert runner._fused_output_is_reduced
    context.moe_comm_type = MoECommType.ALLGATHER
    assert not runner._fused_output_is_reduced
