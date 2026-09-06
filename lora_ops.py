#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LoRA bgmv/sgmv ops: Triton kernels dispatched through torch custom ops.

Importing lora_ops_triton registers the kernels in the ``vllm_ascend_triton``
namespace (torch.library.custom_op), so torch._dynamo treats them like the
stock ``torch.ops._C_ascend.*`` ops.  All runtime checks and AscendC fallbacks
live inside the eager impls; the wrappers below keep the stock
vllm_ascend.lora.lora_ops API and stay free of data-dependent Python branches
so the serving path can be traced with fullgraph=True.
"""
import os

import torch

from vllm_ascend.lora import lora_ops_triton  # noqa: F401  (registers custom ops)

# --- route selection ---------------------------------------------------------
# The triton kernels only beat the AscendC ops at the smallest decode batches.
# Measured per-layer device time on 910B4 (v2 exact config / AscendC):
#     B          1      2      3      4      8     16
#   shrink    0.53x  0.62x  1.07x  1.07x  1.07x  1.06x
#   expand    0.94x  1.10x  1.45x  1.38x  1.90x  2.58x
# Past the crossover, routing through torch.ops.vllm_ascend_triton.* -- a
# torch.library.custom_op whose implementation is a PYTHON function -- costs a
# measured +43.8 us (shrink) / +44.7 us (expand) per call versus calling
# torch.ops._C_ascend.* directly (71.2 vs 27.4 us; 2.60x stock).  For a
# 48-layer model that is ~+23.4 ms of host time per forward, which lands
# straight on TTFT in eager prefill and gives back far more than the kernels
# ever won.  So decide the route HERE, above the custom-op boundary: past the
# threshold this file issues exactly the call stock lora_ops.py issues.
#
# The branch tests inputs.shape[0] -- a shape, not a data value.  Under aclgraph
# every capture size gets its own graph and B is fixed within a capture, so the
# route is baked correctly per graph; batches beyond max_cudagraph_capture_size
# run eager and re-evaluate it per call.
#
# TRITON_LORA_DISABLE=1 forces the stock call unconditionally -- use it where
# the AscendC ops always win (e.g. 910C at high concurrency), and this module
# then costs nothing versus stock.
_SHRINK_MAX_B = int(os.environ.get("TRITON_LORA_SHRINK_MAX_B", "2"))
_EXPAND_MAX_B = int(os.environ.get("TRITON_LORA_EXPAND_MAX_B", "1"))
if os.environ.get("TRITON_LORA_DISABLE", "0") != "0":
    _SHRINK_MAX_B = _EXPAND_MAX_B = -1


def bgmv_shrink(inputs, lora_a_weights, output_tensor, lora_indices_tensor, scaling=1.0):
    if inputs.shape[0] > _SHRINK_MAX_B:
        torch.ops._C_ascend.bgmv_shrink(
            inputs, lora_a_weights, lora_indices_tensor, output_tensor, scaling)
        return output_tensor
    torch.ops.vllm_ascend_triton.bgmv_shrink(
        inputs, lora_a_weights, output_tensor, lora_indices_tensor, scaling)
    return output_tensor


def bgmv_expand(inputs, lora_b_weights, output_tensor, lora_indices_tensor, add_inputs=True):
    return bgmv_expand_slice(inputs, lora_b_weights, output_tensor, lora_indices_tensor,
                             0, output_tensor.size(1), add_inputs)


def bgmv_expand_slice(inputs, lora_b_weights, output_tensor, lora_indices_tensor,
                      slice_offset, slice_size, add_inputs=True):
    if inputs.shape[0] > _EXPAND_MAX_B:
        torch.ops._C_ascend.bgmv_expand(
            inputs, lora_b_weights, lora_indices_tensor, output_tensor,
            slice_offset, slice_size)
        return output_tensor
    torch.ops.vllm_ascend_triton.bgmv_expand_slice(
        inputs, lora_b_weights, output_tensor, lora_indices_tensor, slice_offset, slice_size)
    return output_tensor


def sgmv_shrink(inputs, lora_a_weights, output_tensor, b_seq_start_loc, seq_len_tensor,
                lora_indices_tensor, batches, max_seq_length, token_nums, scaling):
    if inputs.shape[0] > _SHRINK_MAX_B:
        torch.ops._C_ascend.sgmv_shrink(
            inputs, lora_a_weights, lora_indices_tensor, seq_len_tensor,
            output_tensor, scaling)
        return output_tensor
    torch.ops.vllm_ascend_triton.sgmv_shrink(
        inputs, lora_a_weights, output_tensor, seq_len_tensor, lora_indices_tensor, scaling)
    return output_tensor


def sgmv_expand(inputs, lora_b_weights, output_tensor, b_seq_start_loc, seq_len_tensor,
                lora_indices_tensor, batches, max_seq_length, token_nums, add_inputs=False):
    return sgmv_expand_slice(inputs, lora_b_weights, output_tensor, b_seq_start_loc, seq_len_tensor,
                             lora_indices_tensor, batches, max_seq_length, token_nums,
                             0, output_tensor.size(1), add_inputs)


def sgmv_expand_slice(inputs, lora_b_weights, output_tensor, b_seq_start_loc, seq_len_tensor,
                      lora_indices_tensor, batches, max_seq_length, token_nums,
                      slice_offset, slice_size, add_inputs=False):
    if inputs.shape[0] > _EXPAND_MAX_B:
        torch.ops._C_ascend.sgmv_expand(
            inputs, lora_b_weights, lora_indices_tensor, seq_len_tensor,
            output_tensor, slice_offset, slice_size)
        return output_tensor
    torch.ops.vllm_ascend_triton.sgmv_expand_slice(
        inputs, lora_b_weights, output_tensor, seq_len_tensor, lora_indices_tensor,
        slice_offset, slice_size)
    return output_tensor
