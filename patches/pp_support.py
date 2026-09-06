#!/usr/bin/env python3
"""GLM-5.3-Flash pipeline-parallel (PP) enablement patch for vllm-ascend.

Problem
-------
AscendGlm5NextForCausalLM declares SupportsPP (glm5_next.py:2437) but never
defines make_empty_intermediate_tensors.  Because SupportsPP is a *Protocol*
whose method bodies are `...`, the attribute is silently inherited as a real
method returning None.  supports_pp() therefore returns True, vLLM accepts
--pipeline-parallel-size>1 without any warning, and every non-first PP rank
then dies inside _dummy_run with
    AttributeError: 'NoneType' object has no attribute ...

Simply adding the stock factory is NOT enough for this model:

  * With hyper-connections (mhc=True, mhc_num_residual_streams=4) the tensor
    crossing a PP boundary is 3-D [T, n, hidden] -- layer 0 expands it via
    _expand_mhc_residual_streams (glm5_next.py:2213-2215) and only the final
    layer collapses it back with .mean(dim=1) (glm5_next.py:2237-2239).
    make_empty_intermediate_tensors_factory allocates 2-D [T, hidden]
    (vllm/model_executor/models/utils.py:700-713), so the receive buffer shape
    would not match the sent tensor.
  * In the mHC path the decoder layer returns residual=None
    (glm5_next.py:2240), but the model unconditionally packs
    {"hidden_states": ..., "residual": residual} into IntermediateTensors
    (glm5_next.py:2316) and unconditionally reads ["residual"] on the
    receiving rank (glm5_next.py:2309).  A None in that dict breaks the PP
    transport, which iterates the dict and isend()s each value.

This patch fixes all three.  Idempotent; keeps a .pp_orig backup.

NOT covered here (deliberately, documented instead):
  * AscendGlm5NextForConditionalGeneration.__init__ builds the 403M-parameter
    vision tower on EVERY rank (glm5_next_multimodal.py:679-686) with no
    is_first_rank guard.  Wasteful (~0.8 GiB bf16 per rank) but not incorrect,
    and guarding it would change weight loading -- treat separately.

Usage:
    python3 pp_support.py [--root /vllm-workspace/vllm-ascend/vllm_ascend]
    python3 pp_support.py --revert
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

MODEL_INIT_ANCHOR = '''        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
'''

MODEL_INIT_NEW = MODEL_INIT_ANCHOR + '''
        # --- PP enablement -------------------------------------------------
        # SupportsPP is a Protocol: not defining this attribute silently yields
        # an inherited method returning None, so PP only fails later, inside
        # _dummy_run.  With mHC the PP-boundary tensor is [T, n, hidden] (layer 0
        # expands, the last layer collapses), and mHC layers produce
        # residual=None, so the stock 2-D factory cannot be used.
        _mhc_streams = (
            int(getattr(config, "mhc_num_residual_streams", 1) or 1)
            if getattr(config, "mhc", False)
            else 1
        )
        self.pp_mhc_streams = _mhc_streams
        if _mhc_streams > 1:
            _hidden = config.hidden_size

            def _make_empty_intermediate_tensors(
                batch_size: int,
                dtype: torch.dtype,
                device: torch.device,
            ) -> IntermediateTensors:
                return IntermediateTensors(
                    {
                        "hidden_states": torch.zeros(
                            (batch_size, _mhc_streams, _hidden),
                            dtype=dtype,
                            device=device,
                        )
                    }
                )

            self.make_empty_intermediate_tensors = _make_empty_intermediate_tensors
        else:
            self.make_empty_intermediate_tensors = (
                make_empty_intermediate_tensors_factory(
                    ["hidden_states", "residual"], config.hidden_size
                )
            )
        # --- end PP enablement ---------------------------------------------
'''

RECV_ANCHOR = '''            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
'''

RECV_NEW = '''            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            # mHC layers carry no separate residual; the sender omits the key.
            residual = intermediate_tensors.tensors.get("residual")
'''

SEND_ANCHOR = '''        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})
'''

SEND_NEW = '''        if not get_pp_group().is_last_rank:
            # Never put None into IntermediateTensors: the PP transport iterates
            # the dict and isend()s each value.
            pp_tensors = {"hidden_states": hidden_states}
            if residual is not None:
                pp_tensors["residual"] = residual
            return IntermediateTensors(pp_tensors)
'''

CAUSAL_ANCHOR = '''        self.logits_processor = LogitsProcessor(
            self.config.vocab_size,
            scale=logit_scale,
        )
'''

CAUSAL_NEW = CAUSAL_ANCHOR + '''        # Required by SupportsPP; see AscendGlm5NextModel.__init__.
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
'''

IMPORT_ANCHOR = '''from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
    sequence_parallel_chunk,
)
'''

IMPORT_NEW = '''from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
    sequence_parallel_chunk,
)
'''

MM_ANCHOR = '''            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["Glm5NextForCausalLM"],
            )
'''

MM_NEW = MM_ANCHOR + '''
        # This class calls super(Glm4vForConditionalGeneration, self).__init__(),
        # deliberately skipping the parent __init__ that would have forwarded
        # make_empty_intermediate_tensors (glm4_1v.py:1655-1657).  Forward it
        # here or PP fails on every non-first rank.
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )
'''

PPMTP_ANCHOR = '''    mtp_model_types = set(get_args(MTPModelTypes))

    @wraps(original_verify)
    def _patched_verify_with_parallel_config(self, parallel_config):
        hf_config = getattr(self, "hf_config", None)
'''

PPMTP_NEW = '''    @wraps(original_verify)
    def _patched_verify_with_parallel_config(self, parallel_config):
        # Resolve MTPModelTypes LAZILY, at call time.
        #
        # patch_speculative_config.py appends "glm5_next_mtp" to
        # vllm.config.speculative.MTPModelTypes, and it is imported at
        # vllm_ascend/patch/platform/__init__.py:53 -- 30 lines AFTER this
        # module (:23).  A set captured at patch time therefore never contains
        # "glm5_next_mtp", is_mtp_drafter is always False for GLM-5.3-Flash,
        # and this whole patch is dead code for it: the drafter falls through
        # to the unpatched verify and dies with
        #   NotImplementedError: Pipeline parallelism is not supported for
        #   this model. Supported models implement the `SupportsPP` interface.
        from vllm.config import speculative as _speculative_mod

        mtp_model_types = set(get_args(_speculative_mod.MTPModelTypes))
        hf_config = getattr(self, "hf_config", None)
'''

EDITS = {
    "patch/platform/patch_pp_mtp.py": [
        ("patch_pp_mtp: resolve MTPModelTypes lazily", PPMTP_ANCHOR, PPMTP_NEW),
    ],
    "models/glm5_next.py": [
        ("import make_empty_intermediate_tensors_factory", IMPORT_ANCHOR, IMPORT_NEW),
        ("AscendGlm5NextModel.make_empty_intermediate_tensors", MODEL_INIT_ANCHOR, MODEL_INIT_NEW),
        ("forward(): tolerate absent residual", RECV_ANCHOR, RECV_NEW),
        ("forward(): never send None", SEND_ANCHOR, SEND_NEW),
        ("AscendGlm5NextForCausalLM: forward the attribute", CAUSAL_ANCHOR, CAUSAL_NEW),
    ],
    "models/glm5_next_multimodal.py": [
        ("multimodal wrapper: forward the attribute", MM_ANCHOR, MM_NEW),
    ],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/vllm-workspace/vllm-ascend/vllm_ascend")
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    if not root.is_dir():
        print("FAIL: %s is not a directory" % root, file=sys.stderr)
        return 2

    rc = 0
    for rel, edits in EDITS.items():
        f = root / rel
        orig = Path(str(f) + ".pp_orig")
        if a.revert:
            if orig.exists():
                shutil.copy2(orig, f)
                print("REVERTED %s" % f)
            else:
                print("SKIP (no backup) %s" % f)
            continue
        if not f.exists():
            print("FAIL: missing %s" % f, file=sys.stderr)
            rc = 1
            continue
        text = f.read_text(encoding="utf-8")
        if not orig.exists():
            shutil.copy2(f, orig)
        for name, anchor, new in edits:
            if new in text:
                print("  already applied: %s" % name)
                continue
            n = text.count(anchor)
            if n != 1:
                print(
                    "FAIL: anchor for '%s' matched %d times in %s (expected 1)"
                    % (name, n, rel),
                    file=sys.stderr,
                )
                rc = 1
                continue
            text = text.replace(anchor, new, 1)
            print("  applied: %s" % name)
        f.write_text(text, encoding="utf-8")
        print("WROTE %s  (backup %s)" % (f, orig.name))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
