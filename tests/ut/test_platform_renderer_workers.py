import os
from types import SimpleNamespace
from unittest.mock import patch

from tests.ut.base import TestBase
from vllm_ascend.platform import _apply_renderer_num_workers_default

ENV = "VLLM_ASCEND_RENDERER_NUM_WORKERS"


def _config(workers=1, runner_type="generate", mm_cache_gb=None):
    mm = None if mm_cache_gb is None else SimpleNamespace(mm_processor_cache_gb=mm_cache_gb)
    return SimpleNamespace(
        model_config=SimpleNamespace(
            renderer_num_workers=workers,
            runner_type=runner_type,
            multimodal_config=mm,
        )
    )


class TestRendererNumWorkersDefault(TestBase):
    def test_unset_is_a_no_op(self):
        """Unset must leave vLLM's own default alone, not force a value."""
        cfg = _config(workers=1)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(ENV, None)
            _apply_renderer_num_workers_default(cfg)
        self.assertEqual(cfg.model_config.renderer_num_workers, 1)

    def test_zero_is_a_no_op(self):
        cfg = _config(workers=1)
        with patch.dict(os.environ, {ENV: "0"}):
            _apply_renderer_num_workers_default(cfg)
        self.assertEqual(cfg.model_config.renderer_num_workers, 1)

    def test_raises_the_pool_size(self):
        cfg = _config(workers=1)
        with patch.dict(os.environ, {ENV: "8"}):
            _apply_renderer_num_workers_default(cfg)
        self.assertEqual(cfg.model_config.renderer_num_workers, 8)

    def test_never_lowers_an_explicit_value(self):
        """--renderer-num-workers 8 must win over a smaller env value."""
        cfg = _config(workers=8)
        with patch.dict(os.environ, {ENV: "2"}):
            _apply_renderer_num_workers_default(cfg)
        self.assertEqual(cfg.model_config.renderer_num_workers, 8)

    def test_refuses_pooling_with_mm_processor_cache(self):
        """ModelConfig has already validated by now, so this guard is ours.

        Upstream rejects >1 renderer worker for a pooling model while the
        multimodal processor cache is on, because pooling preprocessing runs
        on those workers and the cache is not thread-safe. Raising the value
        after validation would bypass that check silently.
        """
        cfg = _config(workers=1, runner_type="pooling", mm_cache_gb=4)
        with patch.dict(os.environ, {ENV: "8"}):
            _apply_renderer_num_workers_default(cfg)
        self.assertEqual(cfg.model_config.renderer_num_workers, 1)

    def test_allows_pooling_without_mm_processor_cache(self):
        cfg = _config(workers=1, runner_type="pooling", mm_cache_gb=0)
        with patch.dict(os.environ, {ENV: "8"}):
            _apply_renderer_num_workers_default(cfg)
        self.assertEqual(cfg.model_config.renderer_num_workers, 8)

    def test_allows_generate_with_mm_processor_cache(self):
        cfg = _config(workers=1, runner_type="generate", mm_cache_gb=4)
        with patch.dict(os.environ, {ENV: "8"}):
            _apply_renderer_num_workers_default(cfg)
        self.assertEqual(cfg.model_config.renderer_num_workers, 8)

    def test_tolerates_missing_model_config(self):
        cfg = SimpleNamespace(model_config=None)
        with patch.dict(os.environ, {ENV: "8"}):
            _apply_renderer_num_workers_default(cfg)

    def test_tolerates_missing_field(self):
        """Older vLLM pins may not carry renderer_num_workers at all."""
        cfg = SimpleNamespace(model_config=SimpleNamespace())
        with patch.dict(os.environ, {ENV: "8"}):
            _apply_renderer_num_workers_default(cfg)
        self.assertFalse(hasattr(cfg.model_config, "renderer_num_workers"))
