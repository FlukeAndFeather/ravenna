"""Unit tests for ravenna.render.norm."""
from __future__ import annotations

import numpy as np
import pytest
import zarr

from ravenna.render.norm import (
    CalibratedSPLNorm,
    GlobalPercentileNorm,
    NormStrategy,
    make_norm_strategy,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_group(coarse_data: np.ndarray) -> zarr.Group:
    """In-memory zarr group with a single coarsest-level array."""
    g = zarr.group()
    g["zt0_zf0"] = coarse_data
    return g


# ── GlobalPercentileNorm ──────────────────────────────────────────────────────

class TestGlobalPercentileNorm:
    def test_implements_norm_strategy(self):
        n = GlobalPercentileNorm(-100.0, 0.0)
        assert isinstance(n, NormStrategy)

    def test_vmin_vmax(self):
        n = GlobalPercentileNorm(-80.0, -40.0)
        assert n.vmin == -80.0
        assert n.vmax == -40.0

    def test_normalize_midpoint(self):
        n = GlobalPercentileNorm(0.0, 100.0)
        result = n.normalize(np.array([50.0], dtype=np.float32))
        np.testing.assert_allclose(result, [0.5], atol=1e-6)

    def test_normalize_clips_below_zero(self):
        n = GlobalPercentileNorm(0.0, 100.0)
        result = n.normalize(np.array([-10.0], dtype=np.float32))
        assert result[0] == pytest.approx(0.0)

    def test_normalize_clips_above_one(self):
        n = GlobalPercentileNorm(0.0, 100.0)
        result = n.normalize(np.array([110.0], dtype=np.float32))
        assert result[0] == pytest.approx(1.0)

    def test_output_dtype_float32(self):
        n = GlobalPercentileNorm(0.0, 100.0)
        result = n.normalize(np.array([50.0], dtype=np.float32))
        assert result.dtype == np.float32

    def test_noise_floor_clips_to_zero(self):
        """Fill-value pixels (-200 dB) must map to 0, not negative."""
        n = GlobalPercentileNorm(-130.0, -80.0)
        result = n.normalize(np.array([-200.0], dtype=np.float32))
        assert result[0] == pytest.approx(0.0)

    def test_zero_span_returns_zeros(self):
        n = GlobalPercentileNorm(-100.0, -100.0)
        result = n.normalize(np.array([-100.0, -50.0, -150.0], dtype=np.float32))
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0])

    # ── from_pyramid ──────────────────────────────────────────────────────────

    def test_from_pyramid_excludes_fill_values(self):
        """Noise floor (-200 dB) must not affect the percentile computation."""
        data = np.full((4, 4), -200.0, dtype=np.float32)
        data[0, 0] = -100.0
        data[0, 1] = -80.0
        g = _make_group(data)
        n = GlobalPercentileNorm.from_pyramid(g, zoom_f_min=0, zoom_t_min=0)
        assert n.vmin >= -100.0
        assert n.vmax <= -80.0

    def test_from_pyramid_all_fill_returns_default(self):
        """All-fill-value arrays return a sensible default rather than raising."""
        data = np.full((4, 4), -200.0, dtype=np.float32)
        g = _make_group(data)
        n = GlobalPercentileNorm.from_pyramid(g, zoom_f_min=0, zoom_t_min=0)
        assert n.vmin < n.vmax

    def test_from_pyramid_uses_correct_key(self):
        """from_pyramid must read zt{zoom_t_min}_zf{zoom_f_min}."""
        g = zarr.group()
        g["zt2_zf3"] = np.array([[-100.0, -80.0]], dtype=np.float32)
        n = GlobalPercentileNorm.from_pyramid(g, zoom_f_min=3, zoom_t_min=2)
        assert n.vmin <= n.vmax


# ── CalibratedSPLNorm ─────────────────────────────────────────────────────────

class TestCalibratedSPLNorm:
    def test_implements_norm_strategy(self):
        n = CalibratedSPLNorm(-168.0, 80.0, 120.0)
        assert isinstance(n, NormStrategy)

    def test_vmin_vmax(self):
        n = CalibratedSPLNorm(-168.0, 80.0, 120.0)
        assert n.vmin == 80.0
        assert n.vmax == 120.0

    def test_conversion_formula(self):
        """SPL = dBFS - sensitivity_db; then normalize to [spl_min, spl_max]."""
        sensitivity = -168.0
        spl_min, spl_max = 80.0, 120.0
        n = CalibratedSPLNorm(sensitivity, spl_min, spl_max)

        # If dBFS = -68, SPL = -68 - (-168) = 100.0 → midpoint of [80, 120] → 0.5
        dbfs = np.array([-68.0], dtype=np.float32)
        result = n.normalize(dbfs)
        np.testing.assert_allclose(result, [0.5], atol=1e-5)

    def test_clips_below_spl_min(self):
        n = CalibratedSPLNorm(-168.0, 80.0, 120.0)
        # SPL = -200 - (-168) = -32 → way below spl_min
        result = n.normalize(np.array([-200.0], dtype=np.float32))
        assert result[0] == pytest.approx(0.0)

    def test_clips_above_spl_max(self):
        n = CalibratedSPLNorm(-168.0, 80.0, 120.0)
        # SPL = 0 - (-168) = 168 → way above spl_max
        result = n.normalize(np.array([0.0], dtype=np.float32))
        assert result[0] == pytest.approx(1.0)

    def test_output_dtype_float32(self):
        n = CalibratedSPLNorm(-168.0, 80.0, 120.0)
        result = n.normalize(np.ones((3, 3), dtype=np.float32) * -68.0)
        assert result.dtype == np.float32

    def test_invalid_spl_range_raises(self):
        with pytest.raises(ValueError):
            CalibratedSPLNorm(-168.0, spl_min_db=120.0, spl_max_db=80.0)

    def test_equal_spl_bounds_raises(self):
        with pytest.raises(ValueError):
            CalibratedSPLNorm(-168.0, spl_min_db=100.0, spl_max_db=100.0)


# ── make_norm_strategy ────────────────────────────────────────────────────────

class TestMakeNormStrategy:
    def _group(self):
        data = np.linspace(-130, -80, 100, dtype=np.float32).reshape(10, 10)
        return _make_group(data)

    def test_global_percentile_returns_correct_type(self):
        n = make_norm_strategy("global_percentile", self._group(), 0, 0)
        assert isinstance(n, GlobalPercentileNorm)

    def test_calibrated_spl_returns_correct_type(self):
        n = make_norm_strategy(
            "calibrated_spl", self._group(), 0, 0,
            hydrophone_sensitivity_db=-168.0,
            spl_display_min_db=80.0,
            spl_display_max_db=120.0,
        )
        assert isinstance(n, CalibratedSPLNorm)

    def test_calibrated_spl_missing_sensitivity_raises(self):
        with pytest.raises(ValueError, match="hydrophone_sensitivity_db"):
            make_norm_strategy(
                "calibrated_spl", self._group(), 0, 0,
                spl_display_min_db=80.0,
                spl_display_max_db=120.0,
            )

    def test_calibrated_spl_missing_min_raises(self):
        with pytest.raises(ValueError, match="spl_display_min_db"):
            make_norm_strategy(
                "calibrated_spl", self._group(), 0, 0,
                hydrophone_sensitivity_db=-168.0,
                spl_display_max_db=120.0,
            )

    def test_calibrated_spl_missing_max_raises(self):
        with pytest.raises(ValueError, match="spl_display_max_db"):
            make_norm_strategy(
                "calibrated_spl", self._group(), 0, 0,
                hydrophone_sensitivity_db=-168.0,
                spl_display_min_db=80.0,
            )

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown norm_strategy"):
            make_norm_strategy("no_such_strategy", self._group(), 0, 0)

    def test_global_percentile_passes_pct_params(self):
        data = np.linspace(-130, -80, 100, dtype=np.float32).reshape(10, 10)
        g = _make_group(data)
        n = make_norm_strategy(
            "global_percentile", g, 0, 0,
            norm_low_pct=0.0, norm_high_pct=100.0,
        )
        assert n.vmin == pytest.approx(float(data.min()), abs=0.5)
        assert n.vmax == pytest.approx(float(data.max()), abs=0.5)
