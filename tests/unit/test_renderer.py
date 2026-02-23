"""
Unit tests for apply_colormap and TileRenderer.

Uses small in-memory Zarr arrays so no real audio or STFT is needed.
"""
from __future__ import annotations

import io
from datetime import datetime, timezone

import numpy as np
import pytest
import zarr
from PIL import Image

from ravenna.config import PipelineConfig
from ravenna.render.colormap import apply_colormap
from ravenna.render.norm import GlobalPercentileNorm
from ravenna.render.renderer import TileRenderer, _encode_png

# ── Constants ─────────────────────────────────────────────────────────────────

TILE = 16
T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_config(**kw) -> PipelineConfig:
    from datetime import timedelta
    defaults = dict(
        source_uri=".",
        date_start=T0,
        date_end=T0 + timedelta(seconds=5),
        sample_rate=8_000,
        fft_size=512,
        hop_size=128,
        tile_size=TILE,
        zoom_t_max=3,
        zoom_f_max=2,
        zoom_f_min=0,
    )
    defaults.update(kw)
    return PipelineConfig(**defaults)


def _make_group(n_t: int = TILE, n_f: int = TILE, value: float = -100.0) -> zarr.Group:
    """In-memory group with a single zt3_zf2 array (full-res level)."""
    g = zarr.group()
    data = np.full((n_t, n_f), value, dtype=np.float32)
    g["zt3_zf2"] = data
    return g


def _norm(vmin: float = -130.0, vmax: float = -80.0) -> GlobalPercentileNorm:
    return GlobalPercentileNorm(vmin, vmax)


def _decode_png(png_bytes: bytes) -> np.ndarray:
    """Decode PNG bytes to (H, W, 4) uint8 RGBA array."""
    return np.array(Image.open(io.BytesIO(png_bytes)).convert("RGBA"))


# ── apply_colormap ────────────────────────────────────────────────────────────

class TestApplyColormap:
    def test_output_shape_2d(self):
        data = np.zeros((4, 4), dtype=np.float32)
        out = apply_colormap(data)
        assert out.shape == (4, 4, 4)

    def test_output_shape_1d(self):
        data = np.linspace(0, 1, 8, dtype=np.float32)
        out = apply_colormap(data)
        assert out.shape == (8, 4)

    def test_output_dtype_uint8(self):
        data = np.zeros((2, 2), dtype=np.float32)
        out = apply_colormap(data)
        assert out.dtype == np.uint8

    def test_values_in_range(self):
        data = np.linspace(0.0, 1.0, 50, dtype=np.float32).reshape(5, 10)
        out = apply_colormap(data)
        assert out.min() >= 0
        assert out.max() <= 255

    def test_zero_maps_consistently(self):
        """All-zero input should produce a single colour."""
        data = np.zeros((3, 3), dtype=np.float32)
        out = apply_colormap(data)
        assert np.all(out == out[0, 0])

    def test_one_maps_consistently(self):
        """All-one input should produce a single colour."""
        data = np.ones((3, 3), dtype=np.float32)
        out = apply_colormap(data)
        assert np.all(out == out[0, 0])

    def test_zero_and_one_differ(self):
        """Min and max should map to different colours."""
        c0 = apply_colormap(np.array([0.0], dtype=np.float32))
        c1 = apply_colormap(np.array([1.0], dtype=np.float32))
        assert not np.array_equal(c0, c1)

    def test_unknown_colormap_raises(self):
        with pytest.raises(KeyError):
            apply_colormap(np.zeros((2, 2), dtype=np.float32), "not_a_colormap")

    def test_alternative_colormap(self):
        """Should accept any valid matplotlib colormap name."""
        data = np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4)
        out = apply_colormap(data, "plasma")
        assert out.shape == (4, 4, 4)


# ── _encode_png ───────────────────────────────────────────────────────────────

class TestEncodePng:
    def test_returns_bytes(self):
        rgba = np.zeros((8, 8, 4), dtype=np.uint8)
        result = _encode_png(rgba)
        assert isinstance(result, bytes)

    def test_valid_png_header(self):
        rgba = np.zeros((8, 8, 4), dtype=np.uint8)
        result = _encode_png(rgba)
        assert result[:8] == b"\x89PNG\r\n\x1a\n"

    def test_round_trip_dimensions(self):
        rgba = np.zeros((12, 16, 4), dtype=np.uint8)
        result = _encode_png(rgba)
        img = Image.open(io.BytesIO(result))
        assert img.size == (16, 12)   # PIL size is (width, height)

    def test_round_trip_pixel_values(self):
        rgba = np.full((4, 4, 4), 127, dtype=np.uint8)
        result = _encode_png(rgba)
        decoded = _decode_png(result)
        np.testing.assert_array_equal(decoded, rgba)


# ── TileRenderer ──────────────────────────────────────────────────────────────

class TestTileRenderer:
    def _renderer(self, group=None, value=-100.0, **cfg_kw):
        if group is None:
            group = _make_group(value=value)
        cfg = _make_config(**cfg_kw)
        norm = _norm()
        return TileRenderer(cfg, group, norm)

    # ── render_tile output ────────────────────────────────────────────────

    def test_returns_bytes(self):
        r = self._renderer()
        result = r.render_tile(3, 2, 0, 0)
        assert isinstance(result, bytes)

    def test_output_is_valid_png(self):
        r = self._renderer()
        result = r.render_tile(3, 2, 0, 0)
        assert result[:8] == b"\x89PNG\r\n\x1a\n"

    def test_output_dimensions_are_tile_size(self):
        r = self._renderer()
        png = r.render_tile(3, 2, 0, 0)
        img = _decode_png(png)
        assert img.shape == (TILE, TILE, 4)

    def test_output_dtype_uint8(self):
        r = self._renderer()
        png = r.render_tile(3, 2, 0, 0)
        assert _decode_png(png).dtype == np.uint8

    # ── fill-value pixels ─────────────────────────────────────────────────

    def test_fill_pixels_map_to_darkest_colour(self):
        """
        A tile entirely filled with the noise floor (-200 dB) should render
        as the darkest colour (normalised value 0.0).
        """
        group = _make_group(value=-200.0)
        r = self._renderer(group)
        png = r.render_tile(3, 2, 0, 0)
        img = _decode_png(png)
        darkest = apply_colormap(np.array([0.0], dtype=np.float32))[0]
        np.testing.assert_array_equal(img[0, 0], darkest)

    # ── edge / partial tiles ──────────────────────────────────────────────

    def test_partial_tile_still_tile_size(self):
        """Array smaller than tile_size → output still tile_size × tile_size."""
        group = _make_group(n_t=5, n_f=7)
        r = self._renderer(group)
        png = r.render_tile(3, 2, 0, 0)
        img = _decode_png(png)
        assert img.shape == (TILE, TILE, 4)

    def test_partial_tile_fill_region_is_darkest(self):
        """
        Pixels beyond the partial data must be fill-value → darkest colour.
        """
        group = _make_group(n_t=5, n_f=7, value=-100.0)
        r = self._renderer(group)
        png = r.render_tile(3, 2, 0, 0)
        img = _decode_png(png)
        darkest = apply_colormap(np.array([0.0], dtype=np.float32))[0]
        # After flipud(canvas.T): columns TILE-5 to TILE-1 have actual time data.
        # Rows TILE-7 to TILE-1 have actual freq data.
        # Pixels outside that region are fill.
        np.testing.assert_array_equal(img[0, 0], darkest)

    # ── _extract_tile_data orientation ───────────────────────────────────

    def test_extract_shape(self):
        r = self._renderer()
        data = r._extract_tile_data(3, 2, 0, 0)
        assert data.shape == (TILE, TILE)

    def test_extract_dtype_float32(self):
        r = self._renderer()
        data = r._extract_tile_data(3, 2, 0, 0)
        assert data.dtype == np.float32

    def test_high_freq_at_row_0(self):
        """
        arr[t, f] with f increasing → after flipud(canvas.T), row 0 should
        contain the highest frequency values.
        """
        n_t, n_f = TILE, TILE
        data = np.tile(
            np.arange(n_f, dtype=np.float32), (n_t, 1)
        )   # data[t, f] = f
        g = zarr.group()
        g["zt3_zf2"] = data
        r = self._renderer(g)
        extracted = r._extract_tile_data(3, 2, 0, 0)
        # Row 0 of extracted should come from the highest freq index
        assert extracted[0, 0] == pytest.approx(float(n_f - 1))
        assert extracted[-1, 0] == pytest.approx(0.0)

    def test_time_advances_left_to_right(self):
        """
        arr[t, f] with t increasing → after flipud(canvas.T), column index
        should increase with t.
        """
        n_t, n_f = TILE, TILE
        data = np.tile(
            np.arange(n_t, dtype=np.float32).reshape(-1, 1), (1, n_f)
        )   # data[t, f] = t
        g = zarr.group()
        g["zt3_zf2"] = data
        r = self._renderer(g)
        extracted = r._extract_tile_data(3, 2, 0, 0)
        # Column j should contain values from time frame j
        assert extracted[0, 0] == pytest.approx(0.0)
        assert extracted[0, -1] == pytest.approx(float(n_t - 1))
