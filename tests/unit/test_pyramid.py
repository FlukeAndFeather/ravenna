"""
Unit tests for PyramidBuilder (issue 6).

Tests use small synthetic STFT arrays (not real audio) injected directly
as numpy arrays to keep things fast and deterministic.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import zarr

from ravenna.config import PipelineConfig
from ravenna.coordinates import TileCoordinates
from ravenna.pyramid.builder import PyramidBuilder

# ── Helpers ───────────────────────────────────────────────────────────────

def _linmean_db(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Linear-amplitude mean of two dB arrays, returned in dB."""
    return 20.0 * np.log10((10.0 ** (a / 20.0) + 10.0 ** (b / 20.0)) / 2.0)


# ── Constants ─────────────────────────────────────────────────────────────

SR = 8_000
HOP = 128
FFT = 512
TILE = 16          # tiny tile so tests stay fast
ZT_MAX = 3         # small zoom ranges for speed
ZF_MAX = 2
T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
N_FREQ_BINS = FFT // 2 + 1     # 257


# ── Helpers ───────────────────────────────────────────────────────────────

def _make_config(tmp_path, downsample_method: str = "mean") -> PipelineConfig:
    return PipelineConfig(
        source_uri=str(tmp_path),
        date_start=T0,
        date_end=T0 + timedelta(seconds=5),
        sample_rate=SR,
        fft_size=FFT,
        hop_size=HOP,
        tile_size=TILE,
        zoom_t_max=ZT_MAX,
        zoom_f_max=ZF_MAX,
        zarr_path=str(tmp_path / "stft.zarr"),
        pyramid_path=str(tmp_path / "pyramid.zarr"),
        downsample_method=downsample_method,
    )


def _fake_stft(n_t: int, n_f: int = N_FREQ_BINS) -> zarr.Array:
    """
    Return an in-memory Zarr array filled with known float values
    (row index cast to float) so downsampling results are predictable.
    """
    data = np.arange(n_t * n_f, dtype=np.float32).reshape(n_t, n_f)
    z = zarr.array(data, chunks=(16, n_f), dtype="float32")
    return z


def _n_time_frames(n_secs: float = 5) -> int:
    return math.ceil(round(n_secs * SR) / HOP)


# ── _downsample_time ──────────────────────────────────────────────────────

class TestDownsampleTime:
    def _builder(self, tmp_path, method="mean"):
        return PyramidBuilder(_make_config(tmp_path, method))

    def test_mean_2x2_block(self, tmp_path):
        b = self._builder(tmp_path)
        src = np.array([[1, 2], [3, 4]], dtype=np.float32)
        result = b._downsample_time(src)
        expected = _linmean_db(src[0], src[1]).reshape(1, 2)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_max_2x2_block(self, tmp_path):
        b = self._builder(tmp_path, "max")
        src = np.array([[1, 2], [3, 4]], dtype=np.float32)
        result = b._downsample_time(src)
        np.testing.assert_allclose(result, [[3.0, 4.0]])

    def test_output_shape_even(self, tmp_path):
        b = self._builder(tmp_path)
        src = np.ones((8, 4), dtype=np.float32)
        assert b._downsample_time(src).shape == (4, 4)

    def test_output_shape_odd_rows(self, tmp_path):
        b = self._builder(tmp_path)
        src = np.ones((5, 4), dtype=np.float32)
        # ceil(5/2) = 3
        assert b._downsample_time(src).shape == (3, 4)

    def test_output_dtype_float32(self, tmp_path):
        b = self._builder(tmp_path)
        src = np.ones((4, 4), dtype=np.float32)
        assert b._downsample_time(src).dtype == np.float32

    def test_mean_4_rows(self, tmp_path):
        b = self._builder(tmp_path)
        src = np.array([[0, 0], [2, 2], [4, 4], [6, 6]], dtype=np.float32)
        result = b._downsample_time(src)
        expected = np.stack([_linmean_db(src[0], src[1]), _linmean_db(src[2], src[3])])
        np.testing.assert_allclose(result, expected, rtol=1e-5)


# ── _downsample_freq ──────────────────────────────────────────────────────

class TestDownsampleFreq:
    def _builder(self, tmp_path, method="mean"):
        return PyramidBuilder(_make_config(tmp_path, method))

    def test_mean_2x2_block(self, tmp_path):
        b = self._builder(tmp_path)
        src = np.array([[1, 2], [3, 4]], dtype=np.float32)
        result = b._downsample_freq(src)
        expected = np.array([[_linmean_db(1.0, 2.0)], [_linmean_db(3.0, 4.0)]], dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_max_2x2_block(self, tmp_path):
        b = self._builder(tmp_path, "max")
        src = np.array([[1, 2], [3, 4]], dtype=np.float32)
        result = b._downsample_freq(src)
        np.testing.assert_allclose(result, [[2.0], [4.0]])

    def test_output_shape_even(self, tmp_path):
        b = self._builder(tmp_path)
        src = np.ones((4, 8), dtype=np.float32)
        assert b._downsample_freq(src).shape == (4, 4)

    def test_output_shape_odd_cols(self, tmp_path):
        b = self._builder(tmp_path)
        src = np.ones((4, 5), dtype=np.float32)
        # ceil(5/2) = 3
        assert b._downsample_freq(src).shape == (4, 3)

    def test_output_dtype_float32(self, tmp_path):
        b = self._builder(tmp_path)
        src = np.ones((4, 4), dtype=np.float32)
        assert b._downsample_freq(src).dtype == np.float32

    def test_freq_downsample_does_not_touch_time_axis(self, tmp_path):
        """Number of time rows must be unchanged by freq downsampling."""
        b = self._builder(tmp_path)
        src = np.ones((7, 6), dtype=np.float32)
        assert b._downsample_freq(src).shape[0] == 7

    def test_time_downsample_does_not_touch_freq_axis(self, tmp_path):
        """Number of freq bins must be unchanged by time downsampling."""
        b = self._builder(tmp_path)
        src = np.ones((6, 7), dtype=np.float32)
        assert b._downsample_time(src).shape[1] == 7


# ── build_all — structure ─────────────────────────────────────────────────

class TestBuildAll:
    def _run(self, tmp_path, n_t=None, method="mean"):
        cfg = _make_config(tmp_path, method)
        if n_t is None:
            n_t = _n_time_frames()
        stft = _fake_stft(n_t)
        builder = PyramidBuilder(cfg)
        group = builder.build_all(stft)
        return builder, group, stft

    def test_returns_zarr_group(self, tmp_path):
        _, group, _ = self._run(tmp_path)
        assert isinstance(group, zarr.Group)

    def test_all_levels_present(self, tmp_path):
        _, group, _ = self._run(tmp_path)
        for z_t in range(ZT_MAX + 1):
            for z_f in range(ZF_MAX + 1):
                assert f"zt{z_t}_zf{z_f}" in group, \
                    f"missing zt{z_t}_zf{z_f}"

    def test_correct_number_of_arrays(self, tmp_path):
        _, group, _ = self._run(tmp_path)
        assert len(group) == (ZT_MAX + 1) * (ZF_MAX + 1)

    def test_full_res_level_matches_stft(self, tmp_path):
        _, group, stft = self._run(tmp_path)
        key = f"zt{ZT_MAX}_zf{ZF_MAX}"
        np.testing.assert_array_equal(group[key][:], stft[:])

    def test_array_shapes_match_formula(self, tmp_path):
        n_t = _n_time_frames()
        _, group, _ = self._run(tmp_path, n_t=n_t)
        for z_t in range(ZT_MAX + 1):
            for z_f in range(ZF_MAX + 1):
                key = f"zt{z_t}_zf{z_f}"
                arr = group[key]
                exp_t = math.ceil(n_t / 2 ** (ZT_MAX - z_t))
                exp_f = math.ceil(N_FREQ_BINS / 2 ** (ZF_MAX - z_f))
                assert arr.shape == (exp_t, exp_f), \
                    f"{key}: expected {(exp_t, exp_f)}, got {arr.shape}"

    def test_chunk_shape_is_tile_size(self, tmp_path):
        n_t = TILE * 4   # array larger than tile
        _, group, _ = self._run(tmp_path, n_t=n_t)
        key = f"zt{ZT_MAX}_zf{ZF_MAX}"
        arr = group[key]
        assert arr.chunks[0] <= TILE
        assert arr.chunks[1] <= TILE


# ── build_all — values ────────────────────────────────────────────────────

class TestBuildValues:
    """Verify that downsampled values are numerically correct."""

    def test_time_downsampling_mean(self, tmp_path):
        """(zt_max−1, zf_max) must be the mean of each pair of rows."""
        cfg = _make_config(tmp_path, "mean")
        n_t = 4   # 2 output rows
        n_f = N_FREQ_BINS
        data = np.arange(n_t * n_f, dtype=np.float32).reshape(n_t, n_f)
        stft = zarr.array(data, chunks=(n_t, n_f), dtype="float32")

        group = PyramidBuilder(cfg).build_all(stft)

        key = f"zt{ZT_MAX - 1}_zf{ZF_MAX}"
        result = group[key][:]
        # Each output row = linear-amplitude mean of two input rows
        expected = _linmean_db(data[0::2], data[1::2])
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_time_downsampling_max(self, tmp_path):
        cfg = _make_config(tmp_path, "max")
        n_t = 4
        n_f = N_FREQ_BINS
        data = np.arange(n_t * n_f, dtype=np.float32).reshape(n_t, n_f)
        stft = zarr.array(data, chunks=(n_t, n_f), dtype="float32")

        group = PyramidBuilder(cfg).build_all(stft)

        key = f"zt{ZT_MAX - 1}_zf{ZF_MAX}"
        result = group[key][:]
        expected = np.maximum(data[0::2], data[1::2])
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_freq_downsampling_mean(self, tmp_path):
        """(zt_max, zf_max−1) must be the mean of each pair of freq bins."""
        cfg = _make_config(tmp_path, "mean")
        n_t = 4
        n_f = N_FREQ_BINS
        data = np.arange(n_t * n_f, dtype=np.float32).reshape(n_t, n_f)
        stft = zarr.array(data, chunks=(n_t, n_f), dtype="float32")

        group = PyramidBuilder(cfg).build_all(stft)

        key = f"zt{ZT_MAX}_zf{ZF_MAX - 1}"
        result = group[key][:]
        # Pair up adjacent freq bins; ignore trailing odd bin
        n_pairs = n_f // 2
        expected = _linmean_db(data[:, 0::2][:, :n_pairs], data[:, 1::2][:, :n_pairs])
        np.testing.assert_allclose(result[:, :n_pairs], expected, rtol=1e-5)


# ── tile_extent ───────────────────────────────────────────────────────────

class TestTileExtent:
    def test_matches_tile_coordinates(self, tmp_path):
        """tile_extent() must agree with TileCoordinates.tile_extent()."""
        cfg = _make_config(tmp_path)
        n_t = _n_time_frames()
        stft = _fake_stft(n_t)
        builder = PyramidBuilder(cfg)
        builder.build_all(stft)

        tc = TileCoordinates(
            sample_rate=SR,
            hop_size=HOP,
            fft_size=FFT,
            n_time_frames=n_t,
            tile_size=TILE,
            zoom_t_max=ZT_MAX,
            zoom_f_max=ZF_MAX,
        )

        for z_t in range(ZT_MAX + 1):
            for z_f in range(ZF_MAX + 1):
                assert builder.tile_extent(z_t, z_f) == tc.tile_extent(z_t, z_f), \
                    f"mismatch at zt={z_t} zf={z_f}"


# ── Idempotency ───────────────────────────────────────────────────────────

class TestIdempotency:
    def test_second_build_skips_existing(self, tmp_path):
        """Second build_all() call must not alter any array values."""
        cfg = _make_config(tmp_path)
        n_t = _n_time_frames()
        stft = _fake_stft(n_t)

        g1 = PyramidBuilder(cfg).build_all(stft)
        snapshots = {k: g1[k][:].copy() for k in g1}

        # Build again with different data — should be skipped
        stft2 = zarr.array(np.zeros((n_t, N_FREQ_BINS), dtype=np.float32))
        g2 = PyramidBuilder(cfg).build_all(stft2)

        for k, original in snapshots.items():
            np.testing.assert_array_equal(g2[k][:], original, err_msg=f"array {k} changed")

    def test_second_build_all_levels_still_present(self, tmp_path):
        cfg = _make_config(tmp_path)
        stft = _fake_stft(_n_time_frames())
        PyramidBuilder(cfg).build_all(stft)
        g = PyramidBuilder(cfg).build_all(stft)
        assert len(g) == (ZT_MAX + 1) * (ZF_MAX + 1)
