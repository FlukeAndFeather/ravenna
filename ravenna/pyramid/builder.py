"""
PyramidBuilder — Stage 3 of the Ravenna pipeline.

Takes the full-resolution STFT Zarr array produced by STFTProcessor and
builds a 2D pyramid by independently downsampling the time and frequency
axes.

Storage layout
--------------
  Group path : config.pyramid_path
  Key        : zt{z_t}_zf{z_f}
  Shape      : (ceil(n_t / 2^(zt_max−z_t)),  ceil(n_f / 2^(zf_max−z_f)))
  Chunks     : (tile_size, tile_size)
  Fill value : -200.0 (noise floor, for partially-covered tiles)

Build order
-----------
1. (zoom_t_max, zoom_f_max) — copy of the full-resolution STFT array
2. (zoom_t_max, z_f) for z_f < zoom_f_max — freq-only downsampling from
   (zoom_t_max, z_f + 1)
3. For each z_t < zoom_t_max, for each z_f — time-only downsampling from
   (z_t + 1, z_f)

This order ensures every source array is available before it is needed.

Idempotency
-----------
Each (z_t, z_f) level is skipped if the key already exists in the group
with the expected shape.
"""
from __future__ import annotations

import logging
import math
import time

import numpy as np
import zarr

from ravenna.config import PipelineConfig
from ravenna.coordinates import TileCoordinates, TileExtent

# Noise-floor sentinel used to pad odd-length axes before downsampling.
_NOISE_FLOOR = -200.0

logger = logging.getLogger(__name__)


class PyramidBuilder:
    """
    Build the 2D spectrogram pyramid from a full-resolution STFT Zarr array.

    Parameters
    ----------
    config : PipelineConfig
    verbose : bool
        When True, emit INFO-level progress messages to stderr showing each
        pyramid level being built, its shape, and elapsed time.  When False
        (default) the logger is silent unless the caller has configured a
        handler on ``ravenna.pyramid.builder`` or a parent logger.
    """

    def __init__(self, config: PipelineConfig, verbose: bool = False) -> None:
        self.config = config
        self.n_freq_bins: int = config.fft_size // 2 + 1
        self._n_time_frames: int | None = None   # set by build_all()
        self._group: zarr.Group | None = None    # set by build_all()

        if verbose and not logger.handlers:
            _handler = logging.StreamHandler()
            _handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(_handler)
            logger.setLevel(logging.INFO)

    # ── Public API ────────────────────────────────────────────────────────

    def build_all(self, stft_array: zarr.Array) -> zarr.Group:
        """
        Build the complete pyramid from *stft_array* and return the group.

        If the pyramid Zarr group already exists and all levels have the
        expected shapes, this call is a no-op (idempotent).
        """
        self._n_time_frames = stft_array.shape[0]
        self._group = zarr.open_group(self.config.pyramid_path, mode="a")

        zt_max = self.config.zoom_t_max
        zf_max = self.config.zoom_f_max
        n_levels = (zt_max - self.config.zoom_t_min + 1) * (zf_max - self.config.zoom_f_min + 1)
        logger.info("Building pyramid: %d levels", n_levels)
        t_all = time.perf_counter()

        # Step 1: full-res level → copy of STFT
        self._build_full_res(stft_array)

        # Step 2: rest of the zt_max column (freq-only downsampling)
        for z_f in range(zf_max - 1, self.config.zoom_f_min - 1, -1):
            self.build_level(zt_max, z_f)

        # Step 3: all z_t < zt_max (time-only downsampling from z_t+1)
        for z_t in range(zt_max - 1, self.config.zoom_t_min - 1, -1):
            for z_f in range(zf_max, self.config.zoom_f_min - 1, -1):
                self.build_level(z_t, z_f)

        logger.info("Pyramid complete in %.2fs", time.perf_counter() - t_all)
        return self._group

    def build_level(self, z_t: int, z_f: int) -> zarr.Array:
        """
        Build and return the pyramid array for zoom pair (z_t, z_f).

        Must be called after build_all() has initialised the group and
        computed all dependencies (i.e. do not call this directly unless
        the dependency array is already in the group).
        """
        assert self._group is not None, "call build_all() first"
        group = self._group
        zt_max = self.config.zoom_t_max
        zf_max = self.config.zoom_f_max

        if z_t == zt_max and z_f == zf_max:
            return group[f"zt{zt_max}_zf{zf_max}"]

        # Determine source key and downsampling direction
        if z_t == zt_max:
            src_key = f"zt{zt_max}_zf{z_f + 1}"
            direction = "freq"
        else:
            src_key = f"zt{z_t + 1}_zf{z_f}"
            direction = "time"

        src = group[src_key]
        expected_shape = self._expected_shape(src.shape, direction)
        dst_key = f"zt{z_t}_zf{z_f}"

        # Idempotency: skip if already built with correct shape
        if dst_key in group and group[dst_key].shape == expected_shape:
            logger.info("  %s skipped (exists)", dst_key)
            return group[dst_key]

        logger.info("  %s  %s-downsample %s → %s",
                    dst_key, direction, src.shape, expected_shape)
        t0 = time.perf_counter()

        data = src[:].astype(np.float32)
        if direction == "freq":
            result = self._downsample_freq(data)
        else:
            result = self._downsample_time(data)

        dst = self._write(group, dst_key, result)
        logger.info("    done in %.2fs", time.perf_counter() - t0)
        return dst

    def tile_extent(self, z_t: int, z_f: int) -> TileExtent:
        """
        Return the tile grid extent at zoom pair (z_t, z_f).

        Delegates to TileCoordinates so the result is guaranteed to match
        what the renderer and frontend expect.  Requires build_all() to
        have been called first.
        """
        assert self._n_time_frames is not None, "call build_all() first"
        tc = TileCoordinates(
            sample_rate=self.config.sample_rate,
            hop_size=self.config.hop_size,
            fft_size=self.config.fft_size,
            n_time_frames=self._n_time_frames,
            tile_size=self.config.tile_size,
            zoom_t_max=self.config.zoom_t_max,
            zoom_f_max=self.config.zoom_f_max,
        )
        return tc.tile_extent(z_t, z_f)

    # ── Downsampling ──────────────────────────────────────────────────────

    def _downsample_time(self, src: np.ndarray) -> np.ndarray:
        """
        2× downsample *src* along axis 0 (time).

        Pads with the noise floor if the time dimension is odd.
        Returns shape (ceil(n_t / 2), n_f).
        """
        n_t, n_f = src.shape
        if n_t % 2:
            pad = np.full((1, n_f), _NOISE_FLOOR, dtype=np.float32)
            src = np.concatenate([src, pad], axis=0)
        blocks = src.reshape(-1, 2, n_f)
        return self._reduce(blocks, axis=1)

    def _downsample_freq(self, src: np.ndarray) -> np.ndarray:
        """
        2× downsample *src* along axis 1 (frequency).

        Pads with the noise floor if the frequency dimension is odd.
        Returns shape (n_t, ceil(n_f / 2)).
        """
        n_t, n_f = src.shape
        if n_f % 2:
            pad = np.full((n_t, 1), _NOISE_FLOOR, dtype=np.float32)
            src = np.concatenate([src, pad], axis=1)
        blocks = src.reshape(n_t, -1, 2)
        return self._reduce(blocks, axis=2)

    # ── Internal helpers ──────────────────────────────────────────────────

    def _reduce(self, blocks: np.ndarray, axis: int) -> np.ndarray:
        """Apply configured downsample_method along *axis*.

        'mean' averages in linear amplitude space then converts back to dB,
        so that noise-floor values (-200 dB ≈ 0 amplitude) do not drag down
        the result the way a direct dB mean would.
        """
        if self.config.downsample_method == "max":
            return blocks.max(axis=axis).astype(np.float32)
        # Convert dB → linear amplitude, average, convert back to dB.
        # The stored values are 20·log10(|X| + ε), so the inverse is 10^(dB/20).
        linear = np.power(10.0, blocks / 20.0)
        mean_linear = linear.mean(axis=axis)
        return (20.0 * np.log10(mean_linear)).astype(np.float32)

    def _build_full_res(self, stft_array: zarr.Array) -> None:
        """Copy the STFT array into the pyramid as (zt_max, zf_max)."""
        key = f"zt{self.config.zoom_t_max}_zf{self.config.zoom_f_max}"
        group = self._group

        if key in group and group[key].shape == stft_array.shape:
            logger.info("  %s skipped (exists)", key)
            return

        logger.info("  %s  copying full-res STFT %s", key, stft_array.shape)
        t0 = time.perf_counter()

        dst = self._write_empty(group, key, stft_array.shape)
        chunk_t = self.config.chunk_size_frames
        for start in range(0, stft_array.shape[0], chunk_t):
            end = min(start + chunk_t, stft_array.shape[0])
            dst[start:end] = stft_array[start:end]

        logger.info("    done in %.2fs", time.perf_counter() - t0)

    def _expected_shape(
        self, src_shape: tuple[int, int], direction: str
    ) -> tuple[int, int]:
        n_t, n_f = src_shape
        if direction == "time":
            return (math.ceil(n_t / 2), n_f)
        return (n_t, math.ceil(n_f / 2))

    def _write(self, group: zarr.Group, key: str, data: np.ndarray) -> zarr.Array:
        """Write *data* to *group[key]*, overwriting if present."""
        dst = self._write_empty(group, key, data.shape)
        dst[:] = data
        return dst

    def _write_empty(
        self,
        group: zarr.Group,
        key: str,
        shape: tuple[int, ...],
    ) -> zarr.Array:
        tile = self.config.tile_size
        chunks = (min(tile, shape[0]), min(tile, shape[1]))
        return group.require_dataset(
            key,
            shape=shape,
            chunks=chunks,
            dtype="float32",
            fill_value=_NOISE_FLOOR,
            overwrite=True,
        )
