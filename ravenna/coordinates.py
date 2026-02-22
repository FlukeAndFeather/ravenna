"""
Tile coordinate system for the Ravenna spectrogram pyramid.

Tiles are addressed by four coordinates: (z_t, z_f, x, y).

  z_t — time zoom level.      Higher = finer time resolution.
  z_f — frequency zoom level. Higher = finer frequency resolution.
  x   — tile column along the time axis      (x=0 = archive start)
  y   — tile row along the frequency axis    (y=0 = 0 Hz, increases toward Nyquist)

The two zoom axes are fully independent. Zooming in on time (increasing z_t)
does not affect the frequency resolution, and vice versa.

Y=0 is at the bottom, which is the inverse of geographic map convention.
The MapLibre frontend compensates by inverting its Y axis.

No external dependencies. All arithmetic is pure Python.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TileExtent:
    """Number of tiles in each dimension at a given (z_t, z_f) zoom pair."""
    z_t: int   # time zoom level
    z_f: int   # frequency zoom level
    n_x: int   # tile columns (time axis)
    n_y: int   # tile rows  (frequency axis)


class TileCoordinates:
    """
    Converts between tile (z_t, z_f, x, y) addresses and physical
    (time, frequency) coordinates. All conversions are invertible.

    At (z_t_max, z_f_max), one tile pixel = one STFT frame in time and one
    frequency bin in frequency. Each step down in z_t doubles the time span
    per pixel; each step down in z_f doubles the frequency span per pixel.
    The two axes are fully independent.

    Parameters
    ----------
    sample_rate : int
        Audio sample rate in Hz.
    hop_size : int
        STFT hop size in samples (samples between consecutive STFT frames).
    fft_size : int
        FFT window size in samples. Determines frequency bin width and the
        number of frequency bins (fft_size // 2 + 1).
    n_time_frames : int
        Total number of STFT frames in the full-resolution array. Required
        to compute tile extents via tile_extent().
    tile_size : int
        Pixels per tile side (applied to both axes). Default 256.
    zoom_t_max : int
        Finest time zoom level. Default 12.
    zoom_f_max : int
        Finest frequency zoom level. Default 6.
    """

    def __init__(
        self,
        sample_rate: int,
        hop_size: int,
        fft_size: int,
        n_time_frames: int,
        tile_size: int = 256,
        zoom_t_max: int = 12,
        zoom_f_max: int = 6,
    ) -> None:
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.fft_size = fft_size
        self.n_time_frames = n_time_frames
        self.tile_size = tile_size
        self.zoom_t_max = zoom_t_max
        self.zoom_f_max = zoom_f_max
        self.n_freq_bins = fft_size // 2 + 1

    # ── Internal helpers ──────────────────────────────────────────────────

    def _frames_per_pixel(self, z_t: int) -> int:
        """STFT frames represented by one pixel at time zoom level z_t."""
        return 2 ** (self.zoom_t_max - z_t)

    def _bins_per_pixel(self, z_f: int) -> int:
        """Frequency bins represented by one pixel at frequency zoom level z_f."""
        return 2 ** (self.zoom_f_max - z_f)

    # ── Tile → physical ───────────────────────────────────────────────────

    def tile_to_time_range(self, z_t: int, x: int) -> tuple[float, float]:
        """Return (start_sec, end_sec) for tile column x at time zoom level z_t."""
        fpp = self._frames_per_pixel(z_t)
        start_frame = x * self.tile_size * fpp
        end_frame = (x + 1) * self.tile_size * fpp
        sec_per_frame = self.hop_size / self.sample_rate
        return start_frame * sec_per_frame, end_frame * sec_per_frame

    def tile_to_freq_range(self, z_f: int, y: int) -> tuple[float, float]:
        """Return (low_hz, high_hz) for tile row y at frequency zoom level z_f."""
        bpp = self._bins_per_pixel(z_f)
        low_bin = y * self.tile_size * bpp
        high_bin = (y + 1) * self.tile_size * bpp
        hz_per_bin = self.sample_rate / self.fft_size
        return low_bin * hz_per_bin, high_bin * hz_per_bin

    # ── Physical → tile ───────────────────────────────────────────────────

    def time_to_tile(self, z_t: int, t_sec: float) -> int:
        """Return the tile column x that contains time t_sec at zoom level z_t."""
        fpp = self._frames_per_pixel(z_t)
        frame = t_sec * self.sample_rate / self.hop_size
        # Small epsilon guards against floating-point drift at exact tile
        # boundaries (e.g. the start of tile x mapping back to x-1).
        return math.floor(frame / (self.tile_size * fpp) + 1e-10)

    def freq_to_tile(self, z_f: int, f_hz: float) -> int:
        """Return the tile row y that contains frequency f_hz at zoom level z_f."""
        bpp = self._bins_per_pixel(z_f)
        bin_idx = f_hz * self.fft_size / self.sample_rate
        return math.floor(bin_idx / (self.tile_size * bpp) + 1e-10)

    # ── Tile grid extent ──────────────────────────────────────────────────

    def tile_extent(self, z_t: int, z_f: int) -> TileExtent:
        """Return the number of tile columns and rows at zoom pair (z_t, z_f)."""
        fpp = self._frames_per_pixel(z_t)
        bpp = self._bins_per_pixel(z_f)
        n_x = math.ceil(self.n_time_frames / (self.tile_size * fpp))
        n_y = math.ceil(self.n_freq_bins / (self.tile_size * bpp))
        return TileExtent(z_t=z_t, z_f=z_f, n_x=n_x, n_y=n_y)
