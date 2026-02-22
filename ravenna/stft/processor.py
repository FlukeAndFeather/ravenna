"""
STFTProcessor — Stage 2 of the Ravenna pipeline.

Streams AudioChunks from an Ingester, computes the Short-Time Fourier
Transform of each chunk, converts magnitude to dB, and writes the result
to a Zarr array on disk.

Output array
------------
  Path   : config.zarr_path
  Shape  : (n_time_frames, n_freq_bins)   — time-major
  Dtype  : float32
  Chunks : (config.chunk_size_frames, n_freq_bins)
  Units  : dB (20 · log₁₀(|STFT| + ε))

The array is pre-allocated from config.date_start / date_end, then filled
sequentially.  If the array already exists at zarr_path with the expected
shape the stage is a no-op (idempotent).

Carry-over buffer
-----------------
scipy.signal.stft requires fft_size samples to produce the first frame, so
fft_size – hop_size "warm-up" samples are prepended to every chunk from the
tail of the previous chunk.  This means every chunk (including gap chunks)
is windowed correctly at its boundary.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import scipy.signal
import zarr

from ravenna.config import PipelineConfig

if TYPE_CHECKING:
    from ravenna.ingest.base import Ingester


class STFTProcessor:
    """
    Compute the full-resolution STFT and persist it as a Zarr array.

    Parameters
    ----------
    config : PipelineConfig
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.n_freq_bins: int = config.fft_size // 2 + 1

    # ── Public API ────────────────────────────────────────────────────────

    def process(self, ingester: "Ingester") -> zarr.Array:
        """
        Stream all chunks from *ingester*, compute their STFT, and write
        to Zarr.  Returns the populated Zarr array.

        If the Zarr array already exists at the configured path with the
        expected shape this call is a no-op and returns the existing array.
        """
        n_time_frames = self._n_time_frames()
        z = self._open_zarr(n_time_frames)

        # Idempotency: non-empty array with correct shape → already done.
        if z.nchunks_initialized > 0:
            return z

        chunk_samples = self.config.chunk_size_frames * self.config.hop_size
        overlap = self.config.fft_size - self.config.hop_size
        carryover = np.zeros(overlap, dtype=np.float32)
        write_frame = 0

        for audio_chunk in ingester.iter_chunks(chunk_samples):
            x = np.concatenate([carryover, audio_chunk.samples])
            frames = self._stft_to_db(x)          # (n_frames, n_freq_bins)
            n_frames = frames.shape[0]

            end_frame = min(write_frame + n_frames, n_time_frames)
            n_write = end_frame - write_frame
            if n_write > 0:
                z[write_frame:end_frame, :] = frames[:n_write]
            write_frame = end_frame

            # Carry over the last `overlap` samples for correct windowing of
            # the next chunk boundary.
            carryover = x[-overlap:] if len(x) >= overlap else np.zeros(overlap, dtype=np.float32)

            if write_frame >= n_time_frames:
                break

        return z

    # ── Internal helpers ──────────────────────────────────────────────────

    def _n_time_frames(self) -> int:
        """Total STFT frames for the configured date range."""
        total_sec = (self.config.date_end - self.config.date_start).total_seconds()
        total_samples = round(total_sec * self.config.sample_rate)
        return math.ceil(total_samples / self.config.hop_size)

    def _open_zarr(self, n_time_frames: int) -> zarr.Array:
        """Open an existing Zarr array or create a new one."""
        shape = (n_time_frames, self.n_freq_bins)
        chunks = (self.config.chunk_size_frames, self.n_freq_bins)
        path = self.config.zarr_path

        try:
            z = zarr.open_array(path, mode="r+")
            if z.shape == shape:
                return z
        except Exception:
            pass

        return zarr.open_array(
            path,
            mode="w",
            shape=shape,
            chunks=chunks,
            dtype="float32",
            fill_value=-200.0,   # unwritten frames read as noise floor, not 0
        )

    def _stft_to_db(self, x: np.ndarray) -> np.ndarray:
        """
        Compute STFT magnitude in dB on *x*.

        Parameters
        ----------
        x : 1-D float32 array of audio samples

        Returns
        -------
        np.ndarray, shape (n_frames, n_freq_bins), dtype float32
        """
        _, _, Zxx = scipy.signal.stft(
            x,
            fs=self.config.sample_rate,
            window=self.config.window,
            nperseg=self.config.fft_size,
            noverlap=self.config.fft_size - self.config.hop_size,
            boundary=None,
            padded=False,
        )
        # Zxx shape: (n_freq_bins, n_frames) → transpose to time-major
        db = 20.0 * np.log10(np.abs(Zxx) + 1e-10).astype(np.float32)
        return db.T
