"""
Unit tests for STFTProcessor.

A FakeIngester yields controlled AudioChunks so tests don't touch the
filesystem or require real audio files.  Pure-tone signals let us assert
that energy appears at the expected frequency bin.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Iterator

import numpy as np
import pytest

from ravenna.config import PipelineConfig
from ravenna.ingest.base import AudioChunk, AudioFile, Gap, Ingester
from ravenna.stft.processor import STFTProcessor

# ── Constants ─────────────────────────────────────────────────────────────

SR = 8_000          # small sample rate keeps tests fast
FFT = 512
HOP = 128
TILE = 256
T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
DURATION_S = 5      # seconds of synthetic audio per test

N_FREQ_BINS = FFT // 2 + 1


# ── Helpers ───────────────────────────────────────────────────────────────

def _make_config(tmp_path, duration_s: float = DURATION_S) -> PipelineConfig:
    return PipelineConfig(
        source_uri=str(tmp_path),
        date_start=T0,
        date_end=T0 + timedelta(seconds=duration_s),
        sample_rate=SR,
        fft_size=FFT,
        hop_size=HOP,
        zarr_path=str(tmp_path / "stft.zarr"),
        chunk_size_frames=TILE,
    )


def _tone(freq_hz: float, duration_s: float, sr: int = SR) -> np.ndarray:
    """Return a mono float32 pure-tone array."""
    t = np.arange(round(duration_s * sr)) / sr
    return np.sin(2 * np.pi * freq_hz * t).astype(np.float32)


def _expected_bin(freq_hz: float, sr: int = SR, fft_size: int = FFT) -> int:
    """Closest FFT bin for freq_hz."""
    return round(freq_hz * fft_size / sr)


class FakeIngester(Ingester):
    """Yields a list of pre-built AudioChunks; gaps not needed here."""

    def __init__(self, chunks: list[AudioChunk]) -> None:
        self._chunks = chunks

    def list_files(self) -> list[AudioFile]:
        return []

    def detect_gaps(self, files: list[AudioFile]) -> list[Gap]:
        return []

    def iter_chunks(self, chunk_size: int) -> Iterator[AudioChunk]:
        # Re-chunk into the requested chunk_size
        buf = np.concatenate([c.samples for c in self._chunks])
        offset = 0
        frame = 0
        while offset < len(buf):
            n = min(chunk_size, len(buf) - offset)
            yield AudioChunk(
                samples=buf[offset : offset + n],
                start_time=T0 + timedelta(seconds=offset / SR),
                start_frame=frame,
                sample_rate=SR,
                is_gap=False,
            )
            offset += n
            frame += n // HOP


def _ingester_from_samples(samples: np.ndarray) -> FakeIngester:
    return FakeIngester([
        AudioChunk(
            samples=samples,
            start_time=T0,
            start_frame=0,
            sample_rate=SR,
            is_gap=False,
        )
    ])


# ── Shape and dtype ───────────────────────────────────────────────────────

def test_output_shape(tmp_path):
    cfg = _make_config(tmp_path)
    proc = STFTProcessor(cfg)
    ing = _ingester_from_samples(_tone(1000, DURATION_S))
    z = proc.process(ing)

    expected_frames = math.ceil(round(DURATION_S * SR) / HOP)
    assert z.shape == (expected_frames, N_FREQ_BINS)


def test_output_dtype_float32(tmp_path):
    cfg = _make_config(tmp_path)
    proc = STFTProcessor(cfg)
    ing = _ingester_from_samples(_tone(1000, DURATION_S))
    z = proc.process(ing)
    assert z.dtype == np.float32


def test_zarr_chunk_shape(tmp_path):
    cfg = _make_config(tmp_path)
    proc = STFTProcessor(cfg)
    ing = _ingester_from_samples(_tone(1000, DURATION_S))
    z = proc.process(ing)
    assert z.chunks == (TILE, N_FREQ_BINS)


# ── Frequency content ─────────────────────────────────────────────────────

@pytest.mark.parametrize("freq_hz", [500, 1000, 2000])
def test_pure_tone_peak_at_correct_bin(tmp_path, freq_hz):
    """A pure tone must produce maximum energy at the expected frequency bin."""
    cfg = _make_config(tmp_path)
    proc = STFTProcessor(cfg)
    ing = _ingester_from_samples(_tone(freq_hz, DURATION_S))
    z = proc.process(ing)

    # Average power across time; find peak bin
    mean_spectrum = z[:].mean(axis=0)
    peak_bin = int(np.argmax(mean_spectrum))
    expected = _expected_bin(freq_hz)
    # Allow ±1 bin of tolerance for windowing
    assert abs(peak_bin - expected) <= 1, (
        f"freq={freq_hz} Hz → expected bin {expected}, got {peak_bin}"
    )


def test_silence_near_noise_floor(tmp_path):
    """
    An all-zero input should produce uniformly low dB values
    (driven entirely by the log(epsilon) floor).
    """
    cfg = _make_config(tmp_path)
    proc = STFTProcessor(cfg)
    silence = np.zeros(round(DURATION_S * SR), dtype=np.float32)
    ing = _ingester_from_samples(silence)
    z = proc.process(ing)

    # log10(1e-10) * 20 = -200 dB; allow a couple dB above that
    assert z[:].max() < -190.0


# ── Idempotency ───────────────────────────────────────────────────────────

def test_idempotent_second_run_skipped(tmp_path):
    """Second call to process() must return the same array without rewriting."""
    cfg = _make_config(tmp_path)
    proc = STFTProcessor(cfg)
    samples = _tone(1000, DURATION_S)
    ing = _ingester_from_samples(samples)

    z1 = proc.process(ing)
    data1 = z1[:].copy()

    # Second run with a *different* signal — result must not change
    ing2 = _ingester_from_samples(_tone(500, DURATION_S))
    z2 = proc.process(ing2)
    data2 = z2[:]

    np.testing.assert_array_equal(data1, data2)


def test_idempotent_same_zarr_path(tmp_path):
    """Two STFTProcessor instances with the same zarr_path share the array."""
    cfg = _make_config(tmp_path)
    samples = _tone(1000, DURATION_S)
    ing = _ingester_from_samples(samples)

    z1 = STFTProcessor(cfg).process(ing)
    z2 = STFTProcessor(cfg).process(_ingester_from_samples(_tone(500, DURATION_S)))

    np.testing.assert_array_equal(z1[:], z2[:])


# ── Gap handling ──────────────────────────────────────────────────────────

def test_gap_chunk_produces_low_energy(tmp_path):
    """
    A gap chunk (zero-filled) must produce near-noise-floor dB values
    at all frequency bins.
    """
    cfg = _make_config(tmp_path)
    proc = STFTProcessor(cfg)

    n_samples = round(DURATION_S * SR)
    gap_chunk = AudioChunk(
        samples=np.zeros(n_samples, dtype=np.float32),
        start_time=T0,
        start_frame=0,
        sample_rate=SR,
        is_gap=True,
    )
    z = proc.process(FakeIngester([gap_chunk]))
    assert z[:].max() < -190.0


# ── _stft_to_db internals ─────────────────────────────────────────────────

def test_stft_to_db_shape(tmp_path):
    cfg = _make_config(tmp_path)
    proc = STFTProcessor(cfg)
    x = _tone(1000, 1.0)
    result = proc._stft_to_db(x)
    assert result.ndim == 2
    assert result.shape[1] == N_FREQ_BINS


def test_stft_to_db_dtype(tmp_path):
    cfg = _make_config(tmp_path)
    proc = STFTProcessor(cfg)
    result = proc._stft_to_db(_tone(1000, 1.0))
    assert result.dtype == np.float32


def test_stft_to_db_values_are_negative_db(tmp_path):
    """A unit-amplitude tone → dB values should be near 0 dB (not wildly wrong)."""
    cfg = _make_config(tmp_path)
    proc = STFTProcessor(cfg)
    result = proc._stft_to_db(_tone(1000, 1.0))
    # Peak bin should be negative-ish (windowing loss) but well above noise
    assert result.max() > -30.0
