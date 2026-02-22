"""
Unit tests for FilesystemIngester (issue 4).

Tests use synthetic WAV files written to pytest's tmp_path fixture so
no real audio data is required.  soundfile is used for both writing and
reading, which mirrors production usage.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from ravenna.config import PipelineConfig
from ravenna.ingest.base import AudioFile, Gap
from ravenna.ingest.filesystem import FilesystemIngester, _parse_stem

# ── Shared constants ──────────────────────────────────────────────────────

SR = 8_000          # small sample rate keeps test WAVs tiny
HOP = 128
CHUNK = 1_024       # chunk_size used in iter_chunks calls

T0 = datetime(2024, 3, 15, 0, 0, 0, tzinfo=timezone.utc)   # date_start
T1 = datetime(2024, 3, 15, 0, 1, 0, tzinfo=timezone.utc)   # date_end (1 min window)


# ── Helpers ───────────────────────────────────────────────────────────────

def _write_wav(path: Path, n_samples: int, sr: int = SR) -> None:
    """Write a mono float32 WAV with random samples."""
    rng = np.random.default_rng(seed=0)
    data = rng.uniform(-1.0, 1.0, n_samples).astype(np.float32)
    sf.write(str(path), data, sr)


def _make_config(
    root: Path,
    date_start: datetime = T0,
    date_end: datetime = T1,
    fmt: str | None = None,
) -> PipelineConfig:
    return PipelineConfig(
        source_uri=str(root),
        date_start=date_start,
        date_end=date_end,
        sample_rate=SR,
        hop_size=HOP,
        filename_timestamp_format=fmt,
    )


# ── _parse_stem ───────────────────────────────────────────────────────────

class TestParseStem:
    def test_regex_fallback_iso_compact(self):
        dt = _parse_stem("MARS_20240315T000000", None)
        assert dt == datetime(2024, 3, 15, 0, 0, 0, tzinfo=timezone.utc)

    def test_regex_fallback_prefix_only(self):
        dt = _parse_stem("20240315T000000", None)
        assert dt == datetime(2024, 3, 15, 0, 0, 0, tzinfo=timezone.utc)

    def test_regex_fallback_no_match_returns_none(self):
        assert _parse_stem("no_timestamp_here", None) is None

    def test_explicit_format(self):
        dt = _parse_stem("20240315T000000", "%Y%m%dT%H%M%S")
        assert dt == datetime(2024, 3, 15, 0, 0, 0, tzinfo=timezone.utc)

    def test_explicit_format_mismatch_returns_none(self):
        assert _parse_stem("not_a_date", "%Y%m%dT%H%M%S") is None

    def test_result_is_utc(self):
        dt = _parse_stem("20240315T000000", None)
        assert dt.tzinfo is not None
        assert dt.utcoffset() == timedelta(0)


# ── list_files ────────────────────────────────────────────────────────────

class TestListFiles:
    def test_finds_file_in_window(self, tmp_path):
        wav = tmp_path / "20240315T000000.wav"
        _write_wav(wav, SR)
        cfg = _make_config(tmp_path)
        files = FilesystemIngester(cfg).list_files()
        assert len(files) == 1
        assert files[0].uri == str(wav)

    def test_skips_file_before_window(self, tmp_path):
        _write_wav(tmp_path / "20240314T235959.wav", SR)
        cfg = _make_config(tmp_path)
        assert FilesystemIngester(cfg).list_files() == []

    def test_skips_file_at_or_after_date_end(self, tmp_path):
        _write_wav(tmp_path / "20240315T000100.wav", SR)   # == date_end
        cfg = _make_config(tmp_path)
        assert FilesystemIngester(cfg).list_files() == []

    def test_skips_unparseable_filename(self, tmp_path):
        _write_wav(tmp_path / "no_timestamp.wav", SR)
        cfg = _make_config(tmp_path)
        assert FilesystemIngester(cfg).list_files() == []

    def test_sorted_by_start_time(self, tmp_path):
        _write_wav(tmp_path / "20240315T000030.wav", SR)
        _write_wav(tmp_path / "20240315T000000.wav", SR)
        cfg = _make_config(tmp_path)
        files = FilesystemIngester(cfg).list_files()
        assert files[0].start_time < files[1].start_time

    def test_returns_audio_file_metadata(self, tmp_path):
        _write_wav(tmp_path / "20240315T000000.wav", SR * 2)
        cfg = _make_config(tmp_path)
        f = FilesystemIngester(cfg).list_files()[0]
        assert isinstance(f, AudioFile)
        assert f.n_samples == SR * 2
        assert f.sample_rate == SR

    def test_explicit_format(self, tmp_path):
        wav = tmp_path / "20240315T000000.wav"
        _write_wav(wav, SR)
        cfg = _make_config(tmp_path, fmt="%Y%m%dT%H%M%S")
        files = FilesystemIngester(cfg).list_files()
        assert len(files) == 1

    def test_skips_non_matching_extension(self, tmp_path):
        _write_wav(tmp_path / "20240315T000000.flac", SR)
        cfg = _make_config(tmp_path)   # file_pattern defaults to *.wav
        assert FilesystemIngester(cfg).list_files() == []


# ── detect_gaps ───────────────────────────────────────────────────────────

class TestDetectGaps:
    def _ingester(self, tmp_path):
        return FilesystemIngester(_make_config(tmp_path))

    def test_no_files_one_big_gap(self, tmp_path):
        ing = self._ingester(tmp_path)
        gaps = ing.detect_gaps([])
        assert len(gaps) == 1
        assert gaps[0].start_time == T0
        assert gaps[0].end_time == T1

    def test_full_coverage_no_gaps(self, tmp_path):
        # One file covers the entire window (60 s at SR samples/s)
        af = AudioFile(
            uri="x.wav",
            start_time=T0,
            n_samples=SR * 60,
            sample_rate=SR,
        )
        ing = self._ingester(tmp_path)
        assert ing.detect_gaps([af]) == []

    def test_gap_before_first_file(self, tmp_path):
        af = AudioFile(
            uri="x.wav",
            start_time=T0 + timedelta(seconds=10),
            n_samples=SR * 50,
            sample_rate=SR,
        )
        ing = self._ingester(tmp_path)
        gaps = ing.detect_gaps([af])
        assert len(gaps) == 1
        assert gaps[0].start_time == T0
        assert gaps[0].end_time == T0 + timedelta(seconds=10)

    def test_gap_between_files(self, tmp_path):
        af1 = AudioFile("a.wav", T0, SR * 10, SR)
        af2 = AudioFile("b.wav", T0 + timedelta(seconds=20), SR * 40, SR)
        ing = self._ingester(tmp_path)
        gaps = ing.detect_gaps([af1, af2])
        assert any(
            g.start_time == T0 + timedelta(seconds=10)
            and g.end_time == T0 + timedelta(seconds=20)
            for g in gaps
        )

    def test_gap_after_last_file(self, tmp_path):
        af = AudioFile("x.wav", T0, SR * 30, SR)   # covers first 30 s
        ing = self._ingester(tmp_path)
        gaps = ing.detect_gaps([af])
        assert len(gaps) == 1
        assert gaps[0].start_time == T0 + timedelta(seconds=30)
        assert gaps[0].end_time == T1

    def test_gap_duration_seconds(self, tmp_path):
        ing = self._ingester(tmp_path)
        gaps = ing.detect_gaps([])
        assert gaps[0].duration_seconds == pytest.approx(60.0)

    def test_gap_is_frozen_dataclass(self, tmp_path):
        ing = self._ingester(tmp_path)
        gap = ing.detect_gaps([])[0]
        assert isinstance(gap, Gap)
        with pytest.raises((AttributeError, TypeError)):
            gap.start_time = T0  # type: ignore[misc]


# ── iter_chunks ───────────────────────────────────────────────────────────

class TestIterChunks:
    def test_yields_audio_chunks(self, tmp_path):
        _write_wav(tmp_path / "20240315T000000.wav", SR * 2)
        cfg = _make_config(tmp_path)
        chunks = list(FilesystemIngester(cfg).iter_chunks(CHUNK))
        assert len(chunks) > 0

    def test_real_chunk_not_gap(self, tmp_path):
        _write_wav(tmp_path / "20240315T000000.wav", SR * 2)
        cfg = _make_config(tmp_path)
        chunks = [c for c in FilesystemIngester(cfg).iter_chunks(CHUNK) if not c.is_gap]
        assert len(chunks) > 0
        assert all(not c.is_gap for c in chunks)

    def test_gap_chunk_is_gap(self, tmp_path):
        # No files → entire window is a gap
        cfg = _make_config(tmp_path)
        chunks = list(FilesystemIngester(cfg).iter_chunks(CHUNK))
        assert all(c.is_gap for c in chunks)

    def test_gap_chunk_samples_are_zeros(self, tmp_path):
        cfg = _make_config(tmp_path)
        for chunk in FilesystemIngester(cfg).iter_chunks(CHUNK):
            assert np.all(chunk.samples == 0.0)

    def test_chunk_size_respected(self, tmp_path):
        _write_wav(tmp_path / "20240315T000000.wav", SR * 3)
        cfg = _make_config(tmp_path)
        chunks = [c for c in FilesystemIngester(cfg).iter_chunks(CHUNK) if not c.is_gap]
        # All but possibly the last chunk should be exactly CHUNK samples
        for c in chunks[:-1]:
            assert len(c.samples) == CHUNK

    def test_start_frame_increases(self, tmp_path):
        _write_wav(tmp_path / "20240315T000000.wav", SR * 3)
        cfg = _make_config(tmp_path)
        chunks = [c for c in FilesystemIngester(cfg).iter_chunks(CHUNK) if not c.is_gap]
        frames = [c.start_frame for c in chunks]
        assert frames == sorted(frames)
        assert frames[0] < frames[-1]

    def test_first_chunk_start_frame_zero(self, tmp_path):
        # File starts at date_start → start_frame of first chunk must be 0
        _write_wav(tmp_path / "20240315T000000.wav", SR * 2)
        cfg = _make_config(tmp_path)
        first = next(FilesystemIngester(cfg).iter_chunks(CHUNK))
        assert first.start_frame == 0

    def test_samples_dtype_float32(self, tmp_path):
        _write_wav(tmp_path / "20240315T000000.wav", SR * 2)
        cfg = _make_config(tmp_path)
        for chunk in FilesystemIngester(cfg).iter_chunks(CHUNK):
            assert chunk.samples.dtype == np.float32
            break

    def test_gap_interleaved_with_files(self, tmp_path):
        # File 1: 0–10 s; gap 10–20 s; File 2: 20–60 s
        _write_wav(tmp_path / "20240315T000000.wav", SR * 10)
        _write_wav(tmp_path / "20240315T000020.wav", SR * 40)
        cfg = _make_config(tmp_path)
        chunks = list(FilesystemIngester(cfg).iter_chunks(CHUNK))
        gap_chunks = [c for c in chunks if c.is_gap]
        assert len(gap_chunks) > 0
        # Gap samples are zeros
        for gc in gap_chunks:
            assert np.all(gc.samples == 0.0)


# ── End-to-end filename format tests ─────────────────────────────────────

# Three consecutive 10-minute files starting 2018-02-14 22:16:45 UTC.
_MARS_T = [
    datetime(2018, 2, 14, 22, 16, 45, tzinfo=timezone.utc),
    datetime(2018, 2, 14, 22, 26, 45, tzinfo=timezone.utc),
    datetime(2018, 2, 14, 22, 36, 45, tzinfo=timezone.utc),
]
_MARS_N = SR * 60 * 10   # 10 minutes of audio at the test sample rate


def test_ingest_standard_mars_format(tmp_path):
    """
    Three WAV files named MARS_YYYYMMDD_HHMMSS.wav (real MARS convention).
    Verify count, start times, and sample counts.
    """
    names = [
        "MARS_20180214_221645.wav",
        "MARS_20180214_222645.wav",
        "MARS_20180214_223645.wav",
    ]
    for name in names:
        _write_wav(tmp_path / name, _MARS_N)

    cfg = PipelineConfig(
        source_uri=str(tmp_path),
        date_start=_MARS_T[0],
        date_end=datetime(2018, 2, 14, 23, 0, 0, tzinfo=timezone.utc),
        sample_rate=SR,
        file_pattern="*.wav",
        filename_timestamp_format="MARS_%Y%m%d_%H%M%S",
    )

    files = FilesystemIngester(cfg).list_files()

    assert len(files) == 3
    assert files[0].start_time == _MARS_T[0]
    assert files[1].start_time == _MARS_T[1]
    assert files[2].start_time == _MARS_T[2]
    assert all(f.n_samples == _MARS_N for f in files)


def test_ingest_m_d_y_h_m_s_format(tmp_path):
    """
    Three WAV files named MM-DD-YYYY-HH-MM-SS.wav (custom format).
    Verify count, start times, and sample counts.
    """
    names = [
        "02-14-2018-22-16-45.wav",
        "02-14-2018-22-26-45.wav",
        "02-14-2018-22-36-45.wav",
    ]
    for name in names:
        _write_wav(tmp_path / name, _MARS_N)

    cfg = PipelineConfig(
        source_uri=str(tmp_path),
        date_start=_MARS_T[0],
        date_end=datetime(2018, 2, 14, 23, 0, 0, tzinfo=timezone.utc),
        sample_rate=SR,
        file_pattern="*.wav",
        filename_timestamp_format="%m-%d-%Y-%H-%M-%S",
    )

    files = FilesystemIngester(cfg).list_files()

    assert len(files) == 3
    assert files[0].start_time == _MARS_T[0]
    assert files[1].start_time == _MARS_T[1]
    assert files[2].start_time == _MARS_T[2]
    assert all(f.n_samples == _MARS_N for f in files)
