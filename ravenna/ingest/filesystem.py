"""
FilesystemIngester — discovers audio files on a local filesystem.

Timestamp parsing
-----------------
If PipelineConfig.filename_timestamp_format is set it is used as a
strptime format string applied to the full filename stem
(e.g. "%Y%m%dT%H%M%S" → "20200315T143022").

If filename_timestamp_format is None the ingester falls back to a
regex that recognises the ISO-compact pattern YYYYMMDDTHHmmSS anywhere
in the stem (e.g. "MARS_20200315T143022.flac").

All timestamps are assumed to be UTC.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf

from ravenna.config import PipelineConfig
from ravenna.ingest.base import AudioChunk, AudioFile, Gap, Ingester

# Fallback regex: ISO-compact datetime anywhere in the filename stem
_ISO_RE = re.compile(r"(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})")


def _parse_stem(stem: str, fmt: str | None) -> datetime | None:
    """
    Parse a UTC datetime from a filename stem.

    Returns None if parsing fails (file is skipped by the caller).
    """
    if fmt is not None:
        try:
            dt = datetime.strptime(stem, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None

    m = _ISO_RE.search(stem)
    if m is None:
        return None
    y, mo, d, h, mi, s = (int(v) for v in m.groups())
    return datetime(y, mo, d, h, mi, s, tzinfo=timezone.utc)


class FilesystemIngester(Ingester):
    """
    Discovers audio files under a local directory tree.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration.  ``source_uri`` must be a local
        directory path; ``file_pattern`` is the glob pattern used to
        find audio files within that directory.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._root = Path(config.source_uri)

    # ── Ingester interface ────────────────────────────────────────────────

    def list_files(self) -> list[AudioFile]:
        """
        Glob *source_uri* for files matching *file_pattern*, parse their
        timestamps, filter to [date_start, date_end), and return sorted
        by start_time ascending.
        """
        fmt = self.config.filename_timestamp_format
        results: list[AudioFile] = []

        for path in sorted(self._root.rglob(self.config.file_pattern)):
            stem = path.stem
            start_time = _parse_stem(stem, fmt)
            if start_time is None:
                continue  # cannot determine timestamp → skip

            # Filter to configured date window
            if start_time < self.config.date_start:
                continue
            if start_time >= self.config.date_end:
                continue

            try:
                info = sf.info(str(path))
            except Exception:
                continue  # unreadable file → skip

            results.append(
                AudioFile(
                    uri=str(path),
                    start_time=start_time,
                    n_samples=info.frames,
                    sample_rate=info.samplerate,
                )
            )

        results.sort(key=lambda f: f.start_time)
        return results

    def detect_gaps(self, files: list[AudioFile]) -> list[Gap]:
        """
        Return a list of Gaps between consecutive files and between
        date_start / date_end and the first / last file.

        Only gaps of positive duration are returned.
        """
        gaps: list[Gap] = []
        if not files:
            gaps.append(Gap(self.config.date_start, self.config.date_end))
            return gaps

        # Gap before first file
        if files[0].start_time > self.config.date_start:
            gaps.append(Gap(self.config.date_start, files[0].start_time))

        # Gaps between consecutive files
        for prev, curr in zip(files, files[1:]):
            prev_end = _file_end_time(prev)
            if curr.start_time > prev_end:
                gaps.append(Gap(prev_end, curr.start_time))

        # Gap after last file
        last_end = _file_end_time(files[-1])
        if last_end < self.config.date_end:
            gaps.append(Gap(last_end, self.config.date_end))

        return gaps

    def iter_chunks(self, chunk_size: int) -> Iterator[AudioChunk]:
        """
        Yield AudioChunk objects in chronological order.

        Real audio chunks are interleaved with gap chunks (is_gap=True)
        wherever coverage is missing.  The sample offset accumulated
        from date_start is used to compute start_frame for each chunk.
        """
        files = self.list_files()
        gaps = self.detect_gaps(files)

        # Build a time-ordered event list: (start_time, kind, object)
        events: list[tuple[datetime, str, AudioFile | Gap]] = []
        for f in files:
            events.append((f.start_time, "file", f))
        for g in gaps:
            events.append((g.start_time, "gap", g))
        events.sort(key=lambda e: e[0])

        sr = self.config.sample_rate
        hop = self.config.hop_size
        origin = self.config.date_start

        for _, kind, obj in events:
            if kind == "file":
                af: AudioFile = obj  # type: ignore[assignment]
                yield from _read_file_chunks(af, chunk_size, sr, hop, origin)
            else:
                gap: Gap = obj  # type: ignore[assignment]
                yield from _gap_chunks(gap, chunk_size, sr, hop, origin)


# ── Helpers ───────────────────────────────────────────────────────────────


def _file_end_time(f: AudioFile) -> datetime:
    """UTC time of the sample immediately after the last sample in *f*."""
    duration_sec = f.n_samples / f.sample_rate
    from datetime import timedelta
    return f.start_time + timedelta(seconds=duration_sec)


def _sample_offset(t: datetime, origin: datetime, sample_rate: int) -> int:
    """Number of samples from *origin* to *t*."""
    return round((t - origin).total_seconds() * sample_rate)


def _read_file_chunks(
    af: AudioFile,
    chunk_size: int,
    target_sr: int,
    hop_size: int,
    origin: datetime,
) -> Iterator[AudioChunk]:
    """Yield chunks of float32 samples read from *af*."""
    sample_offset = _sample_offset(af.start_time, origin, target_sr)

    with sf.SoundFile(af.uri) as f:
        read_so_far = 0
        while True:
            block = f.read(chunk_size, dtype="float32", always_2d=False)
            if block.size == 0:
                break
            # Mix down to mono if necessary
            if block.ndim == 2:
                block = block.mean(axis=1)

            chunk_sample_offset = sample_offset + read_so_far
            yield AudioChunk(
                samples=block,
                start_time=af.start_time,
                start_frame=chunk_sample_offset // hop_size,
                sample_rate=target_sr,
                is_gap=False,
            )
            read_so_far += len(block)


def _gap_chunks(
    gap: Gap,
    chunk_size: int,
    sample_rate: int,
    hop_size: int,
    origin: datetime,
) -> Iterator[AudioChunk]:
    """Yield zero-filled chunks spanning *gap*."""
    n_gap_samples = round(gap.duration_seconds * sample_rate)
    sample_offset = _sample_offset(gap.start_time, origin, sample_rate)

    emitted = 0
    while emitted < n_gap_samples:
        n = min(chunk_size, n_gap_samples - emitted)
        chunk_sample_offset = sample_offset + emitted
        yield AudioChunk(
            samples=np.zeros(n, dtype=np.float32),
            start_time=gap.start_time,
            start_frame=chunk_sample_offset // hop_size,
            sample_rate=sample_rate,
            is_gap=True,
        )
        emitted += n
