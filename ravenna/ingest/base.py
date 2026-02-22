"""
Abstract base class and shared data types for Ravenna audio ingestors.

An Ingester scans an audio source (filesystem, S3, …) and yields
AudioChunk objects in chronological order.  Gaps in coverage are
represented by gap-flagged chunks so that downstream pipeline stages
can zero-pad the STFT output rather than silently skipping time.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass(frozen=True)
class AudioFile:
    """Metadata for a single audio file discovered by the ingester."""
    uri: str           # absolute path or URI
    start_time: datetime   # UTC timestamp of first sample
    n_samples: int         # total samples in the file
    sample_rate: int       # Hz


@dataclass
class AudioChunk:
    """A contiguous block of audio samples (or a silence-filled gap)."""
    samples: np.ndarray    # shape (n_samples,); dtype float32; zeros for gaps
    start_time: datetime   # UTC timestamp of first sample
    start_frame: int       # STFT frame index of first sample (= sample_offset // hop_size)
    sample_rate: int       # Hz
    is_gap: bool = False   # True → samples are zeros synthesised to fill a coverage gap


@dataclass(frozen=True)
class Gap:
    """A time interval with no audio coverage."""
    start_time: datetime
    end_time: datetime

    @property
    def duration_seconds(self) -> float:
        return (self.end_time - self.start_time).total_seconds()


class Ingester(ABC):
    """
    Abstract base class for audio source ingestors.

    Subclasses discover audio files within a date range, parse their
    timestamps, detect coverage gaps, and yield AudioChunk objects in
    chronological order.
    """

    @abstractmethod
    def list_files(self) -> list[AudioFile]:
        """
        Return all audio files within the configured date range,
        sorted by start_time ascending.
        """

    @abstractmethod
    def detect_gaps(self, files: list[AudioFile]) -> list[Gap]:
        """
        Identify time intervals not covered by any file in *files*.

        The returned list is sorted by start_time ascending.
        """

    @abstractmethod
    def iter_chunks(self, chunk_size: int) -> "Iterator[AudioChunk]":  # noqa: F821
        """
        Yield AudioChunk objects in chronological order.

        Real audio is yielded first; gap chunks (is_gap=True) are
        inserted wherever coverage is missing.

        Parameters
        ----------
        chunk_size : int
            Number of samples per chunk (the final chunk of each file
            or gap may be shorter).
        """
