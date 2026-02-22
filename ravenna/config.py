from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class PipelineConfig:
    # ── Audio source ──────────────────────────────────────────────────────
    source_uri: str           # e.g. 's3://bucket/prefix', '/data/wav', 'https://...'
    date_start: datetime      # inclusive; UTC
    date_end: datetime        # exclusive; UTC
    sample_rate: int          # Hz; must match source audio
    file_pattern: str = "*.wav"                  # glob pattern for audio filenames
    source_credentials: dict = field(default_factory=dict)  # provider-specific

    # ── STFT ──────────────────────────────────────────────────────────────
    fft_size: int = 512       # samples per FFT window
    hop_size: int = 128       # samples between consecutive windows
    window: str = "hann"      # window function name

    # ── Pyramid ───────────────────────────────────────────────────────────
    tile_size: int = 256      # pixels; applied to both axes
    zoom_t_min: int = 0       # coarsest time zoom level
    zoom_t_max: int = 12      # finest time zoom level
    zoom_f_min: int = 0       # coarsest frequency zoom level
    zoom_f_max: int = 6       # finest frequency zoom level
    downsample_method: str = "mean"   # 'mean' or 'max'

    # ── Normalization ─────────────────────────────────────────────────────
    norm_strategy: str = "global_percentile"   # 'global_percentile' or 'calibrated_spl'
    norm_low_pct: float = 1.0      # percentile mapped to minimum display value
    norm_high_pct: float = 99.5    # percentile mapped to maximum display value
    hydrophone_sensitivity_db: float | None = None   # dB re 1 V/µPa
    spl_display_min_db: float | None = None          # dB re 1 µPa → maps to black
    spl_display_max_db: float | None = None          # dB re 1 µPa → maps to white

    # ── Colormap ──────────────────────────────────────────────────────────
    colormap: str = "viridis"   # any matplotlib colormap name

    # ── I/O paths ─────────────────────────────────────────────────────────
    zarr_path: str = "./zarr/full_res"
    pyramid_path: str = "./zarr/pyramid"
    tiles_path: str = "./output/tiles"   # root of flat tile tree: tiles/zt{}/zf{}/{x}/{y}.png

    # ── Compute ───────────────────────────────────────────────────────────
    n_workers: int = 16
    chunk_size_frames: int = 4096

    def to_json(self) -> str:
        d = dataclasses.asdict(self)
        d["date_start"] = self.date_start.isoformat()
        d["date_end"] = self.date_end.isoformat()
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, s: str) -> PipelineConfig:
        d = json.loads(s)
        d["date_start"] = datetime.fromisoformat(d["date_start"])
        d["date_end"] = datetime.fromisoformat(d["date_end"])
        return cls(**d)
