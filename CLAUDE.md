# Ravenna — Tiled Bioacoustic Viewer

A general-purpose pipeline for visualizing long-duration passive acoustic monitoring (PAM) datasets as interactive tiled spectrograms in a web browser. Concept is analogous to a Cloud-Optimized GeoTIFF (COG) applied to the time-frequency domain.

**Full design spec:** `docs/phase1-development-plan.docx`

---

## What This Is

A five-stage offline processing pipeline that takes WAV/FLAC audio files and produces a PMTiles archive of spectrogram tiles, served to a MapLibre GL JS frontend. The viewer allows seamless zoom and pan across the full dataset — from the entire archive on one screen down to sub-second resolution — with no server-side rendering at query time.

---

## Architecture: Five-Stage Linear Pipeline

Each stage has a single input and output contract. No stage reads from any stage other than its immediate predecessor.

1. **Ingest** — enumerate audio files from source (local, S3, HTTP); stream as `AudioChunk` objects; detect and synthesize silence for data gaps
2. **STFT** — Short-Time Fourier Transform each chunk; write magnitude in dB to a Zarr array on disk
3. **Pyramid build** — progressively 2× downsample the full-resolution Zarr into Z_max → Z0; one Zarr array per zoom level
4. **Render** — read each (z, x, y) tile from pyramid Zarr; normalize; apply colormap; write 256×256 PNG; embarrassingly parallel via `ProcessPoolExecutor`
5. **Package** — assemble PNG tiles into a PMTiles archive (production) or MBTiles/SQLite (local dev); sort by Hilbert curve index

---

## Repo Structure

```
ravenna/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── docs/
│   └── phase1-development-plan.docx
├── ravenna/
│   ├── config.py           # PipelineConfig dataclass — all parameters live here
│   ├── coordinates.py      # TileCoordinates — all z/x/y ↔ time/freq math
│   ├── ingest/
│   │   ├── base.py         # Ingester ABC, AudioChunk, AudioFile, Gap dataclasses
│   │   ├── filesystem.py
│   │   └── s3.py
│   ├── stft/
│   │   └── processor.py    # STFTProcessor
│   ├── pyramid/
│   │   └── builder.py      # PyramidBuilder, TileExtent
│   ├── render/
│   │   ├── renderer.py     # TileRenderer
│   │   ├── norm.py         # NormStrategy ABC, GlobalPercentileNorm, CalibratedSPLNorm
│   │   └── colormap.py     # ColormapStrategy
│   ├── package/
│   │   └── packager.py     # TilePackager
│   └── pipeline/
│       └── orchestrator.py # PipelineOrchestrator — imports from all stage packages
├── cli/
│   └── main.py             # click entry point; `ravenna run`, `ravenna status`
├── frontend/
│   └── index.html          # single-file MapLibre GL JS viewer
└── tests/
    ├── conftest.py
    ├── unit/
    └── integration/
```

`pipeline/` is the only package that imports from all other packages. Nothing imports from `pipeline/`. The dependency graph is a strict DAG.

---

## Key Parameters (Phase 1 defaults)

| Parameter | Value | Notes |
|---|---|---|
| FFT size | 512 samples | 2 ms window; 500 Hz/bin |
| Hop size | 128 samples | 0.5 ms/frame; 4× overlap |
| Window | Hann | |
| Tile size | 256 × 256 px | |
| Zoom range | Z0 – Z12 | Overview pyramid only (Phase 1) |
| Z12 resolution | ~0.6 sec/px | Finest overview zoom |
| Z0 resolution | ~2.4 hr/px | Full archive on one screen |
| dB reference | 1 µPa (underwater) | Configurable |
| Output format | PMTiles (prod) / MBTiles (dev) | |

---

## Tile Coordinate System

- **X axis** = time (tile 0 = archive start)
- **Y axis** = frequency (tile 0 = 0 Hz, increases toward Nyquist)
- Y=0 at the bottom — inverse of geographic map convention; MapLibre Y axis is inverted in the frontend to compensate
- All coordinate math lives in `TileCoordinates` (`ravenna/coordinates.py`). Do not put coordinate arithmetic anywhere else.
- `TileCoordinates` has no external dependencies — pure arithmetic only.

---

## Intermediate Storage

- Full-resolution STFT → Zarr array, shape `[n_time_frames, n_freq_bins]`, chunked `[4096, n_freq_bins]`
- Pyramid → Zarr group, one array per zoom level
- Chunks align with tile boundaries — a tile read is always exactly one Zarr chunk read
- All stages are **idempotent**: existing Zarr chunks and PNG tiles are skipped on re-run
- Zarr intermediate is temporary; delete after PMTiles archive is validated

---

## Normalization and Display

- `NormStrategy` maps raw dB values → `[0, 1]` before colormap application
- Two implementations:
  - `GlobalPercentileNorm` — fits low/high dB percentiles from a random sample of ~10K tiles drawn across the full pyramid before rendering begins
  - `CalibratedSPLNorm` — maps absolute Sound Pressure Level (SPL, dB re 1 µPa) to display range using hydrophone sensitivity from calibration file
- Norm parameters are serialized into PMTiles metadata JSON at packaging time
- The frontend `ColormapLegend` reads norm params from PMTiles metadata and renders a dB-annotated color scale bar — this is the sole mechanism for reading pixel values in Phase 1
- `NormStrategy` instances must be serializable (to_dict/from_dict) for safe passing to worker processes

---

## Case Study: MBARI MARS Archive

The reference dataset is the MBARI MARS 256 kHz hydrophone archive, publicly available on AWS S3 (`pacific-sound-256khz-{year}` buckets, us-west-2, no credentials required). Run pipeline compute in us-west-2 to avoid S3 egress fees.

- 10-minute WAV files, 24-bit PCM, mono
- ~460 GB/week; ~151 TB total
- ~95% temporal coverage (gaps present)
- Phase 1 estimated output: ~1 GB PMTiles file for full 10-year Z0–Z12 pyramid

---

## Phase Roadmap

| Phase | Scope |
|---|---|
| **Phase 1 (current)** | Overview pyramid Z0–Z12; ColormapLegend dB scale; MBARI MARS case study |
| Phase 2 | Detail pyramid Z13–Z23; audio clip delivery linked to viewport |
| Phase 3 | Signal detection and annotation overlay |

---

## Testing Conventions

- Unit tests use synthetic audio (pure tones at known frequencies) and known dB values — no real dataset required
- `TileCoordinates` must pass round-trip property tests: `tile→time→tile` and `tile→freq→tile` must be exact identity at all zoom levels
- Integration tests inject synthetic gaps to verify gap handling
- All pipeline stages must pass idempotency test: two runs on identical input produce bit-identical output
- Visual validation at each milestone: inspect known-signal time windows at multiple zoom levels
