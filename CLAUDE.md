# Ravenna — Tiled Bioacoustic Viewer

A general-purpose pipeline for visualizing long-duration passive acoustic monitoring (PAM) datasets as interactive tiled spectrograms in a web browser. Concept is analogous to a Cloud-Optimized GeoTIFF (COG) applied to the time-frequency domain.

**Full design spec:** `docs/phase1-development-plan.docx`

---

## What This Is

A five-stage offline processing pipeline that takes WAV/FLAC audio files and produces a flat tile archive of spectrogram PNG tiles, served to a MapLibre GL JS frontend. The viewer allows independent zoom and pan along the time axis and the frequency axis — from the entire archive on one screen down to sub-second, sub-kHz resolution — with no server-side rendering at query time.

---

## Architecture: Five-Stage Linear Pipeline

Each stage has a single input and output contract. No stage reads from any stage other than its immediate predecessor.

1. **Ingest** — enumerate audio files from source (local, S3, HTTP); stream as `AudioChunk` objects; detect and synthesize silence for data gaps
2. **STFT** — Short-Time Fourier Transform each chunk; write magnitude in dB to a Zarr array on disk
3. **Pyramid build** — build a 2D pyramid by independently downsampling the time axis (Z_t levels) and the frequency axis (Z_f levels); one Zarr array per `(z_t, z_f)` pair
4. **Render** — read each `(z_t, z_f, x, y)` tile from pyramid Zarr; normalize; apply colormap; write 256×256 PNG; embarrassingly parallel via `ProcessPoolExecutor`
5. **Package** — write rendered PNG tiles to a flat file tree: `tiles/zt{z_t}/zf{z_f}/{x}/{y}.png`; served as static files over HTTP

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
│   ├── coordinates.py      # TileCoordinates — all (z_t,z_f,x,y) ↔ time/freq math
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
| FFT size | 512 samples | 2 ms window; 500 Hz/bin at 256 kHz |
| Hop size | 128 samples | 0.5 ms/frame; 4× overlap |
| Window | Hann | |
| Tile size | 256 × 256 px | |
| Time zoom range | Z_t 0 – 12 | Z_t_max = finest time resolution |
| Freq zoom range | Z_f 0 – 6 | Z_f_max = finest frequency resolution |
| dB reference | 1 µPa (underwater) | Configurable |
| Output format | Flat file tree | `tiles/zt{}/zf{}/{x}/{y}.png` |

---

## Tile Coordinate System

Tiles are addressed by four coordinates: `(z_t, z_f, x, y)`.

- **z_t** — time zoom level. Higher = finer time resolution. At `z_t_max`, one pixel = one STFT frame. Each step down doubles the time span per pixel.
- **z_f** — frequency zoom level. Higher = finer frequency resolution. At `z_f_max`, one pixel = one frequency bin. Each step down doubles the frequency span per pixel.
- **x** — tile column along the time axis (x=0 = archive start)
- **y** — tile row along the frequency axis (y=0 = 0 Hz, increases toward Nyquist)

The two zoom axes are **fully independent**: a researcher can zoom deep into a narrow time window while keeping the full frequency range visible, or zoom into a specific frequency band across the full archive duration.

Y=0 is at the bottom — inverse of geographic map convention. The MapLibre frontend compensates by inverting its Y axis.

All coordinate math lives in `TileCoordinates` (`ravenna/coordinates.py`). Do not put coordinate arithmetic anywhere else.

`TileCoordinates` has no external dependencies — pure arithmetic only.

### Frontend zoom mapping

MapLibre's native zoom control drives `z_t` (time zoom). A separate frequency zoom slider in the UI drives `z_f`. When `z_f` changes, the tile source URL is updated:

```
tiles/zt{z_t}/zf{current_z_f}/{x}/{y}.png
```

MapLibre reloads tiles automatically on source URL change. Time panning and zooming work natively; frequency band selection is an application-level layer swap.

---

## Intermediate Storage

- Full-resolution STFT → Zarr array, shape `[n_time_frames, n_freq_bins]`, chunked `[4096, n_freq_bins]`
- Pyramid → Zarr group, one array per `(z_t, z_f)` pair, keyed `zt{z_t}_zf{z_f}`
  - Shape at `(z_t, z_f)`: `[ceil(n_time_frames / 2^(z_t_max−z_t)), ceil(n_freq_bins / 2^(z_f_max−z_f))]`
  - Total arrays: `(z_t_max + 1) × (z_f_max + 1)` (e.g. 13 × 7 = 91 for defaults)
- Chunks align with tile boundaries — a tile read is always exactly one Zarr chunk read
- All stages are **idempotent**: existing Zarr chunks and PNG tiles are skipped on re-run
- Zarr intermediate is temporary; delete after tile archive is validated

---

## Normalization and Display

- `NormStrategy` maps raw dB values → `[0, 1]` before colormap application
- Two implementations:
  - `GlobalPercentileNorm` — fits low/high dB percentiles from a random sample of ~10K tiles drawn across the full pyramid before rendering begins
  - `CalibratedSPLNorm` — maps absolute Sound Pressure Level (SPL, dB re 1 µPa) to display range using hydrophone sensitivity from calibration file
- Norm parameters are serialized into a `metadata.json` file alongside the tile tree at packaging time
- The frontend `ColormapLegend` reads norm params from `metadata.json` and renders a dB-annotated color scale bar — this is the sole mechanism for reading pixel values in Phase 1
- `NormStrategy` instances must be serializable (to_dict/from_dict) for safe passing to worker processes

---

## Case Study: MBARI MARS Archive

The reference dataset is the MBARI MARS 256 kHz hydrophone archive, publicly available on AWS S3 (`pacific-sound-256khz-{year}` buckets, us-west-2, no credentials required). Run pipeline compute in us-west-2 to avoid S3 egress fees.

- 10-minute WAV files, 24-bit PCM, mono
- ~460 GB/week; ~151 TB total
- ~95% temporal coverage (gaps present)

---

## Phase Roadmap

| Phase | Scope |
|---|---|
| **Phase 1 (current)** | 2D overview pyramid (Z_t 0–12, Z_f 0–6); ColormapLegend dB scale; MBARI MARS case study |
| Phase 2 | Detail pyramid (finer Z_t and Z_f levels); audio clip delivery linked to viewport |
| Phase 3 | Signal detection and annotation overlay |

---

## Testing Conventions

- Unit tests use synthetic audio (pure tones at known frequencies) and known dB values — no real dataset required
- `TileCoordinates` must pass round-trip property tests: `tile→time→tile` and `tile→freq→tile` must be exact identity at all zoom levels, for all combinations of `z_t` and `z_f`
- Integration tests inject synthetic gaps to verify gap handling
- All pipeline stages must pass idempotency test: two runs on identical input produce bit-identical output
- Visual validation at each milestone: inspect known-signal time windows at multiple zoom levels
