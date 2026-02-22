# Ravenna

> Pan and zoom across years of bioacoustic recordings in a browser.

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/FlukeAndFeather/ravenna/actions/workflows/ci.yml/badge.svg)](https://github.com/FlukeAndFeather/ravenna/actions/workflows/ci.yml)

Ravenna is an offline processing pipeline that converts long-duration passive acoustic monitoring (PAM) audio archives into interactive, zoomable spectrograms — the [Cloud-Optimized GeoTIFF](https://cogeo.org) concept applied to the time-frequency domain.

A researcher can navigate from a ten-year archive overview down to sub-second resolution in a single browser tab, with no server-side tile rendering at query time.

---

## How it works

Five discrete stages, each idempotent and independently re-runnable:

```
WAV/FLAC files → STFT (Zarr) → Pyramid → Render (PNG tiles) → PMTiles archive
   Ingest          Stage 2      Stage 3       Stage 4              Stage 5
```

1. **Ingest** — enumerate audio files from local disk, S3, or HTTP; stream as `AudioChunk` objects; synthesize silence for data gaps
2. **STFT** — Short-Time Fourier Transform each chunk; write magnitude in dB to a Zarr array on disk
3. **Pyramid** — progressively 2× downsample the full-resolution Zarr from Z_max → Z0; one array per zoom level
4. **Render** — read each (z, x, y) tile from the pyramid; normalize; apply colormap; write 256×256 PNG; embarrassingly parallel via `ProcessPoolExecutor`
5. **Package** — assemble PNG tiles into a PMTiles archive (production) or MBTiles/SQLite (local dev); sort by Hilbert curve index

### Zoom levels

| Zoom | Time per pixel | What you see |
|------|---------------|--------------|
| Z0   | ~2.4 hr       | Full archive on a single tile |
| Z6   | ~2.5 min      | Hour-scale diel patterns |
| Z12  | ~0.6 sec      | Individual whale calls, vessel passes |

---

## Case study: MBARI MARS

The reference dataset is the [MBARI MARS](https://registry.opendata.aws/pacific-sound/) 256 kHz hydrophone archive — ~10 years of continuous deep-sea recordings, ~151 TB of raw audio, publicly available on AWS S3 (no credentials required). Phase 1 target output: ~1 GB PMTiles file covering the full archive at Z0–Z12.

---

## Installation

```bash
pip install -e ".[dev]"
```

**Requirements:** Python 3.11+. See `pyproject.toml` for the full dependency list. Cloud storage backends (S3, GCS) require the optional `cloud` extras:

```bash
pip install -e ".[dev,cloud]"
```

---

## Quick start

```bash
# Process a local WAV file end-to-end
ravenna run --source /path/to/audio/ --output spectrogram.pmtiles

# Check pipeline progress on a running or interrupted job
ravenna status

# Run only specific stages (e.g. after modifying colormap settings)
ravenna run --source /path/to/audio/ --output spectrogram.pmtiles --stages render,package
```

Serve and view locally:

```bash
# Serve the MBTiles output with any compatible tile server, then open:
open frontend/index.html
```

---

## Configuration

All parameters live in `PipelineConfig` (`ravenna/config.py`) and are serialized to JSON alongside every output artifact, enabling exact reproduction of any run.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fft_size` | 512 | Samples per FFT window (2 ms at 256 kHz) |
| `hop_size` | 128 | Samples between windows (4× overlap) |
| `window` | `hann` | Window function |
| `tile_size` | 256 | Pixels per tile (both axes) |
| `zoom_min` | 0 | Coarsest zoom level |
| `zoom_max` | 12 | Finest overview zoom level |
| `norm_strategy` | `global_percentile` | `global_percentile` or `calibrated_spl` |
| `colormap` | `viridis` | Any matplotlib colormap name |
| `output_format` | `pmtiles` | `pmtiles` (production) or `mbtiles` (local dev) |
| `n_workers` | 16 | Parallel worker processes for tile rendering |

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run tests with coverage
pytest --cov=ravenna
```

---

## Roadmap

| Phase | Scope | Status |
|-------|-------|--------|
| **Phase 1** | Overview pyramid Z0–Z12; ColormapLegend dB scale; MBARI MARS case study | In progress |
| Phase 2 | Detail pyramid Z13–Z23; audio clip delivery linked to viewport | Planned |
| Phase 3 | Signal detection and annotation overlay | Planned |

---

## License

MIT — see [LICENSE](LICENSE).
