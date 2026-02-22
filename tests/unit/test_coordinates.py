import pytest

from ravenna.coordinates import TileCoordinates, TileExtent

# Representative STFT parameters (MBARI MARS defaults from design spec)
SAMPLE_RATE = 256_000
HOP_SIZE = 128
FFT_SIZE = 512
TILE_SIZE = 256
ZOOM_T_MAX = 12
ZOOM_F_MAX = 6
# Enough frames/bins to cover a few tiles at the finest zoom level
N_TIME_FRAMES = TILE_SIZE * 10   # 10 tiles wide at z_t_max


@pytest.fixture
def tc():
    return TileCoordinates(
        sample_rate=SAMPLE_RATE,
        hop_size=HOP_SIZE,
        fft_size=FFT_SIZE,
        n_time_frames=N_TIME_FRAMES,
        tile_size=TILE_SIZE,
        zoom_t_max=ZOOM_T_MAX,
        zoom_f_max=ZOOM_F_MAX,
    )


ALL_T_ZOOM_LEVELS = list(range(ZOOM_T_MAX + 1))   # z_t: 0 … 12
ALL_F_ZOOM_LEVELS = list(range(ZOOM_F_MAX + 1))   # z_f: 0 … 6


# ── Round-trip: tile → time → tile ───────────────────────────────────────

@pytest.mark.parametrize("z_t", ALL_T_ZOOM_LEVELS)
@pytest.mark.parametrize("x", [0, 1, 2])
def test_time_roundtrip(tc, z_t, x):
    start_sec, _ = tc.tile_to_time_range(z_t, x)
    assert tc.time_to_tile(z_t, start_sec) == x


# ── Round-trip: tile → freq → tile ───────────────────────────────────────

@pytest.mark.parametrize("z_f", ALL_F_ZOOM_LEVELS)
@pytest.mark.parametrize("y", [0, 1])
def test_freq_roundtrip(tc, z_f, y):
    low_hz, _ = tc.tile_to_freq_range(z_f, y)
    assert tc.freq_to_tile(z_f, low_hz) == y


# ── Axes are independent ──────────────────────────────────────────────────

def test_time_range_independent_of_z_f(tc):
    """tile_to_time_range must not depend on z_f."""
    for z_t in ALL_T_ZOOM_LEVELS:
        r0 = tc.tile_to_time_range(z_t, 0)
        # same z_t, different z_f — result must be identical
        assert r0 == tc.tile_to_time_range(z_t, 0)


def test_freq_range_independent_of_z_t(tc):
    """tile_to_freq_range must not depend on z_t."""
    for z_f in ALL_F_ZOOM_LEVELS:
        r0 = tc.tile_to_freq_range(z_f, 0)
        assert r0 == tc.tile_to_freq_range(z_f, 0)


# ── Axis orientation ──────────────────────────────────────────────────────

def test_origin_is_zero_time(tc):
    """Tile column x=0 must start at t=0 s at all z_t levels."""
    for z_t in ALL_T_ZOOM_LEVELS:
        start_sec, _ = tc.tile_to_time_range(z_t, 0)
        assert start_sec == pytest.approx(0.0)


def test_origin_is_zero_freq(tc):
    """Tile row y=0 must start at 0 Hz at all z_f levels."""
    for z_f in ALL_F_ZOOM_LEVELS:
        low_hz, _ = tc.tile_to_freq_range(z_f, 0)
        assert low_hz == pytest.approx(0.0)


def test_time_increases_with_x(tc):
    s0, _ = tc.tile_to_time_range(ZOOM_T_MAX, 0)
    s1, _ = tc.tile_to_time_range(ZOOM_T_MAX, 1)
    assert s1 > s0


def test_freq_increases_with_y(tc):
    _, high0 = tc.tile_to_freq_range(ZOOM_F_MAX, 0)
    low1, _ = tc.tile_to_freq_range(ZOOM_F_MAX, 1)
    assert low1 == pytest.approx(high0)


# ── Zoom level coarsening — time ──────────────────────────────────────────

def test_coarser_z_t_covers_more_time(tc):
    """Each step down in z_t doubles the time covered by a tile."""
    for z_t in range(1, ZOOM_T_MAX + 1):
        _, end_fine = tc.tile_to_time_range(z_t, 0)
        _, end_coarse = tc.tile_to_time_range(z_t - 1, 0)
        assert end_coarse == pytest.approx(end_fine * 2)


# ── Zoom level coarsening — frequency ────────────────────────────────────

def test_coarser_z_f_covers_more_freq(tc):
    """Each step down in z_f doubles the frequency range covered by a tile."""
    for z_f in range(1, ZOOM_F_MAX + 1):
        _, high_fine = tc.tile_to_freq_range(z_f, 0)
        _, high_coarse = tc.tile_to_freq_range(z_f - 1, 0)
        assert high_coarse == pytest.approx(high_fine * 2)


# ── tile_extent ───────────────────────────────────────────────────────────

def test_tile_extent_at_max_zoom(tc):
    ext = tc.tile_extent(ZOOM_T_MAX, ZOOM_F_MAX)
    assert ext.z_t == ZOOM_T_MAX
    assert ext.z_f == ZOOM_F_MAX
    assert ext.n_x == 10   # N_TIME_FRAMES // TILE_SIZE = 10 tiles
    assert ext.n_y >= 1


def test_tile_extent_n_x_shrinks_with_z_t(tc):
    ext_fine = tc.tile_extent(ZOOM_T_MAX, ZOOM_F_MAX)
    ext_coarse = tc.tile_extent(ZOOM_T_MAX - 1, ZOOM_F_MAX)
    assert ext_coarse.n_x <= ext_fine.n_x


def test_tile_extent_n_y_shrinks_with_z_f(tc):
    ext_fine = tc.tile_extent(ZOOM_T_MAX, ZOOM_F_MAX)
    ext_coarse = tc.tile_extent(ZOOM_T_MAX, ZOOM_F_MAX - 1)
    assert ext_coarse.n_y <= ext_fine.n_y


def test_tile_extent_axes_independent(tc):
    """n_x must not change when only z_f changes, and vice versa."""
    ext_a = tc.tile_extent(ZOOM_T_MAX, ZOOM_F_MAX)
    ext_b = tc.tile_extent(ZOOM_T_MAX, 0)
    assert ext_a.n_x == ext_b.n_x   # z_t unchanged → n_x unchanged

    ext_c = tc.tile_extent(ZOOM_T_MAX, ZOOM_F_MAX)
    ext_d = tc.tile_extent(0, ZOOM_F_MAX)
    assert ext_c.n_y == ext_d.n_y   # z_f unchanged → n_y unchanged


def test_tile_extent_returns_tile_extent(tc):
    ext = tc.tile_extent(0, 0)
    assert isinstance(ext, TileExtent)
    assert ext.z_t == 0
    assert ext.z_f == 0


# ── No external imports ───────────────────────────────────────────────────

def test_no_external_imports():
    import importlib, sys
    importlib.import_module("ravenna.coordinates")
    assert "numpy" not in sys.modules or "ravenna.coordinates" not in str(
        getattr(sys.modules.get("numpy"), "__file__", "")
    )
