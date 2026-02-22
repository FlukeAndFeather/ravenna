from datetime import datetime, timezone

import pytest

from ravenna.config import PipelineConfig


@pytest.fixture
def minimal_config():
    return PipelineConfig(
        source_uri="/data/wav",
        date_start=datetime(2020, 1, 1, tzinfo=timezone.utc),
        date_end=datetime(2020, 1, 2, tzinfo=timezone.utc),
        sample_rate=256_000,
    )


def test_json_roundtrip(minimal_config):
    restored = PipelineConfig.from_json(minimal_config.to_json())
    assert restored == minimal_config


def test_json_roundtrip_preserves_datetimes(minimal_config):
    restored = PipelineConfig.from_json(minimal_config.to_json())
    assert restored.date_start == minimal_config.date_start
    assert restored.date_end == minimal_config.date_end


def test_json_roundtrip_with_optional_fields():
    config = PipelineConfig(
        source_uri="s3://bucket/prefix",
        date_start=datetime(2015, 1, 1, tzinfo=timezone.utc),
        date_end=datetime(2025, 1, 1, tzinfo=timezone.utc),
        sample_rate=256_000,
        norm_strategy="calibrated_spl",
        hydrophone_sensitivity_db=-168.0,
        spl_display_min_db=60.0,
        spl_display_max_db=130.0,
        source_credentials={"profile": "my-aws-profile"},
    )
    assert PipelineConfig.from_json(config.to_json()) == config


def test_default_values(minimal_config):
    assert minimal_config.fft_size == 512
    assert minimal_config.hop_size == 128
    assert minimal_config.window == "hann"
    assert minimal_config.tile_size == 256
    assert minimal_config.zoom_t_min == 0
    assert minimal_config.zoom_t_max == 12
    assert minimal_config.zoom_f_min == 0
    assert minimal_config.zoom_f_max == 6
    assert minimal_config.downsample_method == "mean"
    assert minimal_config.norm_strategy == "global_percentile"
    assert minimal_config.norm_low_pct == 1.0
    assert minimal_config.norm_high_pct == 99.5
    assert minimal_config.colormap == "viridis"
    assert minimal_config.n_workers == 16
    assert minimal_config.chunk_size_frames == 4096
    assert minimal_config.hydrophone_sensitivity_db is None
    assert minimal_config.source_credentials == {}


def test_missing_required_fields():
    with pytest.raises(TypeError):
        PipelineConfig()   # source_uri, date_start, date_end, sample_rate all required
