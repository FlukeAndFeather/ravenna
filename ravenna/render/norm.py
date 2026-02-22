"""
Normalization strategies for converting dB pyramid values to [0, 1] for colormapping.

Two strategies:

  GlobalPercentileNorm — data-driven, computed from the pyramid itself.
      vmin/vmax are the 1st and 99.5th percentile of the coarsest pyramid level.
      Good for exploratory work where absolute SPL is unknown.

  CalibratedSPLNorm — physics-based, requires hydrophone calibration constants.
      Converts dBFS → dB re 1 µPa using the hydrophone sensitivity, then maps
      a user-specified SPL display range to [0, 1].
      Good for scientifically comparable, repeatable visualizations.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import zarr


class NormStrategy(ABC):
    """Map dB pyramid values to the [0, 1] float range."""

    @abstractmethod
    def normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Map *data* (float32 dB array) to float32 in [0, 1].

        Values below vmin clip to 0; values above vmax clip to 1.
        Fill-value pixels (-200 dB) naturally clip to 0.
        """

    @property
    @abstractmethod
    def vmin(self) -> float:
        """dB value that maps to 0 (darkest colour)."""

    @property
    @abstractmethod
    def vmax(self) -> float:
        """dB value that maps to 1 (brightest colour)."""


class GlobalPercentileNorm(NormStrategy):
    """
    Stretch the dynamic range so that the *norm_low_pct* percentile of the
    coarsest pyramid level maps to 0 and *norm_high_pct* maps to 1.

    Parameters
    ----------
    vmin_db, vmax_db : float
        Pre-computed percentile values (dB).  Use ``from_pyramid`` to derive
        them automatically.
    """

    def __init__(self, vmin_db: float, vmax_db: float) -> None:
        self._vmin = float(vmin_db)
        self._vmax = float(vmax_db)

    @property
    def vmin(self) -> float:
        return self._vmin

    @property
    def vmax(self) -> float:
        return self._vmax

    def normalize(self, data: np.ndarray) -> np.ndarray:
        span = self._vmax - self._vmin
        if span == 0:
            return np.zeros_like(data, dtype=np.float32)
        out = (data.astype(np.float32) - self._vmin) / span
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    @classmethod
    def from_pyramid(
        cls,
        group: zarr.Group,
        zoom_f_min: int,
        zoom_t_min: int,
        norm_low_pct: float = 1.0,
        norm_high_pct: float = 99.5,
    ) -> GlobalPercentileNorm:
        """
        Derive vmin/vmax from the coarsest pyramid level.

        The noise floor (-200 dB fill value) is excluded before computing
        percentiles so it does not skew the colour scale.
        """
        coarse = group[f"zt{zoom_t_min}_zf{zoom_f_min}"][:]
        valid = coarse[coarse > -190.0]
        if valid.size == 0:
            return cls(-100.0, 0.0)
        vmin = float(np.percentile(valid, norm_low_pct))
        vmax = float(np.percentile(valid, norm_high_pct))
        return cls(vmin, vmax)


class CalibratedSPLNorm(NormStrategy):
    """
    Convert dBFS values to calibrated Sound Pressure Level (dB re 1 µPa)
    using the hydrophone sensitivity, then map a fixed SPL display range
    to [0, 1].

    Conversion
    ----------
    The STFT magnitudes are stored as::

        dBFS = 20·log10(|X[k]| / full_scale)

    Adding the absolute value of the (negative) hydrophone sensitivity
    gives SPL in dB re 1 µPa::

        SPL = dBFS - sensitivity_db          # sensitivity_db is negative

    Parameters
    ----------
    sensitivity_db : float
        Hydrophone end-to-end sensitivity in dB re 1 V/µPa (typically negative,
        e.g. -168.0 for MARS).  From the calibration certificate.
    spl_min_db, spl_max_db : float
        SPL range (dB re 1 µPa) mapped to 0 and 1 respectively.
    """

    def __init__(
        self,
        sensitivity_db: float,
        spl_min_db: float,
        spl_max_db: float,
    ) -> None:
        if spl_max_db <= spl_min_db:
            raise ValueError(
                f"spl_max_db ({spl_max_db}) must be greater than spl_min_db ({spl_min_db})"
            )
        self._sensitivity_db = float(sensitivity_db)
        self._spl_min = float(spl_min_db)
        self._spl_max = float(spl_max_db)

    @property
    def vmin(self) -> float:
        return self._spl_min

    @property
    def vmax(self) -> float:
        return self._spl_max

    def normalize(self, data: np.ndarray) -> np.ndarray:
        # dBFS → SPL (dB re 1 µPa)
        spl = data.astype(np.float32) - self._sensitivity_db
        # Map [spl_min, spl_max] → [0, 1]
        span = self._spl_max - self._spl_min
        out = (spl - self._spl_min) / span
        return np.clip(out, 0.0, 1.0).astype(np.float32)


def make_norm_strategy(
    norm_strategy: str,
    group: zarr.Group,
    zoom_f_min: int,
    zoom_t_min: int,
    norm_low_pct: float = 1.0,
    norm_high_pct: float = 99.5,
    hydrophone_sensitivity_db: float | None = None,
    spl_display_min_db: float | None = None,
    spl_display_max_db: float | None = None,
) -> NormStrategy:
    """
    Factory that creates the appropriate ``NormStrategy`` from config values.

    Raises
    ------
    ValueError
        If ``norm_strategy`` is ``"calibrated_spl"`` but any of the three
        required calibration parameters is missing.
    ValueError
        If ``norm_strategy`` is not a recognised name.
    """
    if norm_strategy == "global_percentile":
        return GlobalPercentileNorm.from_pyramid(
            group, zoom_f_min, zoom_t_min, norm_low_pct, norm_high_pct
        )

    if norm_strategy == "calibrated_spl":
        missing = [
            name
            for name, val in [
                ("hydrophone_sensitivity_db", hydrophone_sensitivity_db),
                ("spl_display_min_db", spl_display_min_db),
                ("spl_display_max_db", spl_display_max_db),
            ]
            if val is None
        ]
        if missing:
            raise ValueError(
                f"calibrated_spl norm requires: {', '.join(missing)}"
            )
        return CalibratedSPLNorm(
            sensitivity_db=hydrophone_sensitivity_db,
            spl_min_db=spl_display_min_db,
            spl_max_db=spl_display_max_db,
        )

    raise ValueError(
        f"Unknown norm_strategy {norm_strategy!r}. "
        "Expected 'global_percentile' or 'calibrated_spl'."
    )
