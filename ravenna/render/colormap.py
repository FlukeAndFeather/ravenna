"""Apply a matplotlib colormap to a normalised [0, 1] float array."""
from __future__ import annotations

import matplotlib
import numpy as np


def apply_colormap(normalized: np.ndarray, colormap_name: str = "viridis") -> np.ndarray:
    """
    Apply a matplotlib colormap to *normalized* and return an RGBA uint8 array.

    Parameters
    ----------
    normalized : np.ndarray
        Float array with values in [0, 1]. Any shape.
    colormap_name : str
        Any name accepted by ``matplotlib.colormaps``.

    Returns
    -------
    np.ndarray
        RGBA uint8 array; same shape as *normalized* with a trailing ``(4,)``
        dimension appended.

    Raises
    ------
    KeyError
        If *colormap_name* is not a registered matplotlib colormap.
    """
    cmap = matplotlib.colormaps[colormap_name]
    rgba = cmap(normalized)          # float64 in [0, 1], shape (..., 4)
    return (rgba * 255).astype(np.uint8)
