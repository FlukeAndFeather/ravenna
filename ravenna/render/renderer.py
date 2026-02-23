"""
TileRenderer — Stage 4 of the Ravenna pipeline.

Converts pyramid tiles (float32 dB arrays) into PNG bytes by:
  1. Extracting the tile slice from the pyramid Zarr group.
  2. Normalising dB values to [0, 1] via a NormStrategy.
  3. Applying a matplotlib colormap to produce RGBA pixels.
  4. Encoding the result as PNG.

Tile orientation
----------------
The output image has high-frequency bins at the top (row 0) and time
advancing left → right.  Pixels that lie outside the data boundary
(fill value = -200 dB) are coloured with the darkest colormap colour
after clipping through the NormStrategy.
"""
from __future__ import annotations

import io

import numpy as np
import zarr
from PIL import Image

from ravenna.config import PipelineConfig
from ravenna.render.colormap import apply_colormap
from ravenna.render.norm import NormStrategy

_FILL_VALUE: float = -200.0


class TileRenderer:
    """
    Render individual pyramid tiles as PNG bytes.

    Parameters
    ----------
    config : PipelineConfig
    group : zarr.Group
        Pyramid group returned by ``PyramidBuilder.build_all()``.
    norm : NormStrategy
        Pre-configured normalisation strategy (e.g. from
        ``make_norm_strategy``).
    """

    def __init__(
        self,
        config: PipelineConfig,
        group: zarr.Group,
        norm: NormStrategy,
    ) -> None:
        self.config = config
        self.group = group
        self.norm = norm

    # ── Public API ────────────────────────────────────────────────────────

    def render_tile(self, z_t: int, z_f: int, x: int, y: int) -> bytes:
        """
        Render tile *(z_t, z_f, x, y)* and return PNG bytes.

        The output is always ``tile_size × tile_size`` pixels regardless of
        whether the tile lies at the data boundary.
        """
        data = self._extract_tile_data(z_t, z_f, x, y)
        normalized = self.norm.normalize(data)
        rgba = apply_colormap(normalized, self.config.colormap)
        return _encode_png(rgba)

    # ── Internals ─────────────────────────────────────────────────────────

    def _extract_tile_data(
        self, z_t: int, z_f: int, x: int, y: int
    ) -> np.ndarray:
        """
        Return a ``(tile_size, tile_size)`` float32 dB array for the tile,
        oriented with high-frequency bins at row 0 and time left → right.

        Pixels beyond the array boundary are filled with ``_FILL_VALUE``
        (-200 dB), which normalises to 0 and maps to the darkest colour.
        """
        tile = self.config.tile_size
        arr = self.group[f"zt{z_t}_zf{z_f}"]

        t0, f0 = x * tile, y * tile
        t1 = min(t0 + tile, arr.shape[0])
        f1 = min(f0 + tile, arr.shape[1])

        canvas = np.full((tile, tile), _FILL_VALUE, dtype=np.float32)
        canvas[: t1 - t0, : f1 - f0] = arr[t0:t1, f0:f1]

        # (time, freq) → (freq, time), then flip so high freq is at row 0
        return np.flipud(canvas.T)


def _encode_png(rgba: np.ndarray) -> bytes:
    """Encode a ``(H, W, 4)`` uint8 RGBA array as PNG bytes."""
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
