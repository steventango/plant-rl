from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from representations.tile3 import IHT, tiles

Range = Tuple[float, float]


@dataclass
class RichTileCoderConfig:
    tiles: int | Sequence[int]
    tilings: int
    dims: int
    strategy: str | None
    input_ranges: Optional[Sequence[Range | None]] = None


class RichTileCoder:
    def __init__(self, config: RichTileCoderConfig):
        self._c = c = config

        ranges: Sequence[Range | None] = [None] * c.dims
        if c.input_ranges is not None:
            assert len(c.input_ranges) == c.dims
            ranges = c.input_ranges

        self._input_ranges = _normalize_scalars(ranges)

        if isinstance(c.tiles, int):
            self._c.tiles = c.tiles = [c.tiles for _ in range(c.dims)]

        self.scale = [
            self.scaleFactor(c.tiles[i], self._input_ranges[i]) for i in range(c.dims)
        ]

        self.maxSize = self.compute_maxSize(c.tiles)

        self.iht = IHT(self.maxSize)

    def get_indices(self, s: np.ndarray):
        # tc = tile(TOD) + tile(TOD, plant motion) + tile(plant area)
        if self._c.strategy == "tc1":
            tile1 = tiles(self.iht, self._c.tilings, [s[0] * self.scale[0]], [0])
            tile2 = tiles(
                self.iht,
                self._c.tilings,
                [s[0] * self.scale[0], s[2] * self.scale[2]],
                [1],
            )
            tile3 = tiles(self.iht, self._c.tilings, [s[1] * self.scale[1]], [2])
            return tile1 + tile2 + tile3

        # tc = tile(s0) + tile(s1)
        if self._c.strategy == "tc2":
            tile1 = tiles(self.iht, self._c.tilings, [s[0] * self.scale[0]], [0])
            tile2 = tiles(self.iht, self._c.tilings, [s[1] * self.scale[1]], [1])
            return tile1 + tile2

        if self._c.strategy == "onehot":
            num_bins = 12
            value = np.clip(s[0], 0, 1)
            bin_idx = min(int(value * num_bins), num_bins - 1)
            return bin_idx

        # general
        else:
            return tiles(
                self.iht,
                self._c.tilings,
                [s[i] * self.scale[i] for i in range(self._c.dims)],
            )

    def features(self):
        return self.maxSize

    def nonzero_features(self):
        if self._c.strategy == "tc1":
            return 3 * self._c.tilings
        elif self._c.strategy == "tc2":
            return 2 * self._c.tilings
        elif self._c.strategy == "onehot":
            return 1
        else:
            return self._c.tilings

    def scaleFactor(self, num_tiles: int, range: Tuple[float, float]):
        return num_tiles / abs(range[1] - range[0])

    def compute_maxSize(self, x):
        if self._c.strategy == "tc1":
            return self._c.tilings * ((x[0] + 1) + (x[0] + 1) * (x[2] + 1) + (x[1] + 1))
        if self._c.strategy == "tc2":
            return self._c.tilings * ((x[0] + 1) + (x[1] + 1))
        if self._c.strategy == "onehot":
            return 12
        else:
            a = self._c.tilings
            for num_tiles in x:
                a *= num_tiles + 1
            return a

    def encode(self, s: np.ndarray):
        indices = self.get_indices(s)
        vec = np.zeros(self.maxSize)
        vec[indices] = 1.0
        return vec


def _normalize_scalars(sc: Sequence[Tuple[float, float] | None]):
    out: List[Tuple[float, float]] = []
    for r in sc:
        if r is None:
            out.append((0.0, 1.0))

        else:
            out.append(r)

    return np.array(out, dtype=np.float64)
