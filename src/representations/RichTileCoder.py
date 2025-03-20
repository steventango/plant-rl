from representations.tile3 import tiles, IHT
from typing import List, Optional, Sequence, Tuple
from dataclasses import dataclass
import numpy as np

Range = Tuple[float, float]

@dataclass
class RichTileCoderConfig:
    tiles: int | Sequence[int]
    tilings: int
    dims: int
    input_ranges: Optional[Sequence[Range | None]] = None
    
class RichTileCoder():
    def __init__(self, config: RichTileCoderConfig):
        self._c = c = config

        ranges: Sequence[Range | None] = [None] * c.dims
        if c.input_ranges is not None:
            assert len(c.input_ranges) == c.dims
            ranges = c.input_ranges

        self._input_ranges = _normalize_scalars(ranges)

        if isinstance(c.tiles, int):
            c.tiles = [c.tiles for _ in range(c.dims)]

        self.scale = [self.scaleFactor(c.tiles[i], self._input_ranges[i]) for i in range(c.dims)]

        self.maxSize = self.compute_maxSize(c.tiles)
        
        self.iht = IHT(self.maxSize)

    def get_indices(self, s: np.ndarray):   
        return tiles(self.iht, self._c.tilings, [s[i]*self.scale[i] for i in range(self._c.dims)])

    def features(self):
        return self.maxSize
    
    def scaleFactor(self, num_tiles: int, range: Tuple[float, float]):
        return num_tiles / abs(range[1] - range[0])

    def compute_maxSize(self, x):
        a = self._c.tilings
        for num_tiles in x: 
            a *= num_tiles + 1
        return a
    
    def encode(self, s: np.ndarray):
        indices = self.get_indices(s)
        vec = np.zeros(self.maxSize)
        vec[indices] = 1.
        return vec

def _normalize_scalars(sc: Sequence[Tuple[float, float] | None]):
    out: List[Tuple[float, float]] = []
    for r in sc:
        if r is None:
            out.append((0., 1.))

        else:
            out.append(r)

    return np.array(out, dtype=np.float64)