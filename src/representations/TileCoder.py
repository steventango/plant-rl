from PyFixedReps.TileCoder import TileCoder, TileCoderConfig
import numpy as np

class SparseTileCoder(TileCoder):
    def __init__(self, params: TileCoderConfig, rng=None):
        params.scale_output = False
        super().__init__(params, rng=rng)

    def _build_offset(self, n):   # use asymmetrical offsets; see Suttom & Barto p220
        tile_length = 1.0 / self._tiles
        offset = []
        for d in range(self._c.dims):
            offset.append(((1 + 2*d) * n * (tile_length[d] / self._c.tilings)) % tile_length[d] - tile_length[d] / 2)
        return np.array(offset)