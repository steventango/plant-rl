from PyFixedReps.TileCoder import TileCoder, TileCoderConfig
import numpy as np

class SparseTileCoder(TileCoder):
    def __init__(self, params: TileCoderConfig, rng=None):
        params.scale_output = False
        super().__init__(params, rng=rng)

    def _build_offset(self, n):   # use asymmetrical offsets; see Suttom & Barto p220
        tile_length = 1.0 / super()._tiles
        offset = []
        for d in range(super()._c.dims):
            offset.append(((1 + 2*d) * n * (tile_length / super()._c.tilings)) % tile_length - tile_length / 2)
        return np.array(offset)