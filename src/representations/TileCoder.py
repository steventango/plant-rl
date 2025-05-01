from PyFixedReps.TileCoder import TileCoder, TileCoderConfig
import numpy as np

class DenseTileCoder(TileCoder):
    def __init__(self, params: TileCoderConfig, rng=None):
        params.scale_output = False
        super().__init__(params, rng=rng)
    
    def nonzero_features(self):
        return self._c.tilings
