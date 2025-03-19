from representations.TileCoder import SparseTileCoder, TileCoderConfig
import numpy as np

def test_tilecoder():
    tiles = 4
    tilings = 8
    dims = 2

    rep = SparseTileCoder(TileCoderConfig(
            tiles=tiles,
            tilings=tilings,
            dims=dims,
            input_ranges=None,   
        ))
    
    s1 = np.ones(dims)*0.5
    rep1 = rep.encode(s1)
    assert len(rep1) == (tiles**dims)*tilings

    s2 = np.ones(dims)    # at the upperbound
    rep2 = rep.encode(s2)

    s3 = np.ones(dims)*2   # out of upperbound
    rep3 = rep.encode(s3)
    
    assert np.all(rep2 == rep3)

    s4 = np.ones(dims)*0   # at the lowerbound
    rep4 = rep.encode(s4)
    assert np.any(rep4 != rep2)

    s5 = np.ones(dims)*-1   # below the lower bound
    rep5 = rep.encode(s5)
    assert np.all(rep4 == rep5)




