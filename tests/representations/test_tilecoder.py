from representations.TileCoder import DenseTileCoder, TileCoderConfig
from representations.RichTileCoder import RichTileCoder, RichTileCoderConfig
import numpy as np

class RichTileCoder():
    def setup_method(self):
        self.num_tiles = (2, 16)
        self.num_tilings = 8
        self.dims = 2

        self.config = RichTileCoderConfig(
            tiles=self.num_tiles,
            tilings=self.num_tilings,
            dims=self.dims,
            wrap_time=True,
            input_ranges=None
        )
        self.tile_coder = RichTileCoder(self.config)

    def test_initialization(self):
        assert self.tile_coder._c.dims == self.dims
        assert self.tile_coder._c.tilings == self.num_tilings
        assert self.tile_coder._c.tiles == self.num_tiles
        assert np.array_equal(self.tile_coder._input_ranges, np.array([(0.0, 1.0), (0.0, 1.0)]))

    def test_scale_factors(self):
        # Scale factor for x should be 2 / (1.0 - 0.0) = 2
        # Scale factor for y should be 16 / (1.0 - 0.0) = 16
        assert self.tile_coder.scale[0] == self.num_tiles[0]
        assert self.tile_coder.scale[1] == self.num_tiles[1]

    def test_maxSize(self):
        # maxSize should be tilings * (tiles_x + 1) * (tiles_y + 1)
        expected_max_size = self.num_tilings * (self.num_tiles[0] + 1) * (self.num_tiles[1] + 1)
        assert self.tile_coder.maxSize == expected_max_size

    def test_x_values(self):
        point1 = np.array([0, 0.5])
        point2 = np.array([1, 0.5])

        indices1 = self.tile_coder.get_indices(point1)
        indices2 = self.tile_coder.get_indices(point2)

        assert sorted(indices1) == sorted(indices2), \
            f"Close x-values should map to same tiles, but got {indices1} and {indices2}"

        encoding1 = self.tile_coder.encode(point1)
        encoding2 = self.tile_coder.encode(point2)

        assert np.array_equal(encoding1, encoding2), \
            "Encodings for close x-values should be identical"

    def test_y_values(self):
        y_edge = 1 / self.num_tilings / self.num_tiles[1]
        point1 = np.array([0.5, y_edge])
        point2 = np.array([0.5, y_edge + y_edge/2])

        indices1 = self.tile_coder.get_indices(point1)
        indices2 = self.tile_coder.get_indices(point2)

        assert sorted(indices1) == sorted(indices2), \
            f"Close y-values should map to same tiles, but got {indices1} and {indices2}"

        encoding1 = self.tile_coder.encode(point1)
        encoding2 = self.tile_coder.encode(point2)

        assert np.array_equal(encoding1, encoding2), \
            "Encodings for close y-values should be identical"

class AndyTileCoder():
    def setup_method(self):
        self.num_tiles = (2, 16)
        self.num_tilings = 8
        self.dims = 2

        self.config = TileCoderConfig(
            tiles=self.num_tiles,
            tilings=self.num_tilings,
            dims=self.dims,
            input_ranges=None
        )
        self.tile_coder = DenseTileCoder(self.config)

    def test_initialization(self):
        assert self.tile_coder._c.dims == self.dims
        assert self.tile_coder._c.tilings == self.num_tilings
        assert self.tile_coder._c.tiles == self.num_tiles
        assert np.array_equal(self.tile_coder._input_ranges, np.array([(0.0, 1.0), (0.0, 1.0)]))

    def test_x_values(self):
        x_edge = 1 / self.num_tilings / self.num_tiles[0]
        point1 = np.array([x_edge, 0.5])
        point2 = np.array([x_edge + x_edge*0.9, 0.5])

        indices1 = self.tile_coder.get_indices(point1)
        indices2 = self.tile_coder.get_indices(point2)

        assert sorted(indices1) == sorted(indices2), \
            f"Close x-values should map to same tiles, but got {indices1} and {indices2}"

        encoding1 = self.tile_coder.encode(point1)
        encoding2 = self.tile_coder.encode(point2)

        assert np.array_equal(encoding1, encoding2), \
            "Encodings for close x-values should be identical"

    def test_y_values(self):
        y_edge = 1 / self.num_tilings / self.num_tiles[1]
        point1 = np.array([0.5, y_edge])
        point2 = np.array([0.5, y_edge + y_edge/2])

        indices1 = self.tile_coder.get_indices(point1)
        indices2 = self.tile_coder.get_indices(point2)

        assert sorted(indices1) == sorted(indices2), \
            f"Close y-values should map to same tiles, but got {indices1} and {indices2}"

        encoding1 = self.tile_coder.encode(point1)
        encoding2 = self.tile_coder.encode(point2)

        assert np.array_equal(encoding1, encoding2), \
            "Encodings for close y-values should be identical"
