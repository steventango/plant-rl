from datetime import datetime
from itertools import chain
from pathlib import Path

from PIL import Image

from environments.PlantGrowthChamber.PlantGrowthChamber import PlantGrowthChamber


class MockPlantGrowthChamber(PlantGrowthChamber):

    def __init__(self, zone: int, path: str):
        super().__init__(zone)
        self.paths = sorted(chain(Path(path).glob("*.png"), Path(path).glob("*.jpg")))

    def get_image(self):
        path = self.paths[self.step]
        _, side = path.stem.split("_")
        self.images[side] = Image.open(path)

    def get_time(self):
        path = self.paths[self.step]
        time, _ = path.stem.split("_")
        return datetime.fromisoformat(time).timestamp()

    def put_action(self, action: int):
        pass

    def close(self):
        pass
