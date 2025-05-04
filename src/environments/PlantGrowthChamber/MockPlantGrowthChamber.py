from datetime import datetime
from itertools import chain
from pathlib import Path

from PIL import Image

from environments.PlantGrowthChamber.PlantGrowthChamber import PlantGrowthChamber


class MockPlantGrowthChamber(PlantGrowthChamber):

    def __init__(self, zone: int, path: str):
        super().__init__(zone)
        self.paths = sorted(chain(Path(path).glob("*.png"), Path(path).glob("*.jpg")))

    async def get_image(self):
        path = self.paths[self.n_step]
        _, side = path.stem.split("_")
        self.images[side] = Image.open(path)

    def get_time(self):
        path = self.paths[self.n_step]
        time, _ = path.stem.split("_")
        return datetime.fromisoformat(time)

    async def put_action(self, action: int):
        self.last_action = action

    async def sleep_until(self, wake_time: datetime):
        while self.get_time() < wake_time:
            self.n_step += 1

    async def close(self):
        pass
