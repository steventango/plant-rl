import numpy as np

from environments.PlantGrowthChamber.PlantGrowthChamber import PlantGrowthChamber
from utils.metrics import UnbiasedExponentialMovingAverage


class CVPlantGrowthChamber(PlantGrowthChamber):
    def __init__(self, zone: int, total_steps: int = 40320):
        super().__init__(zone)
        self.total_steps = total_steps
        # todo don't hardcode 16
        self.plant_area_emas = [UnbiasedExponentialMovingAverage(alpha=0.1)] * 16

    def get_observation(self):
        time, _, plant_stats = super().get_observation()

        transformed_time = self.transform_time(time)

        # todo don't hardcode 5 minutes
        countdown = self.transform_time(time, self.total_steps * 5 * 60)

        plant_area_emas_prev = [plant_area_ema.compute() for plant_area_ema in self.plant_area_emas]
        mean_plant_area_ema_prev = np.mean(plant_area_emas_prev)

        plant_areas = plant_stats[:, 0]
        for plant_area_ema in self.plant_area_emas:
            plant_area_ema.update(values=plant_areas)

        plant_area_emas = [plant_area_ema.compute() for plant_area_ema in self.plant_area_emas]
        mean_plant_area_ema = np.mean(plant_area_emas)

        observation = np.array(
            [
                *transformed_time,
                *countdown,
                mean_plant_area_ema_prev,
                mean_plant_area_ema,
            ]
        )
        return observation

    def transform_time(self, time, total=86400):
        return np.array([np.sin(2 * np.pi * time / total), np.cos(2 * np.pi * time / total)])
