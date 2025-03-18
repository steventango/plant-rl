import numpy as np

from environments.PlantGrowthChamber.PlantGrowthChamber import PlantGrowthChamber


class CVPlantGrowthChamber(PlantGrowthChamber):
    def __init__(self, zone: int, total_steps: int = 40320):
        super().__init__(zone)
        self.total_steps = total_steps

    def get_observation(self):
        time, _, plant_stats = super().get_observation()

        transformed_time = self.transform_time(time)

        # todo don't hardcode 5 minutes
        countdown = self.transform_time(time, self.total_steps * 5 * 60)

        observation = np.array(
            [
                *transformed_time,
                *countdown,
                self.mean_plant_area_ema_prev,
                self.mean_plant_area_ema,
            ]
        )
        return observation

    def transform_time(self, time, total=86400):
        return np.array([np.sin(2 * np.pi * time / total), np.cos(2 * np.pi * time / total)])
