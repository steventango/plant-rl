import numpy as np
from environments.PlantGrowthChamber.PlantGrowthChamberIntensity import PlantGrowthChamberIntensity
from datetime import timedelta

class CVPlantGrowthChamberIntensity(PlantGrowthChamberIntensity):
    def __init__(self, **kwargs):
        PlantGrowthChamberIntensity.__init__(self, **kwargs)

class CVPlantGrowthChamberIntensity_MotionTracking(CVPlantGrowthChamberIntensity):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.duration = timedelta(minutes=1)

    async def execute_night_transition(self):
        woke = False
        if self.enforce_night:
            if self.is_night():
                await self.lights_off_and_sleep_until_morning()
                await self.put_action(0.35 * np.array([0.398, 0.762, 0.324, 0.000, 0.332, 0.606]))
                woke = True
        return woke

    async def get_observation(self):
        epoch_time, _, df = await super().get_observation()
        local_time = epoch_time.astimezone(self.tz)

        if not df.empty:
            mean_clean_area = df["clean_area"].mean() if "clean_area" in df else 0.0
        else:
            mean_clean_area = 0.0

        return np.array([local_time, mean_clean_area])