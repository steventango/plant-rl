from __future__ import annotations

from typing import Any

import numpy as np

from environments.PlantGrowthChamber.MockPlantGrowthChamber import MockPlantGrowthChamber
from environments.PlantGrowthChamber.PlantGrowthChamber import PlantGrowthChamber
from environments.PlantGrowthChamber.specs import (
    ActionSpec,
    ObservationSpec,
    ACTION_SPECS,
    create_observation_spec,
)
from utils.RlGlue.environment import BaseAsyncEnvironment

BACKENDS: dict[str, type] = {
    "live": PlantGrowthChamber,
    "mock": MockPlantGrowthChamber,
}


class ComposedPlantGrowthChamber(BaseAsyncEnvironment):
    def __init__(
        self,
        backend: PlantGrowthChamber,
        observation_spec: ObservationSpec,
        action_spec: ActionSpec,
        env_params: dict[str, Any],
    ):
        self._backend = backend
        self._observation_spec = observation_spec
        self._action_spec = action_spec
        observation_spec.setup(backend, env_params)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)

    async def get_observation(self) -> np.ndarray:
        raw = await self._backend.get_raw_observation()
        return await self._observation_spec.encode(raw)

    async def start(self):
        raw_obs, info = await self._backend.start()
        encoded = await self._observation_spec.encode(raw_obs)
        return encoded, info

    async def step(self, action: Any):
        decoded = self._action_spec.decode(action)
        reward, raw_obs, terminal, info = await self._backend.step(decoded)
        encoded = await self._observation_spec.encode(raw_obs)
        return reward, encoded, terminal, info

    def update_action_trace(self, action: Any) -> None:
        trace_action = self._action_spec.trace_action(action)
        self._observation_spec.update_action_trace(trace_action)

    async def close(self):
        await self._backend.close()


def create_plant_growth_chamber(**env_params: Any) -> ComposedPlantGrowthChamber:
    params = dict(env_params)
    backend_name = params.pop("backend", "live")
    observation_name = params.pop("observation", "scalar")
    action_name = params.pop("action", "ppfd6")

    if backend_name not in BACKENDS:
        raise ValueError(f"Unknown backend: {backend_name}")
    if action_name not in ACTION_SPECS:
        raise ValueError(f"Unknown action spec: {action_name}")

    action_spec = ACTION_SPECS[action_name]
    observation_spec = create_observation_spec(observation_name, action_spec, params)

    backend_cls = BACKENDS[backend_name]
    backend = backend_cls(**params)

    return ComposedPlantGrowthChamber(backend, observation_spec, action_spec, params)
