import asyncio
import logging

import numpy as np
import polars as pl
import pytest

from environments.PlantGrowthChamber.MockWallStatsActionTraceEmbeddingPlantGrowthChamberColorTriangle import (
    MockWallStatsActionTraceEmbeddingPlantGrowthChamberColorTriangle,
)

logging.basicConfig(level=logging.INFO)


def assert_observation(obs, row):
    actual_wall_time = obs[0]
    desired_wall_time = row["wall_time"].item().mean()
    np.testing.assert_allclose(actual_wall_time, desired_wall_time, rtol=1e-5)

    actual_mean_clean_area = obs[1]
    desired_mean_clean_area = row["clean_area"].item().mean()
    np.testing.assert_allclose(actual_mean_clean_area, desired_mean_clean_area, rtol=5e-2)

    actual_log_clean_area = obs[28]
    desired_log_clean_area = row["log_clean_area"].item().mean()
    np.testing.assert_allclose(actual_log_clean_area, desired_log_clean_area, rtol=1e-2)

    actual_solidity = obs[3]
    desired_solidity = row["clean_solidity"].item().mean()
    np.testing.assert_allclose(actual_solidity, desired_solidity, rtol=1e-1)

    actual_liters = obs[33]
    desired_liters = row["liters_per_pot"].item().mean()
    np.testing.assert_allclose(actual_liters, desired_liters, rtol=1e-5)

    actual_pca = obs[38:48][:4]
    desired_pca = np.stack(row["cls_token_pca"].item()).mean(axis=0)[:4]
    np.testing.assert_allclose(actual_pca, desired_pca, rtol=5e-1)


@pytest.mark.asyncio
async def test_mock_triangle_chamber_cv():
    env = MockWallStatsActionTraceEmbeddingPlantGrowthChamberColorTriangle(
        mock_stats=False,
        experiment=15,
        zone_id=2,
        sterilization_date="2025-12-01",
        plate_date="2025-12-04",
        transplant_date="2025-12-11",
        dome_removal_date="2025-12-15",
        watering_date="2025-12-15",
        start_date="2025-12-17T16:30:00+00:00",
        liters_per_pot=0.0625,
        pca_model_path="/data/plant-rl/offline/v23/pca_model.joblib",
        timezone="America/Edmonton",
    )

    print("Starting environment...")
    obs, info = await env.start()
    print(f"Observation shape: {obs.shape}")
    rows = (
        env.dataset_df.filter((pl.col("experiment") == 15) & (pl.col("zone") == 2))
        .group_by("time", maintain_order=True)
        .agg(
            "wall_time",
            "clean_area",
            "log_clean_area",
            "clean_solidity",
            "liters_per_pot",
            "cls_token_pca",
        )
        .head(4)
    )

    assert len(obs) == 816
    assert_observation(obs, rows[0])

    # Test step
    action = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    env.update_action_trace(action)
    reward, observation, terminal, info = await env.step(action)

    assert reward == 0.0
    assert not terminal

    assert_observation(observation, rows[1])

    # Test step
    action = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    env.update_action_trace(action)
    reward, observation, terminal, info = await env.step(action)

    assert reward != 0.0
    assert not terminal

    assert_observation(observation, rows[3])
    await env.close()


if __name__ == "__main__":
    asyncio.run(test_mock_triangle_chamber_cv())
