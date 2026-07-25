"""CV failure must reuse the last valid plant-stats df (#367)."""

from datetime import datetime
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from environments.PlantGrowthChamber.PlantGrowthChamber import PlantGrowthChamber

UTC = ZoneInfo("Etc/UTC")


def _valid_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pot_id": ["1", "2"],
            "area": [100.0, 110.0],
            "clean_area": [4.0, 5.0],
        }
    )


def _pin_daylight(chamber, monkeypatch, hour: int = 12, minute: int = 0):
    monkeypatch.setattr(
        chamber,
        "get_time",
        lambda: datetime(2026, 7, 24, hour, minute, tzinfo=UTC),
    )
    monkeypatch.setattr(
        chamber,
        "get_local_time",
        lambda: datetime(2026, 7, 24, hour, minute, tzinfo=UTC),
    )


@pytest.fixture
def chamber(monkeypatch):
    env = PlantGrowthChamber(zone="alliance-zone01", timezone="Etc/UTC")
    env.image = np.zeros((8, 8, 3), dtype=np.uint8)
    env.df = _valid_df()
    env.cv_state = {"ok": True}
    env._ensure_cv_session = AsyncMock(return_value=object())
    _pin_daylight(env, monkeypatch)
    return env


@pytest.mark.asyncio
async def test_cv_exception_reuses_last_df(chamber):
    prior = chamber.df.copy()
    chamber.cv_client.propagate = AsyncMock(side_effect=RuntimeError("503"))

    await chamber.get_plant_stats()

    pd.testing.assert_frame_equal(chamber.df, prior)


@pytest.mark.asyncio
async def test_empty_plant_stats_reuses_last_df(chamber):
    prior = chamber.df.copy()
    chamber.cv_client.propagate = AsyncMock(
        return_value={"state": {"ok": True}, "plant_stats": {}}
    )

    await chamber.get_plant_stats()

    pd.testing.assert_frame_equal(chamber.df, prior)
    assert chamber.cv_state == {"ok": True}


@pytest.mark.asyncio
async def test_none_response_reuses_last_df(chamber):
    prior = chamber.df.copy()
    chamber.cv_client.propagate = AsyncMock(return_value=None)

    await chamber.get_plant_stats()

    pd.testing.assert_frame_equal(chamber.df, prior)
    assert chamber.last_cv_time is None  # failed attempt should retry soon


@pytest.mark.asyncio
async def test_night_skip_keeps_last_df(chamber, monkeypatch):
    prior = chamber.df.copy()
    _pin_daylight(chamber, monkeypatch, hour=22, minute=0)

    await chamber.get_plant_stats()

    pd.testing.assert_frame_equal(chamber.df, prior)


@pytest.mark.asyncio
async def test_successful_cv_updates_df(chamber):
    chamber.cv_client.propagate = AsyncMock(
        return_value={
            "state": {"ok": True},
            "plant_stats": {
                "3": {"area": 120.0, "clean_area": 6.0},
                "4": {"area": 130.0, "clean_area": 7.0},
            },
        }
    )

    await chamber.get_plant_stats()

    assert list(chamber.df["pot_id"]) == ["3", "4"]
    assert chamber.df["clean_area"].tolist() == [6.0, 7.0]
    assert chamber.last_cv_time is not None
