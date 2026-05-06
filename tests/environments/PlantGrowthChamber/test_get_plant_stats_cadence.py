from datetime import datetime
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from environments.PlantGrowthChamber.PlantGrowthChamber import PlantGrowthChamber

UTC = ZoneInfo("Etc/UTC")


def _pin_time(chamber, monkeypatch, minute: int):
    monkeypatch.setattr(
        chamber, "get_time", lambda: datetime(2026, 5, 6, 12, minute, tzinfo=UTC)
    )


def _stub_cv_client(chamber):
    """Stub the CV pipeline so tests don't need a running pipeline service."""
    chamber.cv_client.detect = AsyncMock(return_value={"plant_stats": {}, "state": "s"})
    chamber.cv_client.propagate = AsyncMock(
        return_value={"plant_stats": {}, "state": "s"}
    )
    chamber._ensure_session = AsyncMock(return_value=None)


@pytest.mark.asyncio
async def test_get_plant_stats_fetches_on_five_minute_boundary(monkeypatch):
    chamber = PlantGrowthChamber(zone="alliance-zone01", timezone="Etc/UTC")
    chamber.image = np.zeros((10, 10, 3), dtype=np.uint8)
    _stub_cv_client(chamber)

    _pin_time(chamber, monkeypatch, minute=5)
    await chamber.get_plant_stats()
    assert chamber.cv_client.detect.await_count == 1

    _pin_time(chamber, monkeypatch, minute=6)
    await chamber.get_plant_stats()
    assert chamber.cv_client.propagate.await_count == 0

    _pin_time(chamber, monkeypatch, minute=10)
    await chamber.get_plant_stats()
    assert chamber.cv_client.propagate.await_count == 1


@pytest.mark.asyncio
async def test_get_plant_stats_recovers_when_boundary_is_missed(monkeypatch):
    chamber = PlantGrowthChamber(zone="alliance-zone01", timezone="Etc/UTC")
    chamber.image = np.zeros((10, 10, 3), dtype=np.uint8)
    _stub_cv_client(chamber)

    _pin_time(chamber, monkeypatch, minute=5)
    await chamber.get_plant_stats()
    assert chamber.cv_client.detect.await_count == 1

    _pin_time(chamber, monkeypatch, minute=11)
    await chamber.get_plant_stats()
    assert chamber.cv_client.propagate.await_count == 1
