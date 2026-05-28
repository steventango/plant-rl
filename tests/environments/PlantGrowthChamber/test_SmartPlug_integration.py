from datetime import datetime
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest

from environments.PlantGrowthChamber.PlantGrowthChamber import PlantGrowthChamber

UTC = ZoneInfo("Etc/UTC")


@pytest.fixture
def kasa_creds(monkeypatch):
    monkeypatch.setenv("KASA_USERNAME", "u")
    monkeypatch.setenv("KASA_PASSWORD", "p")


def _pin_time(chamber, monkeypatch, minute: int):
    monkeypatch.setattr(
        chamber, "get_time", lambda: datetime(2026, 5, 6, 12, minute, tzinfo=UTC)
    )


@pytest.mark.asyncio
async def test_get_power_carryover_records_null_on_failure(kasa_creds, monkeypatch):
    chamber = PlantGrowthChamber(zone="alliance-zone01", timezone="Etc/UTC")
    assert chamber.zone.smart_plug_host == "142.244.4.73"
    _pin_time(chamber, monkeypatch, minute=5)

    successful = {"power": 5.0, "voltage": 120.0, "current": 0.04}
    chamber.smart_plug_client.read = AsyncMock(side_effect=[successful, None])

    await chamber.get_power()
    assert chamber.power == successful
    assert chamber.power_record == successful
    info = chamber.get_info()
    assert info["power"] == 5.0
    assert info["voltage"] == 120.0
    assert info["current"] == 0.04

    await chamber.get_power()
    assert chamber.power == successful
    assert chamber.power_record == {"power": None, "voltage": None, "current": None}
    info = chamber.get_info()
    assert info["power"] is None
    assert info["voltage"] is None
    assert info["current"] is None


@pytest.mark.asyncio
async def test_get_power_is_noop_when_zone_has_no_plug(kasa_creds):
    chamber = PlantGrowthChamber(zone="mitacs-zone03", timezone="Etc/UTC")
    assert chamber.zone.smart_plug_host is None
    assert chamber.smart_plug_client is None

    await chamber.get_power()

    assert chamber.power_record == {}
    info = chamber.get_info()
    assert "power" not in info
    assert "voltage" not in info
    assert "current" not in info


@pytest.mark.asyncio
async def test_get_power_fetches_on_five_minute_boundary(kasa_creds, monkeypatch):
    chamber = PlantGrowthChamber(zone="alliance-zone01", timezone="Etc/UTC")
    successful = {"power": 5.0, "voltage": 120.0, "current": 0.04}
    chamber.smart_plug_client.read = AsyncMock(return_value=successful)

    _pin_time(chamber, monkeypatch, minute=5)
    await chamber.get_power()
    assert chamber.power_record == successful

    _pin_time(chamber, monkeypatch, minute=6)
    await chamber.get_power()
    assert chamber.smart_plug_client.read.await_count == 1
    assert chamber.power_record == successful
    assert chamber.power == successful

    _pin_time(chamber, monkeypatch, minute=10)
    await chamber.get_power()
    assert chamber.smart_plug_client.read.await_count == 2


@pytest.mark.asyncio
async def test_get_power_recovers_when_boundary_is_missed(kasa_creds, monkeypatch):
    """If the step loop drifts past a 5-min boundary, the next call still fetches."""
    chamber = PlantGrowthChamber(zone="alliance-zone01", timezone="Etc/UTC")
    successful = {"power": 5.0, "voltage": 120.0, "current": 0.04}
    chamber.smart_plug_client.read = AsyncMock(return_value=successful)

    _pin_time(chamber, monkeypatch, minute=5)
    await chamber.get_power()
    assert chamber.smart_plug_client.read.await_count == 1

    # Step loop missed minute=10 entirely, lands at :11. Not on boundary,
    # but elapsed is 6 min > 5 min interval → fetch via the overdue fallback.
    _pin_time(chamber, monkeypatch, minute=11)
    await chamber.get_power()
    assert chamber.smart_plug_client.read.await_count == 2
