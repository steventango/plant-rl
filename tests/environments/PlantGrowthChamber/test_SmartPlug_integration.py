from unittest.mock import AsyncMock

import pytest

from environments.PlantGrowthChamber.PlantGrowthChamber import PlantGrowthChamber


@pytest.fixture
def kasa_creds(monkeypatch):
    monkeypatch.setenv("KASA_USERNAME", "u")
    monkeypatch.setenv("KASA_PASSWORD", "p")


@pytest.mark.asyncio
async def test_get_power_carryover_records_null_on_failure(kasa_creds):
    chamber = PlantGrowthChamber(zone="alliance-zone01", timezone="Etc/UTC")
    assert chamber.zone.smart_plug_host == "142.244.4.73"

    successful = {"power": 5.0, "voltage": 120.0, "current": 0.04}
    chamber.smart_plug_client.read = AsyncMock(side_effect=[successful, None])

    await chamber.get_power()
    assert chamber.power == successful
    assert chamber.power_record == successful
    info = chamber.get_info()
    assert info["power"] == 5.0
    assert info["voltage"] == 120.0
    assert info["current"] == 0.04

    chamber.last_smart_plug_time = None

    await chamber.get_power()
    assert chamber.power == successful
    assert chamber.power_record == {"power": None, "voltage": None, "current": None}
    info = chamber.get_info()
    assert info["power"] is None
    assert info["voltage"] is None
    assert info["current"] is None


@pytest.mark.asyncio
async def test_get_power_is_noop_when_zone_has_no_plug(kasa_creds):
    chamber = PlantGrowthChamber(zone="alliance-zone03", timezone="Etc/UTC")
    assert chamber.zone.smart_plug_host is None
    assert chamber.smart_plug_client is None

    await chamber.get_power()

    assert chamber.power_record == {}
    info = chamber.get_info()
    assert "power" not in info
    assert "voltage" not in info
    assert "current" not in info


@pytest.mark.asyncio
async def test_get_power_respects_5_minute_gate(kasa_creds):
    chamber = PlantGrowthChamber(zone="alliance-zone01", timezone="Etc/UTC")
    successful = {"power": 5.0, "voltage": 120.0, "current": 0.04}
    chamber.smart_plug_client.read = AsyncMock(return_value=successful)

    await chamber.get_power()
    assert chamber.power_record == successful

    await chamber.get_power()
    assert chamber.smart_plug_client.read.await_count == 1
    assert chamber.power_record == successful
    assert chamber.power == successful
