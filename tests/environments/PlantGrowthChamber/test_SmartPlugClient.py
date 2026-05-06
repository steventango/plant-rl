from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kasa.exceptions import KasaException

from environments.PlantGrowthChamber.SmartPlugClient import SmartPlugClient


@pytest.fixture
def kasa_creds(monkeypatch):
    monkeypatch.setenv("KASA_USERNAME", "user@example.com")
    monkeypatch.setenv("KASA_PASSWORD", "pw")


def _fake_device(power=7.268, voltage=122.791, current=0.07, energy=True):
    energy_module = MagicMock()
    energy_module.current_consumption = power
    energy_module.voltage = voltage
    energy_module.current = current

    dev = MagicMock()
    dev.update = AsyncMock()
    dev.disconnect = AsyncMock()
    dev.modules = MagicMock()
    dev.modules.get = MagicMock(return_value=energy_module if energy else None)
    return dev


@pytest.mark.asyncio
async def test_read_returns_renamed_keys(kasa_creds):
    dev = _fake_device(power=7.268, voltage=122.791, current=0.07)
    with patch(
        "environments.PlantGrowthChamber.SmartPlugClient.Discover.discover_single",
        AsyncMock(return_value=dev),
    ):
        client = SmartPlugClient()
        result = await client.read("142.244.4.73")

    assert result == {"power": 7.268, "voltage": 122.791, "current": 0.07}
    dev.disconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_read_returns_none_without_credentials(monkeypatch):
    monkeypatch.delenv("KASA_USERNAME", raising=False)
    monkeypatch.delenv("KASA_PASSWORD", raising=False)
    client = SmartPlugClient()
    assert await client.read("142.244.4.73") is None


@pytest.mark.asyncio
async def test_read_returns_none_on_kasa_exception(kasa_creds):
    with patch(
        "environments.PlantGrowthChamber.SmartPlugClient.Discover.discover_single",
        AsyncMock(side_effect=KasaException("boom")),
    ):
        client = SmartPlugClient()
        assert await client.read("142.244.4.73") is None


@pytest.mark.asyncio
async def test_read_returns_none_when_no_energy_module(kasa_creds):
    dev = _fake_device(energy=False)
    with patch(
        "environments.PlantGrowthChamber.SmartPlugClient.Discover.discover_single",
        AsyncMock(return_value=dev),
    ):
        client = SmartPlugClient()
        assert await client.read("142.244.4.73") is None
    dev.disconnect.assert_awaited_once()
