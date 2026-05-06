import logging
import os

from kasa import Discover, Module
from kasa.exceptions import KasaException

logger = logging.getLogger("plant_rl.SmartPlugClient")

POWER_KEYS = ("power", "voltage", "current")


class SmartPlugClient:
    def __init__(self):
        self.username = os.getenv("KASA_USERNAME")
        self.password = os.getenv("KASA_PASSWORD")
        if not (self.username and self.password):
            logger.warning(
                "KASA_USERNAME / KASA_PASSWORD not set; SmartPlugClient will no-op"
            )

    async def read(self, host: str) -> dict | None:
        """Return {"power", "voltage", "current"} or None on any failure."""
        if not (self.username and self.password):
            return None
        dev = None
        try:
            dev = await Discover.discover_single(
                host, username=self.username, password=self.password
            )
            if dev is None:
                return None
            await dev.update()
            energy = dev.modules.get(Module.Energy)
            if energy is None:
                logger.warning("kasa device %s has no Energy module", host)
                return None
            reading = {
                "power": energy.current_consumption,
                "voltage": energy.voltage,
                "current": energy.current,
            }
            if not any(v is not None for v in reading.values()):
                logger.warning("kasa device %s returned no readings", host)
                return None
            return reading
        except KasaException:
            logger.warning("kasa error reading %s", host, exc_info=True)
            return None
        except Exception:
            logger.warning("unexpected error reading %s", host, exc_info=True)
            return None
        finally:
            if dev is not None:
                await dev.disconnect()
