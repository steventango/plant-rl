import json
from dataclasses import dataclass
from pathlib import Path

from environments.PlantGrowthChamber.Calibration import Calibration

CONFIG_DIR = Path(__file__).parent / "configs"


@dataclass
class Zone:
    identifier: str
    camera_left_url: str | None
    camera_right_url: str | None
    lightbar_url: str | None
    calibration: Calibration | None
    smart_plug_host: str | None = None


def deserialize_zone(zone: dict) -> Zone:
    calibration_data = zone.get("calibration")
    calibration = Calibration(**calibration_data) if calibration_data else None

    return Zone(
        identifier=zone["identifier"],
        camera_left_url=zone.get("camera_left_url"),
        camera_right_url=zone.get("camera_right_url"),
        lightbar_url=zone.get("lightbar_url"),
        smart_plug_host=zone.get("smart_plug_host"),
        calibration=calibration,
    )


def serialize_zone(zone: Zone) -> dict:
    calibration_data = zone.calibration.to_dict() if zone.calibration else None

    return {
        "identifier": zone.identifier,
        "camera_left_url": zone.camera_left_url,
        "camera_right_url": zone.camera_right_url,
        "lightbar_url": zone.lightbar_url,
        "smart_plug_host": zone.smart_plug_host,
        "calibration": calibration_data,
    }


def load_zone_from_config(identifier: str) -> Zone:
    with open(CONFIG_DIR / f"{identifier}.json") as f:
        config = json.load(f)
    return deserialize_zone(config["zone"])


ZONE_IDENTIFIERS = [
    "alliance-zone01",
    "alliance-zone02",
    "alliance-zone03",
    "alliance-zone04",
    "alliance-zone05",
    "alliance-zone06",
    "alliance-zone07",
    "alliance-zone08",
    "alliance-zone09",
    "alliance-zone10",
    "alliance-zone11",
    "alliance-zone12",
    # "mitacs-zone01",
    # "mitacs-zone02",
    # "mitacs-zone03",
    # "mitacs-zone06",
    # "mitacs-zone08",
    # "mitacs-zone09",
]
