import json
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_DIR = Path(__file__).parent / "configs"


@dataclass
class Rect:
    top_left: tuple[int, int]
    top_right: tuple[int, int]
    bottom_left: tuple[int, int]
    bottom_right: tuple[int, int]


@dataclass
class Tray:
    n_tall: int
    n_wide: int
    rect: Rect
    num_plants: int = field(init=False)

    def __post_init__(self):
        self.num_plants = self.n_tall * self.n_wide


@dataclass
class Zone:
    identifier: str
    camera_left_url: str | None
    camera_right_url: str | None
    lightbar_url: str | None
    trays: list[Tray]

    @property
    def num_plants(self) -> int:
        return sum(tray.num_plants for tray in self.trays)


def deserialize_zone(zone: dict) -> Zone:
    return Zone(
        identifier=zone["identifier"],
        camera_left_url=zone.get("camera_left_url"),
        camera_right_url=zone.get("camera_right_url"),
        lightbar_url=zone.get("lightbar_url"),
        trays=[
            Tray(
                n_wide=tray["n_wide"],
                n_tall=tray["n_tall"],
                rect=Rect(
                    top_left=tray["rect"]["top_left"],
                    top_right=tray["rect"]["top_right"],
                    bottom_left=tray["rect"]["bottom_left"],
                    bottom_right=tray["rect"]["bottom_right"],
                ),
            )
            for tray in zone["trays"]
        ],
    )


def serialize_zone(zone: Zone) -> dict:
    return {
        "identifier": zone.identifier,
        "camera_left_url": zone.camera_left_url,
        "camera_right_url": zone.camera_right_url,
        "lightbar_url": zone.lightbar_url,
        "trays": [
            {
                "n_wide": tray.n_wide,
                "n_tall": tray.n_tall,
                "rect": {
                    "top_left": tray.rect.top_left,
                    "top_right": tray.rect.top_right,
                    "bottom_left": tray.rect.bottom_left,
                    "bottom_right": tray.rect.bottom_right,
                },
            }
            for tray in zone.trays
        ],
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

SCALE = 4
POT_HEIGHT = 60 * SCALE
POT_WIDTH = 60 * SCALE
