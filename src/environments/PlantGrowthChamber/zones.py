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
    identifier: int
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


def load_zone_from_config(identifier: int) -> Zone:
    with open(CONFIG_DIR / f"z{identifier}.json") as f:
        config = json.load(f)
    return deserialize_zone(config["zone"])


def get_zone(indentifier: int):
    match indentifier:
        case 1:
            return load_zone_from_config(1)
        case 2:
            return load_zone_from_config(2)
        case 3:
            return load_zone_from_config(3)
        case 6:
            return load_zone_from_config(6)
        case 8:
            return load_zone_from_config(8)
        case 9:
            return load_zone_from_config(9)
        case _:
            raise ValueError(f"Unknown zone indentifier: {indentifier}")


SCALE = 4
POT_HEIGHT = 60 * SCALE
POT_WIDTH = 60 * SCALE
