from dataclasses import dataclass


@dataclass
class Rect:
    top_left: tuple[int, int]
    top_right: tuple[int, int]
    bottom_left: tuple[int, int]
    bottom_right: tuple[int, int]


@dataclass
class Tray:
    num_plants: int
    n_tall: int
    n_wide: int
    rect: Rect


@dataclass
class Zone:
    identifier: int
    camera_left_url: str | None
    camera_right_url: str | None
    lightbar_url: str
    trays: list[Tray]


def get_zone(indentifier: int):
    match indentifier:
        case 1:
            return Zone(
                identifier=1,
                camera_left_url="http://mitacs-zone01-camera02.ccis.ualberta.ca:8080/observation",
                camera_right_url=None,
                lightbar_url="http://mitacs-zone1.ccis.ualberta.ca:8080/action",
                trays=[],
            )
        case 2:
            return Zone(
                identifier=2,
                camera_left_url=None,
                camera_right_url="http://mitacs-zone02-camera02.ccis.ualberta.ca:8080/observation",
                lightbar_url="http://mitacs-zone2.ccis.ualberta.ca:8080/action",
                trays=[
                    Tray(
                        num_plants=16,
                        n_wide=4,
                        n_tall=4,
                        rect=Rect(
                            top_left=(1241, 978),
                            top_right=(2017, 952),
                            bottom_left=(1258, 1804),
                            bottom_right=(1972, 1667),
                        ),
                    )
                ],
            )
        case 3:
            return Zone(
                identifier=3,
                camera_left_url="http://mitacs-zone03-camera01.ccis.ualberta.ca:8080/observation",
                camera_right_url=None,
                lightbar_url="http://mitacs-zone3.ccis.ualberta.ca:8080/action",
                trays=[],
            )
        case 6:
            return Zone(
                identifier=6,
                camera_left_url="http://mitacs-zone06-camera01.ccis.ualberta.ca:8080/observation",
                camera_right_url=None,
                lightbar_url="http://mitacs-zone6.ccis.ualberta.ca:8080/action",
                trays=[],
            )
        case 8:
            return Zone(
                identifier=8,
                camera_left_url="http://mitacs-zone08-camera01.ccis.ualberta.ca:8080/observation",
                camera_right_url=None,
                lightbar_url="http://mitacs-zone8.ccis.ualberta.ca:8080/action",
                trays=[],
            )
        case 9:
            return Zone(
                identifier=9,
                camera_left_url=None,
                camera_right_url="http://mitacs-zone09-camera01.ccis.ualberta.ca:8080/observation",
                lightbar_url="http://mitacs-zone9.ccis.ualberta.ca:8080/action",
                trays=[],
            )
        case _:
            raise ValueError(f"Unknown zone indentifier: {indentifier}")
