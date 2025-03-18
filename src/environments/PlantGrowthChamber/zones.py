from dataclasses import dataclass, field


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
    lightbar_url: str
    trays: list[Tray]

    @property
    def num_plants(self) -> int:
        return sum(tray.num_plants for tray in self.trays)


def get_zone(indentifier: int):
    match indentifier:
        case 1:
            return Zone(
                identifier=1,
                camera_left_url="http://mitacs-zone01-camera02.ccis.ualberta.ca:8080/observation",
                camera_right_url=None,
                lightbar_url="http://mitacs-zone1.ccis.ualberta.ca:8080/action",
                trays=[
                    Tray(
                        n_wide=6,
                        n_tall=4,
                        rect=Rect(
                            top_left=(605, 54),
                            top_right=(1872, 28),
                            bottom_left=(549, 866),
                            bottom_right=(1897, 916),
                        ),
                    ),
                    Tray(
                        n_wide=6,
                        n_tall=4,
                        rect=Rect(
                            top_left=(537, 969),
                            top_right=(1895, 1000),
                            bottom_left=(591, 1747),
                            bottom_right=(1791, 1830),
                        ),
                    ),
                ],
            )
        case 2:
            return Zone(
                identifier=2,
                camera_left_url=None,
                camera_right_url="http://mitacs-zone02-camera02.ccis.ualberta.ca:8080/observation",
                lightbar_url="http://mitacs-zone2.ccis.ualberta.ca:8080/action",
                trays=[
                    Tray(
                        n_wide=6,
                        n_tall=4,
                        rect=Rect(
                            top_left=(656, 52),
                            top_right=(1922, 66),
                            bottom_left=(620, 927),
                            bottom_right=(1974, 900),
                        ),
                    ),
                    Tray(
                        n_wide=6,
                        n_tall=4,
                        rect=Rect(
                            top_left=(639, 1023),
                            top_right=(1987, 979),
                            bottom_left=(680, 1875),
                            bottom_right=(1946, 1764),
                        ),
                    ),
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
                trays=[
                    Tray(
                        n_wide=4,
                        n_tall=4,
                        rect=Rect(
                            top_left=(1278, 137),
                            top_right=(2133, 200),
                            bottom_left=(1260, 1041),
                            bottom_right=(2163, 1050),
                        ),
                    )
                ],
            )
        case 8:
            return Zone(
                identifier=8,
                camera_left_url="http://mitacs-zone08-camera01.ccis.ualberta.ca:8080/observation",
                camera_right_url=None,
                lightbar_url="http://mitacs-zone8.ccis.ualberta.ca:8080/action",
                trays=[
                    Tray(
                        n_wide=4,
                        n_tall=4,
                        rect=Rect(
                            top_left=(1232, 175),
                            top_right=(2157, 258),
                            bottom_left=(1314, 1080),
                            bottom_right=(2158, 1081),
                        ),
                    )
                ],
            )
        case 9:
            return Zone(
                identifier=9,
                camera_left_url=None,
                camera_right_url="http://mitacs-zone09-camera01.ccis.ualberta.ca:8080/observation",
                lightbar_url="http://mitacs-zone9.ccis.ualberta.ca:8080/action",
                trays=[
                    Tray(
                        n_wide=4,
                        n_tall=4,
                        rect=Rect(
                            top_left=(366, 702),
                            top_right=(1188, 698),
                            bottom_left=(377, 1533),
                            bottom_right=(1188, 1604),
                        ),
                    )
                ],
            )
        case _:
            raise ValueError(f"Unknown zone indentifier: {indentifier}")


POT_HEIGHT = 60 * 4
POT_WIDTH = 60 * 4
