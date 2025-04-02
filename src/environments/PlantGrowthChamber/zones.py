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
                        n_tall=3,
                        rect=Rect(
                            top_left=(438, 261),
                            top_right=(1715, 212),
                            bottom_left=(410, 879),
                            bottom_right=(1727, 906),
                        ),
                    ),
                    Tray(
                        n_wide=6,
                        n_tall=3,
                        rect=Rect(
                            top_left=(399, 965),
                            top_right=(1700, 993),
                            bottom_left=(437, 1548),
                            bottom_right=(1655, 1646),
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
                        n_tall=3,
                        rect=Rect(
                            top_left=(882, 240),
                            top_right=(2142, 276),
                            bottom_left=(870, 933),
                            bottom_right=(2190, 906),
                        ),
                    ),
                    Tray(
                        n_wide=6,
                        n_tall=3,
                        rect=Rect(
                            top_left=(891, 1014),
                            top_right=(2193, 984),
                            bottom_left=(936, 1680),
                            bottom_right=(2160, 1578),
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
                trays=[
                    Tray(
                        n_wide=8,
                        n_tall=3,
                        rect=Rect(
                            top_left=(105, 405),
                            top_right=(1717, 198),
                            bottom_left=(110, 992),
                            bottom_right=(1813, 843),
                        ),
                    ),
                    Tray(
                        n_wide=8,
                        n_tall=3,
                        rect=Rect(
                            top_left=(90, 1061),
                            top_right=(1806, 926),
                            bottom_left=(170, 1654),
                            bottom_right=(1833, 1594),
                        ),
                    )
                ],
            )
        case 6:
            return Zone(
                identifier=6,
                camera_left_url="http://mitacs-zone06-camera01.ccis.ualberta.ca:8080/observation",
                camera_right_url=None,
                lightbar_url="http://mitacs-zone6.ccis.ualberta.ca:8080/action",
                trays=[
                    Tray(
                        n_wide=2,
                        n_tall=6,
                        rect=Rect(
                            top_left=(742, 315),
                            top_right=(1190, 293),
                            bottom_left=(783, 1588),
                            bottom_right=(1191, 1605),
                        ),
                    ),
                    Tray(
                        n_wide=2,
                        n_tall=6,
                        rect=Rect(
                            top_left=(1269, 324),
                            top_right=(1731, 337),
                            bottom_left=(1269, 1638),
                            bottom_right=(1685, 1635),
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

SCALE = 4
POT_HEIGHT = 60 * SCALE
POT_WIDTH = 60 * SCALE
