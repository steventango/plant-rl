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
                            top_left=(528, 232),
                            top_right=(1806, 195),
                            bottom_left=(504, 843),
                            bottom_right=(1815, 882),
                        ),
                    ),
                    Tray(
                        n_wide=6,
                        n_tall=3,
                        rect=Rect(
                            top_left=(489, 927),
                            top_right=(1791, 978),
                            bottom_left=(523, 1532),
                            bottom_right=(1728, 1669),
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
                            top_left=(483, 279),
                            top_right=(1752, 300),
                            bottom_left=(471, 969),
                            bottom_right=(1791, 909),
                        ),
                    ),
                    Tray(
                        n_wide=6,
                        n_tall=3,
                        rect=Rect(
                            top_left=(498, 1068),
                            top_right=(1806, 990),
                            bottom_left=(590, 1750),
                            bottom_right=(1817, 1593),
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
                        n_wide=3,
                        n_tall=6,
                        rect=Rect(
                            top_left=(541, 450),
                            top_right=(1199, 421),
                            bottom_left=(624, 1676),
                            bottom_right=(1212, 1717),
                        ),
                    ),
                    Tray(
                        n_wide=3,
                        n_tall=6,
                        rect=Rect(
                            top_left=(1295, 429),
                            top_right=(1961, 446),
                            bottom_left=(1291, 1717),
                            bottom_right=(1893, 1690),
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
                        n_wide=3,
                        n_tall=6,
                        rect=Rect(
                            top_left=(1341, 433),
                            top_right=(1998, 487),
                            bottom_left=(1289, 1733),
                            bottom_right=(1883, 1705),
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
                        n_wide=3,
                        n_tall=6,
                        rect=Rect(
                            top_left=(609, 116),
                            top_right=(1195, 77),
                            bottom_left=(544, 1347),
                            bottom_right=(1190, 1367),
                        ),
                    ),
                    Tray(
                        n_wide=3,
                        n_tall=6,
                        rect=Rect(
                            top_left=(1286, 73),
                            top_right=(1878, 111),
                            bottom_left=(1295, 1360),
                            bottom_right=(1953, 1333),
                        ),
                    )
                ],
            )
        case _:
            raise ValueError(f"Unknown zone indentifier: {indentifier}")

SCALE = 4
POT_HEIGHT = 60 * SCALE
POT_WIDTH = 60 * SCALE
