from dataclasses import dataclass


@dataclass
class Zone:
    indentifier: int
    camera_left_url: str | None
    camera_right_url: str | None
    lightbar_url: str


def get_zone(indentifier: int):
    match indentifier:
        case 1:
            return Zone(
                indentifier=1,
                camera_left_url="http://mitacs-zone01-camera01.ccis.ualberta.ca:8080/observation",
                camera_right_url=None,
                lightbar_url="http://mitacs-zone1.ccis.ualberta.ca:8080/action/left"
            )
        case 2:
            return Zone(
                indentifier=2,
                camera_left_url=None,
                camera_right_url="http://mitacs-zone02-camera01.ccis.ualberta.ca:8080/observation",
                lightbar_url="http://mitacs-zone2.ccis.ualberta.ca:8080/action/left"
            )
        case 3:
            return Zone(
                indentifier=3,
                camera_left_url="http://mitacs-zone03-camera01.ccis.ualberta.ca:8080/observation",
                camera_right_url=None,
                lightbar_url="http://mitacs-zone3.ccis.ualberta.ca:8080/action/left"
            )
        case 6:
            return Zone(
                indentifier=6,
                camera_left_url="http://mitacs-zone06-camera01.ccis.ualberta.ca:8080/observation",
                camera_right_url=None,
                lightbar_url="http://mitacs-zone6.ccis.ualberta.ca:8080/action/left"
            )
        case 8:
            return Zone(
                indentifier=8,
                camera_left_url="http://mitacs-zone08-camera01.ccis.ualberta.ca:8080/observation",
                camera_right_url=None,
                lightbar_url="http://mitacs-zone8.ccis.ualberta.ca:8080/action/left"
            )
        case 9:
            return Zone(
                indentifier=9,
                camera_left_url=None,
                camera_right_url="http://mitacs-zone09-camera01.ccis.ualberta.ca:8080/observation",
                lightbar_url="http://mitacs-zone9.ccis.ualberta.ca:8080/action/left"
            )
        case _:
            raise ValueError(f"Unknown zone indentifier: {indentifier}")
