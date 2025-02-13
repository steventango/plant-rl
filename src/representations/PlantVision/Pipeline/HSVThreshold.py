from pathlib import Path
import json
import numpy as np
import cv2
from plantcv import plantcv as pcv


class HSVThreshold:
    def __init__(
        self,
        hl: int,
        hh: int,
        sl: int,
        sh: int,
        vl: int,
        vh: int,
        fill: int,
        invert: bool,
    ) -> None:
        self.hl: int = hl
        self.hh: int = hh
        self.sl: int = sl
        self.sh: int = sh
        self.vl: int = vl
        self.vh: int = vh
        self.fill: int = fill
        self.invert: bool = invert

        self.upper = np.array([hh, sh, vh])
        self.lower = np.array([hl, sl, vl])

    def threshold(self, image):
        # Returns a binary image with green pixels as white, and the rest black.
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, self.lower, self.upper)
        filled = pcv.fill(mask, self.fill)
        return filled

    def save_to_file(self, dirpath: Path):
        with open(dirpath / "threshold.json", "w") as f:
            json.dump(
                {
                    "hl": str(self.hl),
                    "hh": str(self.hh),
                    "sl": str(self.sl),
                    "sh": str(self.sh),
                    "vl": str(self.vl),
                    "vh": str(self.vh),
                    "fill": str(self.fill),
                    "invert": self.invert,
                },
                f,
            )


def load_threshold_from_file(path: Path) -> HSVThreshold:
    with open(path, "r") as f:
        d = json.load(f)

        return HSVThreshold(
            hl=int(d["hl"]),
            hh=int(d["hh"]),
            sl=int(d["sl"]),
            sh=int(d["sh"]),
            vl=int(d["vl"]),
            vh=int(d["vh"]),
            fill=int(d["fill"]),
            invert=bool(d["invert"]),
        )


def load_threshold_from_dict(d: dict) -> HSVThreshold:
    return HSVThreshold(
        hl=int(d["hl"]),
        hh=int(d["hh"]),
        sl=int(d["sl"]),
        sh=int(d["sh"]),
        vl=int(d["vl"]),
        vh=int(d["vh"]),
        fill=int(d["fill"]),
        invert=bool(d["invert"]),
    )
