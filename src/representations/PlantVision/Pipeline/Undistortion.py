import json
from pathlib import Path

import cv2
import numpy as np


class Undistortion:
    def __init__(
        self,
        fx: float,
        cx: float,
        fy: float,
        cy: float,
        k1: float,
        k2: float,
        k3: float,
        k4: float,
    ) -> None:
        self.fx: float = fx
        self.cx: float = cx
        self.fy: float = fy
        self.cy: float = cy
        self.k1: float = k1
        self.k2: float = k2
        self.k3: float = k3
        self.k4: float = k4

        self.K = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
        )  # camera intrinsic matrix

        self.D = np.array(
            [k1, k2, k3, k4], dtype=np.float64
        )  # camera distortion matrix

    def undistort(self, distorted_img):
        balance = 1
        dim = (distorted_img.shape[1], distorted_img.shape[0])

        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.K, self.D, dim, np.eye(3), balance=balance
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K, self.D, np.eye(3), new_K, dim, cv2.CV_32FC1
        )
        # and then remap:
        undistorted = cv2.remap(
            distorted_img,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        return undistorted

    def save_to_file(self, dirpath: Path):
        with open(dirpath / "undistort.json", "w") as f:
            json.dump(
                {
                    "fx": str(self.fx),
                    "cx": str(self.cx),
                    "fy": str(self.fy),
                    "cy": str(self.cy),
                    "k1": str(self.k1),
                    "k2": str(self.k2),
                    "k3": str(self.k3),
                    "k4": str(self.k4),
                },
                f,
            )


def load_undistortion_from_file(path: Path) -> Undistortion:
    with open(path, "r") as f:
        d = json.load(f)

        return Undistortion(
            fx=float(d["fx"]),
            cx=float(d["cx"]),
            fy=float(d["fy"]),
            cy=float(d["cy"]),
            k1=float(d["k1"]),
            k2=float(d["k2"]),
            k3=float(d["k3"]),
            k4=float(d["k4"]),
        )


def load_undistortion_from_dict(d: dict) -> Undistortion:
    return Undistortion(
        fx=float(d["fx"]),
        cx=float(d["cx"]),
        fy=float(d["fy"]),
        cy=float(d["cy"]),
        k1=float(d["k1"]),
        k2=float(d["k2"]),
        k3=float(d["k3"]),
        k4=float(d["k4"]),
    )
