from pathlib import Path
import json
from plantcv import plantcv as pcv


class Roi:
    def __init__(self, cx: int, cy: int, r: int, number: int, genotype: str) -> None:
        self.cx: int = cx
        self.cy: int = cy
        self.r: int = r
        self.number: int = number
        self.genotype: str = genotype

    def __str__(self) -> str:
        return str(self.__dict__)


class RoiList:
    def __init__(self) -> None:
        self.data: list = []
        self.pcv_rois = None

    def create_pcv_rois(self, undistorted_img):
        if self.pcv_rois == None:
            contours = []
            hiers = []
            for roi in self.data:
                roi : Roi
                r_obj = pcv.roi.circle(img=undistorted_img, x=roi.cx, y=roi.cy, r=roi.r)
                contours.append(r_obj.contours)
                hiers.append(r_obj.hierarchy)
            self.pcv_rois = pcv.Objects(contours=contours, hierarchy=hiers)
        return self.pcv_rois

    def load_from_file(self, path: Path) -> bool:
        try:
            with open(path, "r") as f:
                l = json.load(f)
                for rd in l:
                    cx: int = int(rd["cx"])
                    cy: int = int(rd["cy"])
                    r: int = int(rd["r"])
                    number: int = int(rd["number"])
                    genotype: str = rd["genotype"]
                    self.data.append(
                        Roi(cx=cx, cy=cy, r=r, number=number, genotype=genotype)
                    )
        except Exception as e:
            print("ERROR", e)
            return False
        return True
    
    def load_from_list(self, l: list) -> bool:
        try:
            for rd in l:
                cx: int = int(rd["cx"])
                cy: int = int(rd["cy"])
                r: int = int(rd["r"])
                number: int = int(rd["number"])
                genotype: str = rd["genotype"]
                self.data.append(
                    Roi(cx=cx, cy=cy, r=r, number=number, genotype=genotype)
                )
        except Exception as e:
            print("ERROR", e)
            return False
        return True

    def save_to_file(self, dirpath: Path):
        json_l = []
        for roi in self.data:
            roi: Roi
            json_l.append(
                {
                    "cx": roi.cx,
                    "cy": roi.cy,
                    "r": roi.r,
                    "number": roi.number,
                    "genotype": roi.genotype,
                }
            )

        with open(dirpath / "rois.json", "w") as f:
            json.dump(json_l, f)

