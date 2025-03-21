from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from plantcv import plantcv as pcv

from environments.PlantGrowthChamber.utils import process_image
from environments.PlantGrowthChamber.zones import Rect, Tray, get_zone
from utils.metrics import iqm

TEST_DIR = Path(__file__).parent.parent.parent / "test_data"
SC_TEST_DIR = TEST_DIR / "Spreadsheet-C"


def get_plant_area(zone_id: int):
    dfs = []
    zone = get_zone(zone_id)
    zone_dir = SC_TEST_DIR / f"z{zone_id}"
    out_dir = zone_dir / "results"
    out_dir.mkdir(exist_ok=True)
    paths = sorted(zone_dir.glob("*.png"))
    for path in paths:
        image = np.array(Image.open(path))
        debug_images = {}
        df = process_image(image, zone.trays, debug_images)
        df["intensity"] = path.stem

        avg = iqm(jnp.array(df["area"]), 0.05)
        df = pd.concat([df, pd.DataFrame({
            "plant_id": ["iqm"],
            "area": [avg],
            "intensity": [path.stem]
        })])
        dfs.append(df)
        df.to_csv(out_dir / f"{path.stem}.csv", index=False)
        for key, value in debug_images.items():
            value.save(out_dir / f"{path.stem}_{key}.png")

    df = pd.concat(dfs)
    plot_area_comparison(df, out_dir)


def test_process_zone_1():
    get_plant_area(1)


def test_process_zone_6():
    get_plant_area(6)


def plot_area_comparison(df: pd.DataFrame, out_dir: Path):
    plt.figure(figsize=(4 * len(df) // 32, 3))
    sns.barplot(df, x="plant_id", y="area", hue="intensity")
    plt.ylim(0, df["area"].quantile(0.99))
    plt.savefig(out_dir / "areas.png")


def test_alg():

    gray_img = np.array(Image.open("tests/test_data/z3c1--2022-12-31--08-50-01.png"))[:, :, 1]

    bin_gauss1 = pcv.threshold.gaussian(gray_img=gray_img, ksize=2500, offset=-50, object_type="light")

    # save
    Image.fromarray(bin_gauss1).save("tests/test_data/bin_gauss1.png")

    gray_img2 = np.array(Image.open("tests/test_data/z3c1--2022-12-31--09-00-01.png"))[:, :, 1]
    Image.fromarray(gray_img2).save("tests/test_data/gray_img2.png")
    bin_gauss2 = pcv.threshold.gaussian(gray_img=gray_img2, ksize=2500, offset=-50, object_type="light")

    # save
    Image.fromarray(bin_gauss2).save("tests/test_data/bin_gauss2.png")
