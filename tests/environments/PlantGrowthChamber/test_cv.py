from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

from environments.PlantGrowthChamber.cv import process_image
from environments.PlantGrowthChamber.zones import Rect, Tray, Zone, get_zone
from utils.metrics import iqm

TEST_DIR = Path(__file__).parent.parent.parent / "test_data"
OLD_TEST_DIR = TEST_DIR / "old"
SC_TEST_DIR = TEST_DIR / "Spreadsheet-C"
SC_V2_TEST_DIR = TEST_DIR / "Spreadsheet-C"


def get_plant_area(test_dir: Path, zone: Zone):
    dfs = []
    zone_dir = test_dir / f"z{zone.identifier}"
    out_dir = zone_dir / "results"
    out_dir.mkdir(exist_ok=True)
    paths = sorted(zone_dir.glob("*.png"))
    for path in paths:
        image = np.array(Image.open(path))
        debug_images = {}
        df = process_image(image, zone.trays, debug_images)
        df["intensity"] = path.stem

        avg = iqm(jnp.array(df["area"]), 0.05)
        df = pd.concat([df, pd.DataFrame({"plant_id": ["iqm"], "area": [avg], "intensity": [path.stem]})])
        dfs.append(df)
        df.to_csv(out_dir / f"{path.stem}.csv", index=False)
        for key, value in debug_images.items():
            value.save(out_dir / f"{path.stem}_{key}.png")

    df = pd.concat(dfs)
    plot_area_comparison(df, out_dir)


def test_process_old_zone_1():
    zone = Zone(
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
    get_plant_area(SC_TEST_DIR, zone)


def test_process_old_zone_6():
    zone = Zone(
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
    get_plant_area(SC_TEST_DIR, zone)


def test_process_old_zone_3():
    zone = Zone(
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
            ),
        ],
    )
    get_plant_area(OLD_TEST_DIR, zone)


def plot_area_comparison(df: pd.DataFrame, out_dir: Path):
    plt.figure(figsize=(4 * len(df) // 32, 3))
    sns.barplot(df, x="plant_id", y="area", hue="intensity")
    plt.ylim(0, df["area"].quantile(0.99))
    plt.savefig(out_dir / "areas.png")
