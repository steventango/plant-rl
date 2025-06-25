import os
from itertools import chain
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sns
from PIL import Image

from environments.PlantGrowthChamber.cv import process_image
from environments.PlantGrowthChamber.zones import (
    Rect,
    Tray,
    Zone,
    load_zone_from_config,
)
from utils.metrics import iqm

TEST_DIR = Path(__file__).parent.parent.parent / "test_data"
OLD_TEST_DIR = TEST_DIR / "old"
E4_TEST_DIR = TEST_DIR / "Spreadsheet-C"
E5_TEST_DIR = TEST_DIR / "Spreadsheet-C-v2"
E6_TEST_DIR = TEST_DIR / "Spreadsheet-C-v3"
E7_TEST_DIR = TEST_DIR / "E7/P2"
E8_TEST_DIR = TEST_DIR / "E8"

skipif_github_actions = pytest.mark.skipif(
    os.environ.get("GITHUB_ACTIONS") == "true",
    reason="Skip in GitHub Actions environment",
)


def get_plant_area(test_dir: Path, zone: Zone):
    dfs = []
    zone_dir = test_dir / f"z{zone.identifier}"
    out_dir = zone_dir / "results"
    out_dir.mkdir(exist_ok=True)
    paths = sorted(chain(zone_dir.glob("*.png"), zone_dir.glob("*.jpg")))
    for path in paths:
        image = np.array(Image.open(path))
        debug_images = {}
        df, _ = process_image(image, zone.trays, debug_images)
        df["intensity"] = path.stem

        avg = iqm(jnp.array(df["area"]), 0.05)
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {"plant_id": ["iqm"], "area": [avg], "intensity": [path.stem]}
                ),
            ]
        )
        df = df.reset_index(drop=True)
        dfs.append(df)
        df.to_csv(out_dir / f"{path.stem}.csv", index=False)
        for key, value in debug_images.items():
            value.save(out_dir / f"{path.stem}_{key}.jpg")

    df = pd.concat(dfs)
    plot_area_comparison(df, out_dir)


@skipif_github_actions
def test_process_E8_zone_1():
    zone = load_zone_from_config("mitacs-zone01")
    get_plant_area(E8_TEST_DIR, zone)


@skipif_github_actions
def test_process_E8_zone_2():
    zone = load_zone_from_config("mitacs-zone02")
    get_plant_area(E8_TEST_DIR, zone)


@skipif_github_actions
def test_process_E8_zone_3():
    zone = load_zone_from_config("mitacs-zone03")
    get_plant_area(E8_TEST_DIR, zone)


@skipif_github_actions
def test_process_E8_zone_6():
    zone = load_zone_from_config("mitacs-zone06")
    get_plant_area(E8_TEST_DIR, zone)


@skipif_github_actions
def test_process_E8_zone_8():
    zone = load_zone_from_config("mitacs-zone08")
    get_plant_area(E8_TEST_DIR, zone)


@skipif_github_actions
def test_process_E8_zone_9():
    zone = load_zone_from_config("mitacs-zone09")
    get_plant_area(E8_TEST_DIR, zone)


@skipif_github_actions
def test_process_E5_zone_1():
    zone = Zone(
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
                    bottom_left=(513, 1512),
                    bottom_right=(1731, 1626),
                ),
            ),
        ],
    )
    get_plant_area(E5_TEST_DIR, zone)


@skipif_github_actions
def test_process_E5_zone_2():
    zone = Zone(
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
                    bottom_left=(585, 1722),
                    bottom_right=(1812, 1572),
                ),
            ),
        ],
    )
    get_plant_area(E5_TEST_DIR, zone)


@skipif_github_actions
def test_process_E4_zone_1():
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
    get_plant_area(E4_TEST_DIR, zone)


@skipif_github_actions
def test_process_E4_zone_6():
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
    get_plant_area(E4_TEST_DIR, zone)


@skipif_github_actions
def test_process_E4_zone_3():
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
