import asyncio
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from src.environments.PlantGrowthChamber.PlantGrowthChamber import PlantGrowthChamber

# from src.environments.PlantGrowthChamber.zones import ZONE_IDENTIFIERS


ZONE_IDENTIFIERS = [
    # "alliance-zone01",
    "alliance-zone02",
    # "alliance-zone03",
    # "alliance-zone04",
    "alliance-zone05",
    "alliance-zone06",
    "alliance-zone07",
    "alliance-zone08",
    "alliance-zone09",
    "alliance-zone10",
    "alliance-zone11",
    "alliance-zone12",
]
OUTPUT_DIR = Path(__file__).parent.parent.parent / "test_data" / "results"
LIGHT_INTENSITY = 1
NUM_CHANNELS = 6
SLEEP_DURATION = 5

skipif_github_actions = pytest.mark.skipif(
    os.environ.get("GITHUB_ACTIONS") == "true", reason="Skip in GitHub Actions environment"
)


async def _run_single_zone_test(
    zone_id: str, output_dir: Path, num_channels: int, light_intensity: float, sleep_duration: float
):
    """Internal helper to test a single zone asynchronously, identified by its full zone_id string."""
    print(f"Starting test for Zone ID {zone_id}")

    chamber = PlantGrowthChamber(zone=zone_id, timezone="America/Edmonton")

    # Generate all actions: one-hot for each channel, then all-off
    actions = []
    for ch_idx in range(num_channels):
        arr_vals = [light_intensity if i == ch_idx else 0.0 for i in range(num_channels)]
        actions.append(arr_vals)
    # Add the off action (all zeros)
    off_payload = [0.0] * num_channels
    actions.append(off_payload)

    for arr_vals in actions:
        action_str = str(arr_vals).replace(" ", "")
        print(f"Zone ID {zone_id}: Testing Action {action_str}")

        light_payload = np.array(arr_vals, dtype=float)
        try:
            await chamber.put_action(light_payload)
        except Exception as e:
            print(f"Zone ID {zone_id}: Error setting lightbar via PlantGrowthChamber: {e}")
            continue

        print(f"Zone ID {zone_id}: Waiting {sleep_duration} seconds for lights and camera...")
        await asyncio.sleep(sleep_duration)

        print(f"Zone ID {zone_id}: Attempting observation for action {action_str}")
        try:
            chamber.images = {}
            _, obs_image, _ = await chamber.get_observation()
            print(f"Zone ID {zone_id}: Observation attempt complete for action {action_str}.")

            if obs_image is not None:
                img_filename = f"{zone_id}_{action_str}.jpg"
                img_path = output_dir / img_filename

                try:
                    obs_image = Image.fromarray(obs_image)
                    obs_image.save(img_path)
                    print(f"Zone ID {zone_id}: Saved image for action {action_str} to {img_path}")
                except Exception as e_save:
                    print(f"Zone ID {zone_id}: Failed to save image {img_path}: {e_save}")
            else:
                print(f"Zone ID {zone_id}: No image data received from observation for action {action_str}.")
        except Exception as e:
            print(f"Zone ID {zone_id}: Failed to get observation for action {action_str}: {e}")

    await chamber.close()  # Close the chamber resources
    print(f"Finished test for Zone ID {zone_id}")


def plot_images(image_dir: Path, show_plot_flag: bool):
    """Plots images from the specified directory into a grid."""
    image_dir_path = Path(image_dir)
    if not image_dir_path.is_dir():
        print(f"Error: Directory '{image_dir_path}' not found.")
        return

    images_data = {}
    action_set = set()  # Keep track of all unique actions

    for item in image_dir_path.iterdir():
        if item.is_file() and item.name.lower().endswith((".png", ".jpg", ".jpeg")):
            if item.name == "growth_chamber_test_summary.png":
                continue
            try:
                # Split filename into zone_id and action
                zone_id, action_str = item.stem.split("_", 1)

                if zone_id not in images_data:
                    images_data[zone_id] = {}
                images_data[zone_id][action_str] = item
                action_set.add(action_str)  # Add to set of unique actions
            except Exception as e:
                print(f"Skipping file with unexpected name format: {item.name} ({e})")
                continue

    if not images_data:
        print(f"No images found or parsed in '{image_dir_path}'.")
        return

    sorted_zones = sorted(images_data.keys())
    num_rows = len(sorted_zones)

    action_strs = sorted(action_set, reverse=True)
    num_cols = len(action_strs)

    # Dynamically adjust figure size and font sizes
    cell_width = 5
    cell_height = 2.5
    figsize = (num_cols * cell_width, num_rows * cell_height)

    suptitle_fontsize = 24
    ylabel_fontsize = 24
    title_fontsize = 24

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, squeeze=False)
    fig.suptitle(f"Growth Chamber Test Results", fontsize=suptitle_fontsize)

    for i, zone_id in enumerate(sorted_zones):
        axes[i, 0].set_ylabel(f"{zone_id}", rotation=0, size=ylabel_fontsize, labelpad=100)

        for col_idx, action_str in enumerate(action_strs):
            if action_str in images_data.get(zone_id, {}):
                img_path = images_data[zone_id][action_str]
                try:
                    img = Image.open(img_path)
                    axes[i, col_idx].imshow(img)
                    if i == 0:
                        axes[i, col_idx].set_title(action_str, fontsize=title_fontsize)
                except Exception as e:
                    axes[i, col_idx].text(0.5, 0.5, "Error", ha="center", va="center")
                    axes[i, col_idx].set_title("(err)", fontsize=title_fontsize)
                    print(f"Error loading {img_path}: {e}")
            else:
                axes[i, col_idx].text(0.5, 0.5, "N/A", ha="center", va="center")
                axes[i, col_idx].set_title("(miss)", fontsize=title_fontsize)

            axes[i, col_idx].set_xticks([])
            axes[i, col_idx].set_yticks([])

    # Adjust layout to prevent overlap
    plt.tight_layout()

    output_plot_filename = image_dir_path / "growth_chamber_test_summary.png"
    try:
        plt.savefig(output_plot_filename)
        print(f"Plot saved to {output_plot_filename}")
        if show_plot_flag:
            plt.show()
    except Exception as e:
        print(f"Error saving/showing plot: {e}")


@skipif_github_actions
@pytest.mark.asyncio
async def test_cycle_lights_and_observe_all_zones():
    """Pytest test function to run the chamber tests with default parameters."""
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    else:
        for item in OUTPUT_DIR.iterdir():
            if item.is_file() and item.name.lower().endswith((".png", ".jpg", ".jpeg")):
                item.unlink()
    print(f"Pytest: Output directory set to {OUTPUT_DIR}")

    tasks = []

    if not ZONE_IDENTIFIERS:
        print("Pytest: No default zones to test. Skipping.")
        return

    for zone_id_val in ZONE_IDENTIFIERS:
        tasks.append(
            _run_single_zone_test(
            zone_id=zone_id_val,
            output_dir=OUTPUT_DIR,
            num_channels=NUM_CHANNELS,
            light_intensity=LIGHT_INTENSITY,
            sleep_duration=SLEEP_DURATION,
        )
        )
    await asyncio.gather(*tasks)
    print(f"Pytest: All zone tests completed. Images are in their respective chamber-defined locations.")

    print("Pytest: Generating plot...")
    plot_images(OUTPUT_DIR, False)
