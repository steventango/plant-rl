import asyncio
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from src.environments.PlantGrowthChamber.PlantGrowthChamber import PlantGrowthChamber

# from src.environments.PlantGrowthChamber.zones import ZONE_IDENTIFIERS


# ZONE_IDENTIFIERS = ZONE_IDENTIFIERS
ZONE_IDENTIFIERS = ["mitacs-zone01"]
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

    for ch_idx in range(num_channels):
        channel_idx_plus_1 = ch_idx + 1
        print(f"Zone ID {zone_id}: Testing Channel {channel_idx_plus_1}")

        arr_vals = [light_intensity if i == ch_idx else 0.0 for i in range(num_channels)]
        light_payload = np.array(arr_vals, dtype=float)

        try:
            await chamber.put_action(light_payload)
        except Exception as e:
            print(f"Zone ID {zone_id}: Error setting lightbar via PlantGrowthChamber: {e}")
            continue

        print(f"Zone ID {zone_id}: Waiting {sleep_duration} seconds for lights and camera...")
        await asyncio.sleep(sleep_duration)

        print(f"Zone ID {zone_id}: Attempting observation for Channel {channel_idx_plus_1}")
        try:
            _, obs_image, _ = await chamber.get_observation()
            print(f"Zone ID {zone_id} Channel {channel_idx_plus_1}: Observation attempt complete.")

            if obs_image is not None:
                # Extract numeric part of zone_id for filename, e.g., "01" from "mitacs-zone01"
                zone_suffix = zone_id.split("-")[-1]  # e.g., "zone01"
                zone_numeric_part = zone_suffix.replace("zone", "")  # e.g., "01"

                # Construct filename compatible with plot_images
                img_filename = f"zone{zone_numeric_part}_cam1_channel{channel_idx_plus_1}.jpg"
                img_path = output_dir / img_filename

                try:
                    obs_image = Image.fromarray(obs_image)
                    obs_image.save(img_path)
                    print(f"Zone ID {zone_id}: Saved image for channel {channel_idx_plus_1} to {img_path}")
                except Exception as e_save:
                    print(
                        f"Zone ID {zone_id}: Failed to save image {img_path} for channel {channel_idx_plus_1}: {e_save}"
                    )
            else:
                print(f"Zone ID {zone_id} Channel {channel_idx_plus_1}: No image data received from observation.")

        except Exception as e:
            print(f"Zone ID {zone_id} Channel {channel_idx_plus_1}: Failed to get observation: {e}")

    print(f"Zone ID {zone_id}: Turning off lights.")
    off_payload = np.zeros(num_channels)  # Light off payload for PlantGrowthChamber
    try:
        await chamber.put_action(off_payload)
    except Exception as e:
        print(f"Zone ID {zone_id}: Error turning off lights via PlantGrowthChamber: {e}")

    await chamber.close()  # Close the chamber resources
    print(f"Finished test for Zone ID {zone_id}")


def plot_images(image_dir: Path, num_channels_for_plot: int, show_plot_flag: bool):
    """
    Plots images from the specified directory into a grid.
    """
    image_dir_path = Path(image_dir)
    if not image_dir_path.is_dir():
        print(f"Error: Directory '{image_dir_path}' not found.")
        return

    images_data = {}
    for item in image_dir_path.iterdir():
        if item.is_file() and item.name.lower().endswith((".png", ".jpg", ".jpeg")):
            try:
                parts = item.name.split("_")
                zone_str = parts[0].replace("zone", "")
                zone = int(zone_str)
                cam_str = parts[1].replace("cam", "")
                cam = int(cam_str)
                channel_str = parts[2].replace("channel", "").split(".")[0]
                channel = int(channel_str)

                if zone not in images_data:
                    images_data[zone] = {}
                if cam not in images_data[zone]:
                    images_data[zone][cam] = {}
                images_data[zone][cam][channel] = item
            except (IndexError, ValueError) as e:
                print(f"Skipping file with unexpected name format: {item.name} ({e})")
                continue

    if not images_data:
        print(f"No images found or parsed in '{image_dir_path}'. Ensure filenames are like 'zoneZ_camC_channelCH.jpg'.")
        return

    sorted_zones = sorted(images_data.keys())
    num_rows = len(sorted_zones)
    if num_rows == 0:
        print("No zone data to plot.")
        return

    max_cols = 0
    for zone_key in sorted_zones:
        current_zone_cols = 0
        if 1 in images_data[zone_key] and isinstance(images_data[zone_key][1], dict):
            current_zone_cols += len(images_data[zone_key][1])
        if 2 in images_data[zone_key] and isinstance(images_data[zone_key][2], dict):
            current_zone_cols += len(images_data[zone_key][2])
        if current_zone_cols > max_cols:
            max_cols = current_zone_cols

    if max_cols == 0:
        max_possible_ch_per_cam = num_channels_for_plot
        has_cam1_data = any(1 in images_data[z] for z in sorted_zones)
        has_cam2_data = any(2 in images_data[z] for z in sorted_zones)
        if has_cam1_data and has_cam2_data:
            max_cols = 2 * max_possible_ch_per_cam
        elif has_cam1_data or has_cam2_data:
            max_cols = max_possible_ch_per_cam
        else:
            print("No channel images found for any camera in any zone.")
            return

    fig, axes = plt.subplots(num_rows, max_cols, figsize=(max_cols * 2.5, num_rows * 2.5), squeeze=False)
    fig.suptitle(f"Growth Chamber Test Results from: {image_dir_path.name}", fontsize=16)

    for i, zone_key in enumerate(sorted_zones):
        axes[i, 0].set_ylabel(f"Zone {zone_key}", rotation=0, size="large", labelpad=30)
        col_idx = 0

        for cam_num_plot in [1, 2]:  # Iterate through camera 1, then camera 2
            for ch in range(1, num_channels_for_plot + 1):
                if col_idx >= max_cols:
                    break
                title = f"C{cam_num_plot} Ch{ch}"
                img_path = ""  # Initialize img_path
                if cam_num_plot in images_data[zone_key] and ch in images_data[zone_key][cam_num_plot]:
                    try:
                        img_path = images_data[zone_key][cam_num_plot][ch]
                        img = Image.open(img_path)
                        axes[i, col_idx].imshow(img)
                        axes[i, col_idx].set_title(title)
                    except FileNotFoundError:
                        axes[i, col_idx].text(0.5, 0.5, "No Img", ha="center", va="center")
                        axes[i, col_idx].set_title(f"{title} (exp)")
                    except Exception as e:
                        axes[i, col_idx].text(0.5, 0.5, "Error", ha="center", va="center")
                        axes[i, col_idx].set_title(f"{title} (err)")
                        print(f"Error loading {img_path or 'path_not_set'}: {e}")
                else:
                    axes[i, col_idx].text(0.5, 0.5, "N/A", ha="center", va="center")
                    axes[i, col_idx].set_title(f"{title} (miss)")
                axes[i, col_idx].set_xticks([])
                axes[i, col_idx].set_yticks([])
                col_idx += 1
            if (
                col_idx >= max_cols and cam_num_plot == 1 and (2 in images_data[zone_key])
            ):  # if max_cols is based on only one cam
                pass  # continue to fill placeholders for the second camera if it was expected
            elif col_idx >= max_cols:  # if all expected columns are filled
                break

        while col_idx < max_cols:
            axes[i, col_idx].axis("off")
            col_idx += 1

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))  # Use tuple for rect

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
async def test_cycle_lights_and_observe_all_default_zones():
    """Pytest test function to run the chamber tests with default parameters."""
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
    plot_images(OUTPUT_DIR, NUM_CHANNELS, False)
