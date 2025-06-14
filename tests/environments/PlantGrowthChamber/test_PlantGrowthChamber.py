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
    # "alliance-zone05",
    # "alliance-zone06",
    # "alliance-zone07",
    # "alliance-zone08",
    # "alliance-zone09",
    # "alliance-zone10",
    # "alliance-zone11",
    # "alliance-zone12",
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

    # Capture an image after turning off lights
    print(f"Zone ID {zone_id}: Waiting {sleep_duration} seconds for camera after turning off lights...")
    await asyncio.sleep(sleep_duration)
    print(f"Zone ID {zone_id}: Attempting observation after turning off lights.")
    try:
        _, obs_image_off, _ = await chamber.get_observation()
        print(f"Zone ID {zone_id}: Observation attempt after lights off complete.")

        if obs_image_off is not None:
            zone_suffix = zone_id.split("-")[-1]
            zone_numeric_part = zone_suffix.replace("zone", "")
            img_filename_off = f"zone{zone_numeric_part}_cam1_channel_off.jpg"
            img_path_off = output_dir / img_filename_off
            try:
                obs_image_off = Image.fromarray(obs_image_off)
                obs_image_off.save(img_path_off)
                print(f"Zone ID {zone_id}: Saved image after lights off to {img_path_off}")
            except Exception as e_save:
                print(f"Zone ID {zone_id}: Failed to save image {img_path_off} after lights off: {e_save}")
        else:
            print(f"Zone ID {zone_id}: No image data received from observation after lights off.")
    except Exception as e:
        print(f"Zone ID {zone_id}: Failed to get observation after lights off: {e}")

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
                filename_stem = item.stem  # e.g., "zone01_cam1_channel3" or "zone01_cam1_channel_off"
                parts = filename_stem.split("_")

                if len(parts) < 3:  # Basic check for "zone_cam_channel" structure
                    print(f"Skipping file with unexpected name format (too few parts): {item.name}")
                    continue

                zone_str = parts[0].replace("zone", "")
                zone = int(zone_str)
                cam_str = parts[1].replace("cam", "")
                cam = int(cam_str)

                channel_key_str_part = parts[2]
                if channel_key_str_part == "channel" and len(parts) > 3 and parts[3] == "off":
                    channel_key = "off"
                elif (
                    channel_key_str_part.startswith("channel") and channel_key_str_part != "channel"
                ):  # Avoid "channel" itself if not followed by "off"
                    channel_key = int(channel_key_str_part.replace("channel", ""))
                else:
                    print(f"Skipping file with unexpected channel format: {item.name}")
                    continue

                if zone not in images_data:
                    images_data[zone] = {}
                if cam not in images_data[zone]:
                    images_data[zone][cam] = {}
                images_data[zone][cam][channel_key] = item
            except (IndexError, ValueError) as e:
                print(f"Skipping file with unexpected name format or value error: {item.name} ({e})")
                continue

    if not images_data:
        print(
            f"No images found or parsed in '{image_dir_path}'. Ensure filenames are like 'zoneZ_camC_channelCH.jpg' or 'zoneZ_camC_channel_off.jpg'."
        )
        return

    sorted_zones = sorted(images_data.keys())
    num_rows = len(sorted_zones)
    if num_rows == 0:
        print("No zone data to plot.")
        return

    # Determine if cam1 or cam2 have any data across all zones
    has_cam1_overall = any(1 in images_data.get(z, {}) and images_data[z][1] for z in sorted_zones)
    has_cam2_overall = any(2 in images_data.get(z, {}) and images_data[z][2] for z in sorted_zones)

    # num_channels_for_plot are the explicit numbered channels (e.g., 1-6)
    # We add one more column for the "off" state image.
    num_display_cols_per_cam = num_channels_for_plot + 1

    max_cols = 0
    if has_cam1_overall:
        max_cols += num_display_cols_per_cam
    if has_cam2_overall:
        max_cols += num_display_cols_per_cam

    if max_cols == 0:
        print(
            f"No valid camera (1 or 2) image data found to plot in '{image_dir_path}'. Check image data and filenames."
        )
        return

    fig, axes = plt.subplots(num_rows, max_cols, figsize=(max_cols * 2.5, num_rows * 2.5), squeeze=False)
    fig.suptitle(f"Growth Chamber Test Results from: {image_dir_path.name}", fontsize=16)

    channels_to_render_keys = list(range(1, num_channels_for_plot + 1)) + ["off"]

    for i, zone_key in enumerate(sorted_zones):
        axes[i, 0].set_ylabel(f"Zone {zone_key}", rotation=0, size="large", labelpad=30)
        col_idx = 0

        for cam_num_plot in [1, 2]:
            # Determine if this camera column block should be rendered for this row
            # It should be rendered if the camera has data overall, to maintain consistent column structure.
            should_render_cam_block = (cam_num_plot == 1 and has_cam1_overall) or (
                cam_num_plot == 2 and has_cam2_overall
            )

            if not should_render_cam_block:
                continue

            for ch_key in channels_to_render_keys:
                if col_idx >= max_cols:  # Should not happen if max_cols is calculated correctly
                    print(
                        f"Warning: col_idx {col_idx} exceeded max_cols {max_cols}. Skipping plot for Zone {zone_key}, Cam {cam_num_plot}, Ch {ch_key}"
                    )
                    break

                title = f"C{cam_num_plot} Ch{ch_key}" if isinstance(ch_key, int) else f"C{cam_num_plot} Off"
                img_path_to_load_str = ""  # For error messages

                # Check if image exists for this specific zone, cam, and channel_key
                if cam_num_plot in images_data.get(zone_key, {}) and ch_key in images_data[zone_key].get(
                    cam_num_plot, {}
                ):
                    try:
                        img_path_to_load = images_data[zone_key][cam_num_plot][ch_key]
                        img_path_to_load_str = str(img_path_to_load)
                        img = Image.open(img_path_to_load)
                        axes[i, col_idx].imshow(img)
                        axes[i, col_idx].set_title(title)
                    except FileNotFoundError:
                        axes[i, col_idx].text(0.5, 0.5, "No Img", ha="center", va="center")
                        axes[i, col_idx].set_title(f"{title} (exp)")
                    except Exception as e_load:
                        axes[i, col_idx].text(0.5, 0.5, "Error", ha="center", va="center")
                        axes[i, col_idx].set_title(f"{title} (err)")
                        print(f"Error loading {img_path_to_load_str or 'path_not_set'}: {e_load}")
                else:  # Image missing for this specific slot (zone/cam/channel)
                    axes[i, col_idx].text(0.5, 0.5, "N/A", ha="center", va="center")
                    axes[i, col_idx].set_title(f"{title} (miss)")

                axes[i, col_idx].set_xticks([])
                axes[i, col_idx].set_yticks([])
                col_idx += 1

        # Fill any remaining columns in the row with blank axes if one camera is missing entirely
        # but space was allocated in max_cols.
        while col_idx < max_cols:
            axes[i, col_idx].axis("off")
            col_idx += 1

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))

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
