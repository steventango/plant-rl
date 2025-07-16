import json
import logging
import re
from pathlib import Path

from environments.PlantGrowthChamber.zones import (
    ZONE_IDENTIFIERS,
    Rect,
    Tray,
    Zone,
    load_zone_from_config,
    serialize_zone,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("tray_config.log")],
)
logger = logging.getLogger("TrayConfigApp")

project_root = Path(__file__).resolve().parents[3]


def extract_zone_identifier(dataset_path: Path) -> str | None:
    """Extract zone identifier from the dataset directory path."""
    zone_name = dataset_path.name
    logger.debug(f"Attempting to extract zone identifier from '{zone_name}'")

    # Check for full match first
    for identifier in ZONE_IDENTIFIERS:
        if identifier == zone_name:
            logger.debug(f"Found exact match for zone identifier: '{identifier}'")
            return identifier

    # Check for partial match
    for identifier in ZONE_IDENTIFIERS:
        if identifier in zone_name:
            logger.debug(f"Found partial match for zone identifier: '{identifier}'")
            return identifier

    # Fallback for old naming convention like 'z01', 'z12'
    match = re.search(r"z(\d+)", zone_name)
    if match:
        zone_num = int(match.group(1))
        identifier = f"alliance-zone{zone_num:02d}"
        if identifier in ZONE_IDENTIFIERS:
            logger.debug(
                f"Found legacy zone identifier 'z{zone_num}' and mapped to '{identifier}'"
            )
            return identifier

    logger.warning(f"Could not extract zone identifier from {zone_name}.")
    return None


def find_all_datasets(base_dirs):
    """Find all datasets in the base directories"""
    dataset_dirs = []

    for base_dir in base_dirs:
        if base_dir.exists():
            for dir_path in sorted(base_dir.glob("*")):
                if (dir_path / "images").exists():
                    dataset_dirs.append(dir_path)

    logger.info(f"Found {len(dataset_dirs)} datasets")
    for idx, dataset in enumerate(dataset_dirs):
        logger.info(f"{idx + 1}. {dataset}")

    return dataset_dirs


def load_existing_config(dataset_dir, zone_identifier, update_config=True):
    """Load existing configuration if available."""
    if zone_identifier is None:
        return []

    if update_config:
        try:
            zone = load_zone_from_config(zone_identifier)
            tray_configs = [
                {
                    "n_tall": tray.n_tall,
                    "n_wide": tray.n_wide,
                    "rect": {
                        "top_left": tray.rect.top_left,
                        "top_right": tray.rect.top_right,
                        "bottom_left": tray.rect.bottom_left,
                        "bottom_right": tray.rect.bottom_right,
                    },
                }
                for tray in zone.trays
            ]
            logger.debug(f"Loaded {len(tray_configs)} trays for zone {zone_identifier}")
            return tray_configs
        except FileNotFoundError:
            logger.warning(
                f"No config file found for zone {zone_identifier}. Starting fresh."
            )
            return []
        except Exception as e:
            logger.error(f"Error loading config for zone {zone_identifier}: {e}")
            return []
    else:
        config_path = dataset_dir / "config.json"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    existing_config = json.load(f)

                if "zone" in existing_config and "trays" in existing_config["zone"]:
                    tray_configs = existing_config["zone"]["trays"]
                    logger.debug(f"Loaded {len(tray_configs)} trays from {config_path}")
                    return tray_configs
                else:
                    logger.warning(f"No trays found in existing config: {config_path}")
                    return []
            except Exception as e:
                logger.error(f"Error loading config file {config_path}: {e}")
                return []
        else:
            return []


def save_config(dataset_dir, zone_identifier, tray_configs, update_config=True):
    """Save tray configuration to file."""
    if zone_identifier is None:
        logger.error("Cannot save, zone identifier is not set.")
        return False

    if update_config:
        try:
            try:
                zone = load_zone_from_config(zone_identifier)
            except FileNotFoundError:
                logger.warning(
                    f"Config for {zone_identifier} not found. Creating a new one."
                )
                zone = Zone(
                    identifier=zone_identifier,
                    camera_left_url=None,
                    camera_right_url=None,
                    lightbar_url=None,
                    calibration=None,
                    trays=[],
                )

            # Convert tray_configs (list of dicts) to list of Tray objects
            zone.trays = [
                Tray(
                    n_tall=tc["n_tall"],
                    n_wide=tc["n_wide"],
                    rect=Rect(
                        top_left=tuple(tc["rect"]["top_left"]),
                        top_right=tuple(tc["rect"]["top_right"]),
                        bottom_left=tuple(tc["rect"]["bottom_left"]),
                        bottom_right=tuple(tc["rect"]["bottom_right"]),
                    ),
                )
                for tc in tray_configs
            ]

            # Serialize and save
            config_data = serialize_zone(zone)
            config_path = (
                project_root
                / "src"
                / "environments"
                / "PlantGrowthChamber"
                / "configs"
                / f"{zone_identifier}.json"
            )
            with open(config_path, "w") as f:
                json.dump({"zone": config_data}, f, indent=4)

            logger.debug(
                f"Successfully saved config for zone {zone_identifier} to {config_path}"
            )
            return True

        except Exception as e:
            logger.error(f"Error saving config for zone {zone_identifier}: {e}")
            return False
    else:
        # For local config, just save the trays and identifier
        final_config = {"zone": {"identifier": zone_identifier, "trays": tray_configs}}
        config_path = dataset_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(final_config, f, indent=4)
        logger.debug(f"Successfully saved configuration to {config_path}")
        return True
