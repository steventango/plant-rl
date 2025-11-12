"""
Web-based Tray Configuration Tool using Flask
Replaces the tkinter-based app with a browser-based interface
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image
from tray_config_utils import (
    extract_zone_identifier,
    find_all_datasets,
    load_existing_config,
    save_config,
)


# Camera calibration parameters (same as in cv.py)
CAMERA_MATRIX = np.array([[1800.0, 0.0, 1296.0], [0.0, 1800.0, 972.0], [0.0, 0.0, 1.0]])
DIST_COEFFS = np.array([0.0, 0.0, 0.0, 0.0])

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("tray_config.log")],
)
logger = logging.getLogger("TrayConfigWebApp")

app = Flask(__name__)
app.config["SECRET_KEY"] = "tray-config-secret-key"

# Global state
state = {
    "dataset_dirs": [],
    "current_dataset_idx": -1,
    "dataset_dir": None,
    "zone_identifier": None,
    "tray_configs": [],
    "update_config": True,
    "default_n_tall": 3,
    "default_n_wide": 6,
}


def initialize_datasets(base_dirs, update_config=True):
    """Initialize the dataset list"""
    state["update_config"] = update_config
    state["dataset_dirs"] = find_all_datasets(base_dirs)
    state["current_dataset_idx"] = -1
    logger.info(f"Initialized with {len(state['dataset_dirs'])} datasets")


def load_next_dataset():
    """Load the next dataset to be processed"""
    if state["current_dataset_idx"] >= len(state["dataset_dirs"]) - 1:
        return None  # All datasets processed

    state["current_dataset_idx"] += 1
    state["dataset_dir"] = state["dataset_dirs"][state["current_dataset_idx"]]

    # Extract zone identifier
    state["zone_identifier"] = extract_zone_identifier(state["dataset_dir"])

    if state["zone_identifier"] is None:
        logger.warning(
            f"Could not determine zone for {state['dataset_dir'].name}. Skipping."
        )
        return load_next_dataset()  # Try next dataset

    # Load existing tray configurations
    state["tray_configs"] = load_existing_config(
        state["dataset_dir"], state["zone_identifier"], state["update_config"]
    )

    return state["dataset_dir"]


def get_image_path(dataset_dir):
    """Get the path to the image for display - exclude nighttime images (20:30-9:30 America/Edmonton)"""
    if dataset_dir is None:
        return None

    image_dir = dataset_dir / "images"
    if not image_dir.exists():
        return None

    image_files = [
        f
        for f in sorted(image_dir.glob("*"), reverse=True)
        if f.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]

    edmonton_tz = ZoneInfo("America/Edmonton")

    for image_file in image_files:
        try:
            # Try to extract datetime from filename
            # Common formats: YYYY-MM-DD_HH-MM-SS, YYYYMMDD_HHMMSS, etc.
            name = image_file.stem

            # Try different datetime patterns
            patterns = [
                r"(\d{4})-?(\d{2})-?(\d{2})[_T-](\d{2})[:-]?(\d{2})[:-]?(\d{2})",  # ISO-like
                r"(\d{4})(\d{2})(\d{2})[_-](\d{2})(\d{2})(\d{2})",  # Compact
            ]

            dt = None
            for pattern in patterns:
                match = re.search(pattern, name)
                if match:
                    year, month, day, hour, minute, second = map(int, match.groups())
                    # Create datetime in UTC first, then localize to Edmonton
                    dt = datetime(year, month, day, hour, minute, second)
                    # Assume the timestamp in filename is already in Edmonton time
                    dt = dt.replace(tzinfo=edmonton_tz)
                    break

            if dt:
                # Get the hour in Edmonton time
                hour = dt.hour
                minute = dt.minute
                time_decimal = hour + minute / 60.0

                # Exclude nighttime: 20:30 (20.5 hours) to 9:30 (9.5 hours)
                # Night is from 20:30 PM to 9:30 AM next day
                if time_decimal >= 20.5 or time_decimal < 9.5:
                    logger.debug(
                        f"Skipping nighttime image: {image_file.name} (time: {hour:02d}:{minute:02d})"
                    )
                    continue

            # If we can't parse time or it's daytime, use this image
            logger.info(f"Selected image: {image_file.name}")
            return image_file

        except Exception as e:
            logger.warning(f"Error parsing time from {image_file.name}: {e}")
            # If parsing fails, use the image anyway
            return image_file

    # Fallback: return last image if no daytime image found
    if image_files:
        logger.warning("No daytime images found, using most recent image")
        return image_files[0]
    return None


def get_image_dimensions(image_path):
    """Get undistorted image dimensions"""
    try:
        # Load and undistort image to get correct dimensions
        image = cv2.imread(str(image_path))
        if image is None:
            return None

        undistorted_image = cv2.undistort(image, CAMERA_MATRIX, DIST_COEFFS)
        height, width = undistorted_image.shape[:2]
        return (width, height)  # (width, height)
    except Exception as e:
        logger.error(f"Error getting image dimensions: {e}")
        return None


@app.route("/")
def index():
    """Serve the main page"""
    return render_template("index.html")


@app.route("/api/init", methods=["GET"])
def api_init():
    """Initialize and load the first dataset"""
    dataset_dir = load_next_dataset()

    if dataset_dir is None:
        return jsonify({"status": "complete", "message": "All datasets processed"})

    image_path = get_image_path(dataset_dir)
    if image_path is None:
        return jsonify({"status": "error", "message": "No image found"}), 404

    dimensions = get_image_dimensions(image_path)
    if dimensions is None:
        return jsonify({"status": "error", "message": "Could not load image"}), 500

    return jsonify(
        {
            "status": "ok",
            "dataset_name": dataset_dir.name,
            "dataset_index": state["current_dataset_idx"],
            "total_datasets": len(state["dataset_dirs"]),
            "zone_identifier": state["zone_identifier"],
            "tray_configs": state["tray_configs"],
            "image_width": dimensions[0],
            "image_height": dimensions[1],
            "default_n_tall": state["default_n_tall"],
            "default_n_wide": state["default_n_wide"],
        }
    )


@app.route("/api/image")
def api_image():
    """Serve the current undistorted image"""
    if state["dataset_dir"] is None:
        return jsonify({"error": "No dataset loaded"}), 404

    image_path = get_image_path(state["dataset_dir"])
    if image_path is None:
        return jsonify({"error": "No image found"}), 404

    try:
        # Load image with OpenCV for undistortion
        image = cv2.imread(str(image_path))
        if image is None:
            return jsonify({"error": "Could not load image"}), 500

        # Undistort the image using camera calibration parameters
        undistorted_image = cv2.undistort(image, CAMERA_MATRIX, DIST_COEFFS)

        # Convert back to PIL Image for serving
        pil_image = Image.fromarray(cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2RGB))

        # Save to BytesIO for serving
        from io import BytesIO

        img_io = BytesIO()
        pil_image.save(img_io, "JPEG", quality=95)
        img_io.seek(0)

        return send_file(img_io, mimetype="image/jpeg")

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return jsonify({"error": f"Image processing failed: {str(e)}"}), 500


@app.route("/api/add_tray", methods=["POST"])
def api_add_tray():
    """Add a new tray configuration"""
    data = request.get_json()

    try:
        n_tall = int(data["n_tall"])
        n_wide = int(data["n_wide"])
        points = data["points"]  # List of 4 points [x, y]

        if len(points) != 4:
            return jsonify({"error": "Exactly 4 points required"}), 400

        if n_tall < 1 or n_wide < 1:
            return jsonify({"error": "Rows and columns must be positive"}), 400

        tray_config = {
            "n_tall": n_tall,
            "n_wide": n_wide,
            "rect": {
                "top_left": points[0],
                "top_right": points[1],
                "bottom_left": points[2],
                "bottom_right": points[3],
            },
        }

        state["tray_configs"].append(tray_config)

        logger.info(
            f"Added tray {len(state['tray_configs'])} with dimensions {n_tall}×{n_wide}"
        )

        return jsonify(
            {
                "status": "ok",
                "tray_index": len(state["tray_configs"]) - 1,
                "total_trays": len(state["tray_configs"]),
            }
        )

    except Exception as e:
        logger.error(f"Error adding tray: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/update_tray", methods=["POST"])
def api_update_tray():
    """Update an existing tray configuration"""
    data = request.get_json()

    try:
        tray_index = int(data["tray_index"])
        n_tall = int(data["n_tall"])
        n_wide = int(data["n_wide"])
        points = data["points"]  # List of 4 points [x, y]

        if len(points) != 4:
            return jsonify({"error": "Exactly 4 points required"}), 400

        if n_tall < 1 or n_wide < 1:
            return jsonify({"error": "Rows and columns must be positive"}), 400

        if tray_index < 0 or tray_index >= len(state["tray_configs"]):
            return jsonify({"error": "Invalid tray index"}), 400

        tray_config = {
            "n_tall": n_tall,
            "n_wide": n_wide,
            "rect": {
                "top_left": points[0],
                "top_right": points[1],
                "bottom_left": points[2],
                "bottom_right": points[3],
            },
        }

        state["tray_configs"][tray_index] = tray_config

        logger.info(f"Updated tray {tray_index + 1} with dimensions {n_tall}×{n_wide}")

        return jsonify({"status": "ok", "tray_index": tray_index})

    except Exception as e:
        logger.error(f"Error updating tray: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/delete_tray", methods=["POST"])
def api_delete_tray():
    """Delete a tray configuration"""
    data = request.get_json()

    try:
        tray_index = int(data["tray_index"])

        if tray_index < 0 or tray_index >= len(state["tray_configs"]):
            return jsonify({"error": "Invalid tray index"}), 400

        state["tray_configs"].pop(tray_index)

        logger.info(f"Deleted tray {tray_index + 1}")

        return jsonify({"status": "ok", "total_trays": len(state["tray_configs"])})

    except Exception as e:
        logger.error(f"Error deleting tray: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/save_and_next", methods=["POST"])
def api_save_and_next():
    """Save current configuration and load next dataset"""
    if state["dataset_dir"] is None:
        return jsonify({"error": "No dataset loaded"}), 400

    if state["zone_identifier"] is None:
        return jsonify({"error": "Zone identifier not set"}), 400

    # Save configuration
    success = save_config(
        state["dataset_dir"],
        state["zone_identifier"],
        state["tray_configs"],
        state["update_config"],
    )

    if not success:
        return (
            jsonify({"error": f"Could not save config for {state['zone_identifier']}"}),
            500,
        )

    # Load next dataset
    dataset_dir = load_next_dataset()

    if dataset_dir is None:
        return jsonify({"status": "complete", "message": "All datasets processed"})

    image_path = get_image_path(dataset_dir)
    if image_path is None:
        return jsonify({"status": "error", "message": "No image found"}), 404

    dimensions = get_image_dimensions(image_path)
    if dimensions is None:
        return jsonify({"status": "error", "message": "Could not load image"}), 500

    return jsonify(
        {
            "status": "ok",
            "dataset_name": dataset_dir.name,
            "dataset_index": state["current_dataset_idx"],
            "total_datasets": len(state["dataset_dirs"]),
            "zone_identifier": state["zone_identifier"],
            "tray_configs": state["tray_configs"],
            "image_width": dimensions[0],
            "image_height": dimensions[1],
            "default_n_tall": state["default_n_tall"],
            "default_n_wide": state["default_n_wide"],
        }
    )


@app.route("/api/save_and_previous", methods=["POST"])
def api_save_and_previous():
    """Save current configuration and load previous dataset"""
    if state["dataset_dir"] is None:
        return jsonify({"error": "No dataset loaded"}), 400

    if state["zone_identifier"] is None:
        return jsonify({"error": "Zone identifier not set"}), 400

    # Check if we can go back
    if state["current_dataset_idx"] <= 0:
        return jsonify({"status": "error", "message": "Already at first dataset"}), 400

    # Save configuration
    success = save_config(
        state["dataset_dir"],
        state["zone_identifier"],
        state["tray_configs"],
        state["update_config"],
    )

    if not success:
        return (
            jsonify({"error": f"Could not save config for {state['zone_identifier']}"}),
            500,
        )

    # Go to previous dataset
    state["current_dataset_idx"] -= (
        2  # Subtract 2 because load_next_dataset will increment by 1
    )
    dataset_dir = load_next_dataset()

    if dataset_dir is None:
        return jsonify(
            {"status": "error", "message": "Could not load previous dataset"}
        ), 500

    image_path = get_image_path(dataset_dir)
    if image_path is None:
        return jsonify({"status": "error", "message": "No image found"}), 404

    dimensions = get_image_dimensions(image_path)
    if dimensions is None:
        return jsonify({"status": "error", "message": "Could not load image"}), 500

    return jsonify(
        {
            "status": "ok",
            "dataset_name": dataset_dir.name,
            "dataset_index": state["current_dataset_idx"],
            "total_datasets": len(state["dataset_dirs"]),
            "zone_identifier": state["zone_identifier"],
            "tray_configs": state["tray_configs"],
            "image_width": dimensions[0],
            "image_height": dimensions[1],
            "default_n_tall": state["default_n_tall"],
            "default_n_wide": state["default_n_wide"],
        }
    )


@app.route("/api/status", methods=["GET"])
def api_status():
    """Get current status"""
    return jsonify(
        {
            "dataset_name": (
                state["dataset_dir"].name if state["dataset_dir"] else None
            ),
            "dataset_index": state["current_dataset_idx"],
            "total_datasets": len(state["dataset_dirs"]),
            "zone_identifier": state["zone_identifier"],
            "tray_count": len(state["tray_configs"]),
        }
    )


@app.route("/api/current", methods=["GET"])
def api_current():
    """Get current dataset without advancing"""
    if state["dataset_dir"] is None:
        # If no dataset loaded, load the first one
        dataset_dir = load_next_dataset()
        if dataset_dir is None:
            return jsonify({"status": "complete", "message": "All datasets processed"})

    image_path = get_image_path(state["dataset_dir"])
    if image_path is None:
        return jsonify({"status": "error", "message": "No image found"}), 404

    dimensions = get_image_dimensions(image_path)
    if dimensions is None:
        return jsonify({"status": "error", "message": "Could not load image"}), 500

    return jsonify(
        {
            "status": "ok",
            "dataset_name": state["dataset_dir"].name,
            "dataset_index": state["current_dataset_idx"],
            "total_datasets": len(state["dataset_dirs"]),
            "zone_identifier": state["zone_identifier"],
            "tray_configs": state["tray_configs"],
            "image_width": dimensions[0],
            "image_height": dimensions[1],
            "default_n_tall": state["default_n_tall"],
            "default_n_wide": state["default_n_wide"],
        }
    )


def run_app(base_dirs, update_config=True, host="0.0.0.0", port=5000, debug=False):
    """Run the Flask application"""
    initialize_datasets(base_dirs, update_config)
    logger.info(f"Starting web server on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    # This won't be used directly, but keeping for testing
    from itertools import chain

    BASE_DIRS = chain(
        Path("/data/online/E14/P0").rglob("*"),
    )
    run_app(BASE_DIRS, update_config=True, debug=True)
