import json
import logging
import re
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from environments.PlantGrowthChamber.zones import (
    ZONE_IDENTIFIERS,
    Rect,
    Tray,
    Zone,
    load_zone_from_config,
    serialize_zone,
)

project_root = Path(__file__).resolve().parents[2]

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("tray_config.log")],
)
logger = logging.getLogger("TrayConfigApp")

# Flag to update existing config files instead of creating new ones
UPDATE_CONFIG = True  # When True, updates configs in PlantGrowthChamber directory

# Base directories containing the datasets
BASE_DIRS = Path("/data/online/E10/P0").rglob("*")


class TrayConfigApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tray Configuration Tool")

        # Initialize variables
        self.points = []
        self.tray_configs = []
        self.dataset_dirs = []
        self.current_dataset_idx = -1
        self.dataset_dir = None
        self.original_width = 0
        self.original_height = 0
        self.scale_factor = 1.0
        self.tk_image = None
        self.tray_markers = []  # Store tray visualization markers
        self.selected_tray_index = None  # Track which tray is being edited
        self.zone_identifier: str | None = None

        # Default tray dimensions
        self.default_n_tall = 3
        self.default_n_wide = 6

        # Find all dataset directories
        self.find_all_datasets()

        # Create main frame
        self.frame = tk.Frame(root)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Create a label to display current dataset
        self.dataset_label = tk.Label(self.frame, text="", font=("Arial", 12))
        self.dataset_label.pack(side=tk.TOP, fill=tk.X)

        # Add tray dimension inputs
        self.tray_dim_frame = tk.Frame(self.frame)
        self.tray_dim_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        tk.Label(self.tray_dim_frame, text="Tray Dimensions:").grid(
            row=0, column=0, sticky="w"
        )

        tk.Label(self.tray_dim_frame, text="Rows (n_tall):").grid(
            row=0, column=1, padx=5
        )
        self.n_tall_var = tk.StringVar(value=str(self.default_n_tall))
        self.n_tall_entry = ttk.Spinbox(
            self.tray_dim_frame, from_=1, to=20, width=5, textvariable=self.n_tall_var
        )
        self.n_tall_entry.grid(row=0, column=2)

        tk.Label(self.tray_dim_frame, text="Columns (n_wide):").grid(
            row=0, column=3, padx=5
        )
        self.n_wide_var = tk.StringVar(value=str(self.default_n_wide))
        self.n_wide_entry = ttk.Spinbox(
            self.tray_dim_frame, from_=1, to=20, width=5, textvariable=self.n_wide_var
        )
        self.n_wide_entry.grid(row=0, column=4)

        # Add tray counter and editing status
        self.tray_count_frame = tk.Frame(self.tray_dim_frame)
        self.tray_count_frame.grid(row=0, column=5, padx=(20, 5), sticky="e")

        self.tray_count_label = tk.Label(
            self.tray_count_frame, text="Trays: 0", font=("Arial", 10)
        )
        self.tray_count_label.pack(side=tk.LEFT)

        self.edit_status_label = tk.Label(
            self.tray_count_frame, text="", font=("Arial", 10, "italic"), fg="blue"
        )
        self.edit_status_label.pack(side=tk.RIGHT, padx=(10, 0))

        # Create canvas for the image
        self.canvas = tk.Canvas(self.frame)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Create button frame
        self.button_frame = tk.Frame(self.frame)
        self.button_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Add cancel edit button
        self.cancel_edit_button = tk.Button(
            self.button_frame,
            text="Cancel Edit",
            command=self.cancel_edit,
            state=tk.DISABLED,
        )
        self.cancel_edit_button.pack(side=tk.LEFT, padx=(10, 5), pady=10)

        # Add delete tray button
        self.delete_tray_button = tk.Button(
            self.button_frame,
            text="Delete Tray",
            command=self.delete_selected_tray,
            state=tk.DISABLED,
        )
        self.delete_tray_button.pack(side=tk.LEFT, padx=5, pady=10)

        # Add reset button
        self.reset_button = tk.Button(
            self.button_frame, text="Reset Points", command=self.reset_points
        )
        self.reset_button.pack(side=tk.LEFT, padx=(5, 10), pady=10)

        # Add save and next button
        self.save_button = tk.Button(
            self.button_frame, text="Save & Next Dataset", command=self.save_and_next
        )
        self.save_button.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.X, expand=True)

        # Add status bar for instructions
        self.status_bar = tk.Label(
            self.frame,
            text="Select 4 points for a tray: top-left, top-right, bottom-left, bottom-right",
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Bind mouse click event
        self.canvas.bind("<Button-1>", self.select_point)

        # Bind right-click for tray selection
        self.canvas.bind("<Button-3>", self.select_tray)

        # Start with the first dataset
        self.load_next_dataset()

    def load_next_dataset(self):
        """Load the next dataset to be processed"""
        if self.current_dataset_idx >= len(self.dataset_dirs) - 1:
            messagebox.showinfo("Info", "All datasets have been processed.")
            self.root.quit()
            return

        self.current_dataset_idx += 1
        self.dataset_dir = self.dataset_dirs[self.current_dataset_idx]

        # Reset state for the new dataset
        self.reset_for_next_dataset()

        self.dataset_label.config(text=f"Current Dataset: {self.dataset_dir.name}")

        # Extract zone identifier
        self.zone_identifier = self.extract_zone_identifier(self.dataset_dir)

        if self.zone_identifier is None:
            messagebox.showwarning(
                "Zone Not Found",
                f"Could not determine zone for {self.dataset_dir.name}. Skipping.",
            )
            self.load_next_dataset()
            return

        # Load image for the current dataset
        self.load_image()

        # Load existing tray configurations
        self.load_existing_config()

        # Visualize existing trays
        self.visualize_existing_trays()

    def find_all_datasets(self):
        """Find all datasets in the base directories"""
        self.dataset_dirs = []

        for base_dir in BASE_DIRS:
            if base_dir.exists():
                for dir_path in sorted(base_dir.glob("*")):
                    if (dir_path / "images").exists():
                        self.dataset_dirs.append(dir_path)

        print(f"Found {len(self.dataset_dirs)} datasets")
        for idx, dataset in enumerate(self.dataset_dirs):
            print(f"{idx + 1}. {dataset}")

    def select_point(self, event):
        """Handle mouse click to select a point"""
        if len(self.points) < 4:
            self.points.append((event.x, event.y))
            self.canvas.create_oval(
                event.x - 3,
                event.y - 3,
                event.x + 3,
                event.y + 3,
                fill="red",
                tags="current_points",
            )
            print(f"Point {len(self.points)} selected at ({event.x}, {event.y})")

            # Update status bar with point count
            self.update_status_bar()

            # Automatically add tray when 4 points are selected
            if len(self.points) == 4:
                self.save_tray()

    def save_and_next(self):
        """Save the current configuration and load the next dataset"""
        if self.dataset_dir is None:
            return

        if self.zone_identifier is None:
            messagebox.showerror("Error", "Cannot save, zone identifier is not set.")
            return

        if UPDATE_CONFIG:
            try:
                try:
                    zone = load_zone_from_config(self.zone_identifier)
                except FileNotFoundError:
                    logger.warning(
                        f"Config for {self.zone_identifier} not found. Creating a new one."
                    )
                    zone = Zone(
                        identifier=self.zone_identifier,
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
                    for tc in self.tray_configs
                ]

                # Serialize and save
                config_data = serialize_zone(zone)
                config_path = (
                    project_root
                    / "src"
                    / "environments"
                    / "PlantGrowthChamber"
                    / "configs"
                    / f"{self.zone_identifier}.json"
                )
                with open(config_path, "w") as f:
                    json.dump({"zone": config_data}, f, indent=4)

                logger.debug(
                    f"Successfully saved config for zone {self.zone_identifier} to {config_path}"
                )

            except Exception as e:
                logger.error(
                    f"Error saving config for zone {self.zone_identifier}: {e}"
                )
                messagebox.showerror(
                    "Error", f"Could not save config for {self.zone_identifier}: {e}"
                )
                return
        else:
            # For local config, just save the trays and identifier
            final_config = {
                "zone": {"identifier": self.zone_identifier, "trays": self.tray_configs}
            }
            config_path = self.dataset_dir / "config.json"
            with open(config_path, "w") as f:
                json.dump(final_config, f, indent=4)
            logger.debug(f"Successfully saved configuration to {config_path}")

        # Load the next dataset
        self.load_next_dataset()

    def update_status_bar(self):
        """Update the status bar with current point count"""
        if len(self.points) < 4:
            self.status_bar.config(
                text=f"Select point {len(self.points) + 1}/4: "
                + ["top-left", "top-right", "bottom-left", "bottom-right"][
                    len(self.points)
                ]
            )
        else:
            self.status_bar.config(
                text="Tray added. Select 4 more points for another tray or save and continue."
            )

    def reset_points(self):
        """Reset the current points selection"""
        self.points = []
        self.canvas.delete("current_points")
        self.update_status_bar()

    def scale_point_to_original(self, point):
        """Scale point from display coordinates back to original image coordinates"""
        x, y = point
        orig_x = int(x / self.scale_factor)
        orig_y = int(y / self.scale_factor)
        return (orig_x, orig_y)

    def scale_point_to_display(self, point):
        """Scale point from original image coordinates to display coordinates"""
        x, y = point
        disp_x = int(x * self.scale_factor)
        disp_y = int(y * self.scale_factor)
        return (disp_x, disp_y)

    def save_tray(self):
        """Save the current tray configuration"""
        if len(self.points) == 4:
            # Scale points back to original image coordinates
            scaled_points = [
                self.scale_point_to_original(point) for point in self.points
            ]

            # Get user-defined tray dimensions
            try:
                n_tall = int(self.n_tall_var.get())
                n_wide = int(self.n_wide_var.get())

                # Validate input
                if n_tall < 1 or n_wide < 1:
                    messagebox.showerror(
                        "Invalid Input", "Rows and columns must be positive integers"
                    )
                    return

            except ValueError:
                messagebox.showerror(
                    "Invalid Input", "Please enter valid numbers for rows and columns"
                )
                return

            tray_config = {
                "n_tall": n_tall,
                "n_wide": n_wide,
                "rect": {
                    "top_left": scaled_points[0],
                    "top_right": scaled_points[1],
                    "bottom_left": scaled_points[2],
                    "bottom_right": scaled_points[3],
                },
            }

            # Check if we are editing an existing tray
            if self.selected_tray_index is not None:
                # Update existing tray
                old_index = self.selected_tray_index
                self.tray_configs[old_index] = tray_config

                # Remove old visualization
                old_tag = self.tray_markers[old_index]
                self.canvas.delete(old_tag)

                # Create new visualization
                marker_id = self.visualize_tray(
                    self.points, f"Tray {old_index + 1}: {n_tall}×{n_wide}"
                )
                self.tray_markers[old_index] = marker_id

                print(f"Tray {old_index + 1} updated with dimensions {n_tall}×{n_wide}")

                # Reset editing state
                self.selected_tray_index = None
                self.delete_tray_button.config(state=tk.DISABLED)
                self.cancel_edit_button.config(state=tk.DISABLED)
                self.edit_status_label.config(text="")
            else:
                # Add new tray
                self.tray_configs.append(tray_config)

                # Visualize the tray with a permanent marker
                marker_id = self.visualize_tray(
                    self.points, f"Tray {len(self.tray_configs)}: {n_tall}×{n_wide}"
                )
                self.tray_markers.append(marker_id)

                print(
                    f"Tray added with dimensions {n_tall}×{n_wide}. {len(self.tray_configs)} trays configured so far."
                )

            # Update tray count
            self.tray_count_label.config(text=f"Trays: {len(self.tray_configs)}")

            # Reset points for next tray
            self.points = []
            self.canvas.delete("current_points")
            self.update_status_bar()

        else:
            print(f"Please select exactly 4 points. Current: {len(self.points)}")

    def visualize_tray(self, points, label=None):
        """Draw a permanent visualization of a tray on the canvas"""
        if len(points) != 4:
            return None

        # Create a unique tag for this tray
        tag = f"tray_{len(self.tray_configs)}"

        # Draw polygon connecting the points with semi-transparent fill
        self.canvas.create_polygon(
            points[0][0],
            points[0][1],  # top-left
            points[1][0],
            points[1][1],  # top-right
            points[3][0],
            points[3][1],  # bottom-right
            points[2][0],
            points[2][1],  # bottom-left
            fill="",
            outline="green",
            width=2,
            tags=tag,
        )

        # Add label if provided
        if label:
            # Calculate center point
            center_x = sum(p[0] for p in points) / 4
            center_y = sum(p[1] for p in points) / 4

            self.canvas.create_text(
                center_x,
                center_y,
                text=label,
                fill="green",
                font=("Arial", 12, "bold"),
                tags=tag,
            )

        return tag

    def extract_zone_identifier(self, dataset_path: Path) -> str | None:
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

    def load_existing_config(self):
        """Load existing configuration if available."""
        if self.zone_identifier is None:
            self.tray_configs = []
            return

        if UPDATE_CONFIG:
            try:
                zone = load_zone_from_config(self.zone_identifier)
                self.tray_configs = [
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
                logger.debug(
                    f"Loaded {len(self.tray_configs)} trays for zone {self.zone_identifier}"
                )
            except FileNotFoundError:
                logger.warning(
                    f"No config file found for zone {self.zone_identifier}. Starting fresh."
                )
                self.tray_configs = []
            except Exception as e:
                logger.error(
                    f"Error loading config for zone {self.zone_identifier}: {e}"
                )
                self.tray_configs = []
        else:
            config_path = self.dataset_dir / "config.json"  # type: ignore
            if config_path.exists():
                try:
                    with open(config_path, "r") as f:
                        existing_config = json.load(f)

                    if "zone" in existing_config and "trays" in existing_config["zone"]:
                        self.tray_configs = existing_config["zone"]["trays"]
                        logger.debug(
                            f"Loaded {len(self.tray_configs)} trays from {config_path}"
                        )
                    else:
                        self.tray_configs = []
                        logger.warning(
                            f"No trays found in existing config: {config_path}"
                        )

                except Exception as e:
                    logger.error(f"Error loading config file {config_path}: {e}")
                    self.tray_configs = []
            else:
                self.tray_configs = []

    def load_image(self):
        """Load the first image from the 'images' subdirectory of the current dataset"""
        if self.dataset_dir is None:
            return

        image_dir = self.dataset_dir / "images"
        if not image_dir.exists():
            messagebox.showerror("Error", f"Image directory not found: {image_dir}")
            return

        # Find the last image file (e.g., .png, .jpg)
        image_path = next(
            (
                image_file
                for image_file in sorted(image_dir.glob("*"), reverse=True)
                if image_file.suffix.lower() in {".png", ".jpg", ".jpeg"}
            ),
            None,
        )

        if image_path is None:
            messagebox.showerror("Error", f"No images found in {image_dir}")
            return

        # Load image with OpenCV
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise IOError("Could not read image file")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.original_height, self.original_width, _ = image.shape

            # Resize for display
            max_height = 800
            self.scale_factor = max_height / self.original_height
            display_width = int(self.original_width * self.scale_factor)
            display_height = int(self.original_height * self.scale_factor)

            # Create PhotoImage
            img_pil = Image.fromarray(image)
            img_pil = img_pil.resize(
                (display_width, display_height), Image.Resampling.LANCZOS
            )
            self.tk_image = ImageTk.PhotoImage(img_pil)

            # Update canvas
            self.canvas.config(width=display_width, height=display_height)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        except Exception as e:
            messagebox.showerror("Image Load Error", f"Could not load image: {e}")

    def reset_for_next_dataset(self):
        """Reset the state for the next dataset"""
        self.points = []
        self.tray_configs = []
        self.canvas.delete("all")
        self.tk_image = None
        self.tray_markers = []
        self.selected_tray_index = None
        self.tray_count_label.config(text="Trays: 0")
        self.edit_status_label.config(text="")
        self.update_status_bar()

    def visualize_existing_trays(self):
        """Draw visualizations for trays that already exist in the config"""
        for i, tray_config in enumerate(self.tray_configs):
            rect = tray_config["rect"]
            points = [
                self.scale_point_to_display(rect["top_left"]),
                self.scale_point_to_display(rect["top_right"]),
                self.scale_point_to_display(rect["bottom_left"]),
                self.scale_point_to_display(rect["bottom_right"]),
            ]

            # Reorder for polygon drawing
            poly_points = [points[0], points[1], points[3], points[2]]

            label = f"Tray {i + 1}: {tray_config['n_tall']}×{tray_config['n_wide']}"
            marker_id = self.visualize_tray(poly_points, label)
            if marker_id:
                self.tray_markers.append(marker_id)

    def select_tray(self, event):
        """Handle right-click to select a tray for editing or deletion"""
        if not self.tray_configs:
            return

        # Find the tray that was clicked
        for i, tray_config in enumerate(self.tray_configs):
            rect = tray_config["rect"]
            points = [
                self.scale_point_to_display(rect["top_left"]),
                self.scale_point_to_display(rect["top_right"]),
                self.scale_point_to_display(rect["bottom_left"]),
                self.scale_point_to_display(rect["bottom_right"]),
            ]

            # Create a polygon to check if the click is inside
            poly = np.array([points[0], points[1], points[3], points[2]])
            if cv2.pointPolygonTest(poly, (event.x, event.y), False) >= 0:
                self.selected_tray_index = i
                self.edit_status_label.config(text=f"Editing Tray {i + 1}", fg="blue")
                self.delete_tray_button.config(state=tk.NORMAL)
                self.cancel_edit_button.config(state=tk.NORMAL)

                # Load tray points into current selection for editing
                self.points = [points[0], points[1], points[2], points[3]]
                self.canvas.delete("current_points")
                for x, y in self.points:
                    self.canvas.create_oval(
                        x - 3, y - 3, x + 3, y + 3, fill="blue", tags="current_points"
                    )
                self.update_status_bar()
                return

    def delete_selected_tray(self):
        """Delete the tray that is currently selected"""
        if self.selected_tray_index is not None:
            # Confirm deletion
            if not messagebox.askyesno(
                "Confirm Deletion",
                f"Are you sure you want to delete Tray {self.selected_tray_index + 1}?",
            ):
                return

            # Remove from config and visualization
            index = self.selected_tray_index
            self.tray_configs.pop(index)
            tag_to_delete = self.tray_markers.pop(index)
            self.canvas.delete(tag_to_delete)

            # Reset editing state
            self.cancel_edit()
            self.tray_count_label.config(text=f"Trays: {len(self.tray_configs)}")
            logger.debug(f"Tray {index + 1} deleted.")

    def cancel_edit(self):
        """Cancel the current tray editing operation"""
        self.selected_tray_index = None
        self.points = []
        self.canvas.delete("current_points")
        self.edit_status_label.config(text="")
        self.delete_tray_button.config(state=tk.DISABLED)
        self.cancel_edit_button.config(state=tk.DISABLED)
        self.update_status_bar()


if __name__ == "__main__":
    root = tk.Tk()
    app = TrayConfigApp(root)
    root.mainloop()
