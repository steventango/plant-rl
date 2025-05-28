import json
import logging
import os
import tkinter as tk
from datetime import datetime
from itertools import chain
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

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
BASE_DIRS = [
    # Path("/data/phytochrome_exp"),
    # Path("/data/nazmus_exp"),
]
BASE_DIRS.extend(
    chain(
        Path("/data/online/E8/P0").rglob("*"),
    )
)


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

        tk.Label(self.tray_dim_frame, text="Tray Dimensions:").grid(row=0, column=0, sticky="w")

        tk.Label(self.tray_dim_frame, text="Rows (n_tall):").grid(row=0, column=1, padx=5)
        self.n_tall_var = tk.StringVar(value=str(self.default_n_tall))
        self.n_tall_entry = ttk.Spinbox(self.tray_dim_frame, from_=1, to=20, width=5, textvariable=self.n_tall_var)
        self.n_tall_entry.grid(row=0, column=2)

        tk.Label(self.tray_dim_frame, text="Columns (n_wide):").grid(row=0, column=3, padx=5)
        self.n_wide_var = tk.StringVar(value=str(self.default_n_wide))
        self.n_wide_entry = ttk.Spinbox(self.tray_dim_frame, from_=1, to=20, width=5, textvariable=self.n_wide_var)
        self.n_wide_entry.grid(row=0, column=4)

        # Add tray counter and editing status
        self.tray_count_frame = tk.Frame(self.tray_dim_frame)
        self.tray_count_frame.grid(row=0, column=5, padx=(20, 5), sticky="e")

        self.tray_count_label = tk.Label(self.tray_count_frame, text="Trays: 0", font=("Arial", 10))
        self.tray_count_label.pack(side=tk.LEFT)

        self.edit_status_label = tk.Label(self.tray_count_frame, text="", font=("Arial", 10, "italic"), fg="blue")
        self.edit_status_label.pack(side=tk.RIGHT, padx=(10, 0))

        # Create canvas for the image
        self.canvas = tk.Canvas(self.frame)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Create button frame
        self.button_frame = tk.Frame(self.frame)
        self.button_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Add cancel edit button
        self.cancel_edit_button = tk.Button(
            self.button_frame, text="Cancel Edit", command=self.cancel_edit, state=tk.DISABLED
        )
        self.cancel_edit_button.pack(side=tk.LEFT, padx=(10, 5), pady=10)

        # Add delete tray button
        self.delete_tray_button = tk.Button(
            self.button_frame, text="Delete Tray", command=self.delete_selected_tray, state=tk.DISABLED
        )
        self.delete_tray_button.pack(side=tk.LEFT, padx=5, pady=10)

        # Add reset button
        self.reset_button = tk.Button(self.button_frame, text="Reset Points", command=self.reset_points)
        self.reset_button.pack(side=tk.LEFT, padx=(5, 10), pady=10)

        # Add save and next button
        self.save_button = tk.Button(self.button_frame, text="Save & Next Dataset", command=self.save_and_next)
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

    def find_all_datasets(self):
        """Find all datasets in the base directories"""
        self.dataset_dirs = []

        for base_dir in BASE_DIRS:
            if base_dir.exists():
                for dir_path in sorted(base_dir.glob("z*")):
                    if (dir_path / "images").exists():
                        self.dataset_dirs.append(dir_path)

        print(f"Found {len(self.dataset_dirs)} datasets")
        for idx, dataset in enumerate(self.dataset_dirs):
            print(f"{idx+1}. {dataset}")

    def select_point(self, event):
        """Handle mouse click to select a point"""
        if len(self.points) < 4:
            self.points.append((event.x, event.y))
            self.canvas.create_oval(
                event.x - 3, event.y - 3, event.x + 3, event.y + 3, fill="red", tags="current_points"
            )
            print(f"Point {len(self.points)} selected at ({event.x}, {event.y})")

            # Update status bar with point count
            self.update_status_bar()

            # Automatically add tray when 4 points are selected
            if len(self.points) == 4:
                self.save_tray()

    def update_status_bar(self):
        """Update the status bar with current point count"""
        if len(self.points) < 4:
            self.status_bar.config(
                text=f"Select point {len(self.points) + 1}/4: "
                + ["top-left", "top-right", "bottom-left", "bottom-right"][len(self.points)]
            )
        else:
            self.status_bar.config(text="Tray added. Select 4 more points for another tray or save and continue.")

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
            scaled_points = [self.scale_point_to_original(point) for point in self.points]

            # Get user-defined tray dimensions
            try:
                n_tall = int(self.n_tall_var.get())
                n_wide = int(self.n_wide_var.get())

                # Validate input
                if n_tall < 1 or n_wide < 1:
                    messagebox.showerror("Invalid Input", "Rows and columns must be positive integers")
                    return

            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter valid numbers for rows and columns")
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
                marker_id = self.visualize_tray(self.points, f"Tray {old_index + 1}: {n_tall}×{n_wide}")
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
                marker_id = self.visualize_tray(self.points, f"Tray {len(self.tray_configs)}: {n_tall}×{n_wide}")
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
        polygon_id = self.canvas.create_polygon(
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

            label_id = self.canvas.create_text(
                center_x, center_y, text=label, fill="green", font=("Arial", 12, "bold"), tags=tag
            )

        return tag

    def extract_zone_identifier(self, dataset_path):
        """Extract zone identifier from the dataset directory path"""
        zone_name = dataset_path.name
        logger.info(f"Extracting zone identifier from path: {dataset_path}")

        if zone_name.startswith("z"):
            try:
                # First try to parse z followed by numbers until non-digit
                import re

                match = re.match(r"z(\d+)", zone_name)
                if match:
                    zone_id = int(match.group(1))
                    logger.info(f"Successfully extracted zone identifier {zone_id} using regex")
                    return zone_id

                # Fallback to old method if regex doesn't match
                if "c" in zone_name:
                    identifier_str = zone_name[1 : zone_name.find("c")]
                    zone_id = int(identifier_str)
                    logger.info(f"Extracted zone identifier {zone_id} using 'c' delimiter method")
                    return zone_id

                # If no 'c' found, try to extract numeric part after 'z'
                numeric_part = "".join(c for c in zone_name[1:] if c.isdigit())
                if numeric_part:
                    zone_id = int(numeric_part)
                    logger.info(f"Extracted zone identifier {zone_id} using numeric extraction")
                    return zone_id

                logger.warning(f"Could not extract zone identifier from {zone_name} using any method")
            except ValueError as e:
                logger.error(f"ValueError while parsing identifier from {zone_name}: {e}")

        # Print a warning about using default zone
        logger.warning(f"Using default zone 1 for path {dataset_path}")
        return 1

    def load_existing_config(self):
        """Load existing configuration if available"""
        config_path = self.dataset_dir / "config.json"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)

                if "zone" in config and "trays" in config["zone"]:
                    self.tray_configs = config["zone"]["trays"]

                    # Display existing trays on canvas
                    for i, tray in enumerate(self.tray_configs):
                        rect = tray["rect"]
                        display_points = [
                            self.scale_point_to_display(rect["top_left"]),
                            self.scale_point_to_display(rect["top_right"]),
                            self.scale_point_to_display(rect["bottom_left"]),
                            self.scale_point_to_display(rect["bottom_right"]),
                        ]
                        label = f"Tray {i+1}: {tray['n_tall']}×{tray['n_wide']}"
                        marker_id = self.visualize_tray(display_points, label)
                        self.tray_markers.append(marker_id)

                    # Update tray count
                    self.tray_count_label.config(text=f"Trays: {len(self.tray_configs)}")

                    print(f"Loaded {len(self.tray_configs)} existing trays from config")

                    # If configuration already exists, ask if user wants to edit it
                    if len(self.tray_configs) > 0:
                        if not messagebox.askyesno(
                            "Configuration Exists",
                            f"Found existing configuration with {len(self.tray_configs)} trays. Do you want to edit it?",
                        ):
                            # User chose not to edit, go to next dataset
                            print("Skipping to next dataset...")
                            self.load_next_dataset()
                        else:
                            # User chose to edit - clear existing trays to allow fresh selection
                            self.clear_all_trays()
                            messagebox.showinfo(
                                "Edit Mode", "Existing trays have been cleared. Please select new tray positions."
                            )
            except Exception as e:
                print(f"Error loading configuration: {e}")

    def clear_all_trays(self):
        """Clear all existing trays to allow fresh selection"""
        # Clear canvas markers
        for tag in self.tray_markers:
            self.canvas.delete(tag)

        # Reset data structures
        self.tray_markers = []
        self.tray_configs = []
        self.points = []
        self.canvas.delete("current_points")

        # Reset UI elements
        self.tray_count_label.config(text="Trays: 0")
        self.update_status_bar()

        print("All existing trays cleared for fresh selection")

    def save_current_config(self):
        """Save the current configuration to a JSON file"""
        if not self.dataset_dir:
            return

        if self.tray_configs:
            # Extract the zone identifier from the dataset directory path
            zone_identifier = self.extract_zone_identifier(self.dataset_dir)
            logger.info(f"Saving configuration for zone {zone_identifier}")

            config = {"zone": {"identifier": zone_identifier, "trays": self.tray_configs}}

            if UPDATE_CONFIG:
                # Find and update existing config file in PlantGrowthChamber configs
                return self.update_existing_config(zone_identifier, config)
            else:
                # Save to the original dataset directory
                config_path = self.dataset_dir / "config.json"
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=4)
                logger.info(f"Configuration saved to {config_path} with zone identifier: {zone_identifier}")
                return True
        else:
            logger.warning("No trays to save.")
            return False

    def find_config_file(self, zone_identifier):
        """Find the corresponding config file in PlantGrowthChamber configs directory"""
        config_dir = Path("/workspaces/plant-rl/src/environments/PlantGrowthChamber/configs")
        logger.info(f"Looking for config files for zone {zone_identifier} in {config_dir}")

        # Look for config files matching the zone identifier
        pattern = f"z{zone_identifier}*.json"
        matching_files = list(config_dir.glob(pattern))

        if matching_files:
            # Return the first matching file
            logger.info(f"Found existing config file: {matching_files[0]}")
            return matching_files[0]

        # If no matching file, construct a default filename
        default_file = config_dir / f"z{zone_identifier}.json"
        logger.info(f"No existing config file found. Will create: {default_file}")
        return default_file

    def update_existing_config(self, zone_identifier, new_config):
        """Update an existing config file with new tray configuration"""
        config_file = self.find_config_file(zone_identifier)
        logger.info(f"Updating/creating config for zone {zone_identifier} at {config_file}")

        # Check if the config file exists
        if config_file.exists():
            try:
                # Load existing config
                with open(config_file, "r") as f:
                    existing_config = json.load(f)
                logger.info(f"Loaded existing config from {config_file}")

                # Only update the trays section, preserve other settings
                if "zone" not in existing_config:
                    existing_config["zone"] = {}
                    logger.info("Adding missing 'zone' section to config")

                existing_config["zone"]["trays"] = new_config["zone"]["trays"]
                logger.info(f"Updated trays section with {len(new_config['zone']['trays'])} trays")

                # Make sure the identifier is preserved/set
                existing_config["zone"]["identifier"] = zone_identifier
                logger.info(f"Ensured zone identifier is set to {zone_identifier}")

                # Save the updated config
                with open(config_file, "w") as f:
                    json.dump(existing_config, f, indent=4)

                logger.info(f"Successfully updated configuration in {config_file}")
                return True

            except Exception as e:
                logger.error(f"Error updating configuration file {config_file}: {e}")
                # Fallback: create a new file
                with open(config_file, "w") as f:
                    json.dump(new_config, f, indent=4)
                logger.info(f"Created new configuration file at {config_file} after error")
                return True
        else:
            # File doesn't exist, create it
            os.makedirs(config_file.parent, exist_ok=True)
            with open(config_file, "w") as f:
                json.dump(new_config, f, indent=4)
            logger.info(f"Created new configuration file at {config_file}")
            return True

    def load_next_dataset(self):
        """Load the next dataset in the sequence"""
        self.current_dataset_idx += 1

        if self.current_dataset_idx >= len(self.dataset_dirs):
            print("All datasets have been processed!")
            messagebox.showinfo("Complete", "All datasets have been processed!")
            self.root.destroy()
            return

        # Reset for the new dataset
        self.dataset_dir = self.dataset_dirs[self.current_dataset_idx]
        self.tray_configs = []
        self.points = []
        self.tray_markers = []

        # Update the dataset label
        self.dataset_label.config(
            text=f"Dataset {self.current_dataset_idx + 1}/{len(self.dataset_dirs)}: {self.dataset_dir}"
        )

        # Reset the tray count
        self.tray_count_label.config(text="Trays: 0")

        # Reset the status bar
        self.update_status_bar()

        # Load image from this dataset
        self.load_image()

        # Load existing configuration if available
        self.load_existing_config()

    def save_and_next(self):
        """Save the current configuration and move to the next dataset"""
        if self.points:
            # Points are selected but not added to a tray
            response = messagebox.askyesno(
                "Unsaved Points", "You have selected points that haven't been added as a tray. Add them now?"
            )
            if response:
                self.save_tray()

        # Save the configuration
        if self.save_current_config():
            messagebox.showinfo("Success", f"Configuration saved for {self.dataset_dir}")

        # Load the next dataset
        self.load_next_dataset()

    def load_image(self):
        """Find and load an image around 10 AM from the current dataset"""
        images_dir = self.dataset_dir / "images"
        image_files = sorted(list(images_dir.glob("*.jpg")), reverse=True)

        if not image_files:
            print(f"No images found in {images_dir}")
            # Show placeholder in the canvas
            self.canvas.delete("all")
            self.canvas.create_text(200, 200, text=f"No images found in {self.dataset_dir}")
            return

        # Find an image near 10 AM
        target_hour = 10
        best_image = None
        min_diff = float("inf")

        for image_file in image_files:
            try:
                # Extract time from filename (format: z11c1--2022-07-22--HH-MM-SS.jpg)
                parts = image_file.name.split("--")
                if len(parts) >= 3:
                    time_part = parts[2].split(".")[0]  # Get HH-MM-SS
                    hour = int(time_part.split("-")[0])
                    diff = abs(hour - target_hour)

                    if diff < min_diff:
                        min_diff = diff
                        best_image = image_file
            except Exception as e:
                continue

        if best_image is None and image_files:
            # Fallback: just use the first image if no 10 AM image found
            best_image = image_files[0]

        if best_image:
            print(f"Loading image: {best_image}")
            img = cv2.imread(str(best_image))
            if img is None:
                print(f"Failed to load image: {best_image}")
                return

            # Undistort the image
            camera_matrix = np.array([[1800.0, 0.0, 1296.0], [0.0, 1800.0, 972.0], [0.0, 0.0, 1.0]])
            dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0])

            # Undistort
            undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs)

            print(f"Image undistorted successfully")

            # Convert to RGB for display
            img = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB)

            # Store original dimensions
            self.original_height, self.original_width = img.shape[:2]

            # Calculate resize factor to fit screen
            screen_width = self.root.winfo_screenwidth() - 300
            screen_height = self.root.winfo_screenheight() - 200

            self.scale_factor = min(screen_width / self.original_width, screen_height / self.original_height)

            # Resize the image
            new_width = int(self.original_width * self.scale_factor)
            new_height = int(self.original_height * self.scale_factor)
            resized_img = cv2.resize(img, (new_width, new_height))

            print(f"Original image dimensions: {self.original_width}x{self.original_height}")
            print(f"Display dimensions: {new_width}x{new_height}")
            print(f"Scale factor: {self.scale_factor}")

            # Update canvas size
            self.canvas.config(width=new_width, height=new_height)

            # Convert to PhotoImage and display
            pil_img = Image.fromarray(resized_img)
            self.tk_image = ImageTk.PhotoImage(pil_img)

            # Clear canvas and display the new image
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def select_tray(self, event):
        """Right-click handler to select a tray for editing"""
        # Get canvas coordinates of the click
        x, y = event.x, event.y

        # Check if we clicked inside any tray
        for i, tray_tag in enumerate(self.tray_markers):
            # Get the polygon coordinates for this tray
            items = self.canvas.find_withtag(tray_tag)
            if not items:
                continue

            # Get the polygon (should be first item with this tag)
            polygon = items[0]

            # Check if point is inside polygon
            if self.point_in_polygon(x, y, self.canvas.coords(polygon)):
                # Select this tray for editing
                self.start_editing_tray(i)
                return

        # If we got here, no tray was clicked
        self.cancel_edit()

    def point_in_polygon(self, x, y, poly_coords):
        """Check if a point (x,y) is inside a polygon defined by coordinates"""
        # Convert flat list to pairs
        vertices = [(poly_coords[i], poly_coords[i + 1]) for i in range(0, len(poly_coords), 2)]

        # Ray casting algorithm
        inside = False
        j = len(vertices) - 1
        for i in range(len(vertices)):
            xi, yi = vertices[i]
            xj, yj = vertices[j]

            # Check if point is on edge
            if (yi == y and xi == x) or (yi > y) != (yj > y) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i

        return inside

    def start_editing_tray(self, tray_index):
        """Start editing the selected tray"""
        # Cancel any current editing
        self.cancel_edit()

        # Set the selected tray index
        self.selected_tray_index = tray_index

        # Enable delete and cancel buttons
        self.delete_tray_button.config(state=tk.NORMAL)
        self.cancel_edit_button.config(state=tk.NORMAL)

        # Highlight the selected tray
        tray_tag = self.tray_markers[tray_index]
        items = self.canvas.find_withtag(tray_tag)
        for item in items:
            if self.canvas.type(item) == "polygon":
                self.canvas.itemconfig(item, outline="red", width=3)

        # Load tray dimensions into spinboxes
        tray_config = self.tray_configs[tray_index]
        self.n_tall_var.set(str(tray_config["n_tall"]))
        self.n_wide_var.set(str(tray_config["n_wide"]))

        # Set up points for re-drawing
        rect = tray_config["rect"]
        self.points = [
            self.scale_point_to_display(rect["top_left"]),
            self.scale_point_to_display(rect["top_right"]),
            self.scale_point_to_display(rect["bottom_left"]),
            self.scale_point_to_display(rect["bottom_right"]),
        ]

        # Draw the current points
        for point in self.points:
            x, y = point
            self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="red", tags="current_points")

        # Update status bar and edit label
        tray_num = tray_index + 1
        self.status_bar.config(text=f"Editing Tray {tray_num}. Click to set new corner points.")
        self.edit_status_label.config(text=f"Editing Tray {tray_num}")

        print(f"Started editing tray {tray_num}")

    def cancel_edit(self):
        """Cancel the current tray editing operation"""
        if self.selected_tray_index is not None:
            # Restore original tray appearance
            tray_tag = self.tray_markers[self.selected_tray_index]
            items = self.canvas.find_withtag(tray_tag)
            for item in items:
                if self.canvas.type(item) == "polygon":
                    self.canvas.itemconfig(item, outline="green", width=2)

            # Clear selection and points
            self.selected_tray_index = None
            self.points = []
            self.canvas.delete("current_points")

            # Update UI
            self.delete_tray_button.config(state=tk.DISABLED)
            self.cancel_edit_button.config(state=tk.DISABLED)
            self.edit_status_label.config(text="")
            self.update_status_bar()

            print("Cancelled tray editing")

    def delete_selected_tray(self):
        """Delete the currently selected tray"""
        if self.selected_tray_index is not None:
            tray_num = self.selected_tray_index + 1

            # Confirm deletion
            if messagebox.askyesno("Confirm Delete", f"Delete Tray {tray_num}?"):
                # Remove from data structures
                del self.tray_configs[self.selected_tray_index]

                # Remove from canvas
                tray_tag = self.tray_markers[self.selected_tray_index]
                self.canvas.delete(tray_tag)
                del self.tray_markers[self.selected_tray_index]

                # Reset editing state
                self.selected_tray_index = None
                self.points = []
                self.canvas.delete("current_points")

                # Update UI
                self.delete_tray_button.config(state=tk.DISABLED)
                self.cancel_edit_button.config(state=tk.DISABLED)
                self.tray_count_label.config(text=f"Trays: {len(self.tray_configs)}")
                self.edit_status_label.config(text="")
                self.update_status_bar()

                # Redraw all trays with updated numbers
                self.redraw_all_trays()

                print(f"Deleted tray {tray_num}")

    def redraw_all_trays(self):
        """Redraw all trays with updated numbers"""
        # Clear all existing tray visualizations
        for tag in self.tray_markers:
            self.canvas.delete(tag)
        self.tray_markers = []

        # Redraw all trays
        for i, tray in enumerate(self.tray_configs):
            rect = tray["rect"]
            display_points = [
                self.scale_point_to_display(rect["top_left"]),
                self.scale_point_to_display(rect["top_right"]),
                self.scale_point_to_display(rect["bottom_left"]),
                self.scale_point_to_display(rect["bottom_right"]),
            ]
            label = f"Tray {i+1}: {tray['n_tall']}×{tray['n_wide']}"
            marker_id = self.visualize_tray(display_points, label)
            self.tray_markers.append(marker_id)


# Main entry point
if __name__ == "__main__":
    root = tk.Tk()
    app = TrayConfigApp(root)
    root.mainloop()
