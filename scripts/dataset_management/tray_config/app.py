import tkinter as tk
from tkinter import messagebox, ttk

from tray_config_utils import (
    extract_zone_identifier,
    load_existing_config,
    logger,
    save_config,
)
from visualization import (
    draw_point,
    is_point_in_tray,
    load_and_prepare_image,
    scale_point_to_display,
    scale_point_to_original,
    visualize_tray,
)
from environments.PlantGrowthChamber.zones import Rect, Tray


class TrayConfigApp:
    def __init__(self, root, base_dirs, update_config=True):
        self.root = root
        self.root.title("Tray Configuration Tool")
        self.update_config = update_config
        self.base_dirs = base_dirs

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
        self.zone_identifier = None

        # Default tray dimensions
        self.default_n_tall = 4
        self.default_n_wide = 8

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
        self.zone_identifier = extract_zone_identifier(self.dataset_dir)

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

        for base_dir in self.base_dirs:
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
            draw_point(self.canvas, event.x, event.y)
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

        success = save_config(
            self.dataset_dir,
            self.zone_identifier,
            self.tray_configs,
            self.update_config,
        )

        if not success:
            messagebox.showerror(
                "Error", f"Could not save config for {self.zone_identifier}"
            )
            return

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

    def save_tray(self):
        """Save the current tray configuration"""
        if len(self.points) == 4:
            # Scale points back to original image coordinates
            scaled_points = [
                scale_point_to_original(point, self.scale_factor)
                for point in self.points
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
                tray = Tray(
                    n_tall=tray_config["n_tall"],
                    n_wide=tray_config["n_wide"],
                    rect=Rect(
                        top_left=tuple(tray_config["rect"]["top_left"]),
                        top_right=tuple(tray_config["rect"]["top_right"]),
                        bottom_left=tuple(tray_config["rect"]["bottom_left"]),
                        bottom_right=tuple(tray_config["rect"]["bottom_right"]),
                    ),
                )
                marker_id = visualize_tray(
                    self.canvas,
                    tray,
                    self.scale_factor,
                    f"Tray {old_index + 1}: {n_tall}×{n_wide}",
                    old_index,
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

                tray = Tray(
                    n_tall=tray_config["n_tall"],
                    n_wide=tray_config["n_wide"],
                    rect=Rect(
                        top_left=tuple(tray_config["rect"]["top_left"]),
                        top_right=tuple(tray_config["rect"]["top_right"]),
                        bottom_left=tuple(tray_config["rect"]["bottom_left"]),
                        bottom_right=tuple(tray_config["rect"]["bottom_right"]),
                    ),
                )
                # Visualize the tray with a permanent marker
                marker_id = visualize_tray(
                    self.canvas,
                    tray,
                    self.scale_factor,
                    f"Tray {len(self.tray_configs)}: {n_tall}×{n_wide}",
                    len(self.tray_markers),
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

    def load_existing_config(self):
        """Load existing configuration if available."""
        self.tray_configs = load_existing_config(
            self.dataset_dir, self.zone_identifier, self.update_config
        )
        if self.tray_configs:
            self.tray_count_label.config(text=f"Trays: {len(self.tray_configs)}")

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
        # # Find the 20th most recent image file (e.g., .png, .jpg)
        # sorted_images = [
        #     image_file
        #     for image_file in sorted(image_dir.glob("*"), reverse=True)
        #     if image_file.suffix.lower() in {".png", ".jpg", ".jpeg"}
        # ]
        # image_path = sorted_images[19] if len(sorted_images) > 19 else None

        if image_path is None:
            messagebox.showerror("Error", f"No images found in {image_dir}")
            return

        # Load image
        try:
            (
                tk_image,
                scale_factor,
                original_width,
                original_height,
                display_width,
                display_height,
            ) = load_and_prepare_image(image_path)

            self.original_width = original_width
            self.original_height = original_height
            self.scale_factor = scale_factor
            self.tk_image = tk_image

            # Update canvas
            self.canvas.config(width=display_width, height=display_height)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        except Exception as e:
            messagebox.showerror("Image Load Error", str(e))

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
            tray = Tray(
                n_tall=tray_config["n_tall"],
                n_wide=tray_config["n_wide"],
                rect=Rect(
                    top_left=tuple(rect["top_left"]),
                    top_right=tuple(rect["top_right"]),
                    bottom_left=tuple(rect["bottom_left"]),
                    bottom_right=tuple(rect["bottom_right"]),
                ),
            )
            label = f"Tray {i + 1}: {tray_config['n_tall']}×{tray_config['n_wide']}"
            marker_id = visualize_tray(self.canvas, tray, self.scale_factor, label, i)
            if marker_id:
                self.tray_markers.append(marker_id)

    def select_tray(self, event):
        """Handle right-click to select a tray for editing or deletion"""
        if not self.tray_configs:
            return

        for i, tray_config in enumerate(self.tray_configs):
            if is_point_in_tray(tray_config, (event.x, event.y), self.scale_factor):
                self.selected_tray_index = i
                self.edit_status_label.config(text=f"Editing Tray {i + 1}", fg="blue")
                self.delete_tray_button.config(state=tk.NORMAL)
                self.cancel_edit_button.config(state=tk.NORMAL)

                # Load tray points into current selection for editing
                rect = tray_config["rect"]
                points = [
                    scale_point_to_display(rect["top_left"], self.scale_factor),
                    scale_point_to_display(rect["top_right"], self.scale_factor),
                    scale_point_to_display(rect["bottom_left"], self.scale_factor),
                    scale_point_to_display(rect["bottom_right"], self.scale_factor),
                ]
                self.points = points
                self.canvas.delete("current_points")
                for x, y in self.points:
                    draw_point(self.canvas, x, y, "blue")
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
