import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk


def scale_point_to_original(point, scale_factor):
    """Scale point from display coordinates back to original image coordinates"""
    x, y = point
    orig_x = int(x / scale_factor)
    orig_y = int(y / scale_factor)
    return (orig_x, orig_y)


def scale_point_to_display(point, scale_factor):
    """Scale point from original image coordinates to display coordinates"""
    x, y = point
    disp_x = int(x * scale_factor)
    disp_y = int(y * scale_factor)
    return (disp_x, disp_y)


def visualize_tray(canvas, tray, scale_factor, label=None, tray_markers_count=0):
    """Draw a permanent visualization of a tray on the canvas"""
    # Scale points to display coordinates
    points = [
        scale_point_to_display(tray.rect.top_left, scale_factor),
        scale_point_to_display(tray.rect.top_right, scale_factor),
        scale_point_to_display(tray.rect.bottom_right, scale_factor),
        scale_point_to_display(tray.rect.bottom_left, scale_factor),
    ]

    # Create a unique tag for this tray
    tag = f"tray_{tray_markers_count}"

    # Draw polygon connecting the points with semi-transparent fill
    canvas.create_polygon(
        points[0][0],
        points[0][1],  # top-left
        points[1][0],
        points[1][1],  # top-right
        points[2][0],
        points[2][1],  # bottom-right
        points[3][0],
        points[3][1],  # bottom-left
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

        canvas.create_text(
            center_x,
            center_y,
            text=label,
            fill="green",
            font=("Arial", 12, "bold"),
            tags=tag,
        )

    return tag


def load_and_prepare_image(image_path):
    """Load and prepare an image for display on canvas with optimized performance"""
    try:
        try:
            # Determine screen dimensions to calculate max window size
            root = tk.Tk()
            root.withdraw()  # Hide the root window
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            root.destroy()
        except Exception:
            # Fallback to a conservative 720p resolution if screen size detection fails
            screen_width, screen_height = 1280, 720

        # Set max display size to 80% of the screen dimensions
        max_display_width = int(screen_width * 0.8)
        max_display_height = int(screen_height * 0.8)

        # Use PIL directly for faster loading instead of cv2
        image = Image.open(str(image_path))
        if not image:
            raise IOError("Could not read image file")

        # Get original dimensions
        original_width, original_height = image.size

        # Calculate scale factor to fit within max dimensions while preserving aspect ratio
        width_ratio = max_display_width / original_width
        height_ratio = max_display_height / original_height
        scale_factor = min(width_ratio, height_ratio)

        # Do not scale up small images
        if scale_factor > 1.0:
            scale_factor = 1.0

        display_width = int(original_width * scale_factor)
        display_height = int(original_height * scale_factor)

        # Resize image using more efficient BILINEAR resampling instead of LANCZOS
        img_resized = image.resize(
            (display_width, display_height), Image.Resampling.BILINEAR
        )

        # Convert to Tkinter PhotoImage
        tk_image = ImageTk.PhotoImage(img_resized)

        return (
            tk_image,
            scale_factor,
            original_width,
            original_height,
            display_width,
            display_height,
        )
    except Exception as e:
        raise Exception(f"Could not load image: {e}") from e


def draw_point(canvas, x, y, color="red", tag="current_points"):
    """Draw a point on the canvas"""
    canvas.create_oval(
        x - 3,
        y - 3,
        x + 3,
        y + 3,
        fill=color,
        tags=tag,
    )


def is_point_in_tray(tray_config, point, scale_factor):
    """Check if a point is inside a tray"""
    rect = tray_config["rect"]
    points = [
        scale_point_to_display(rect["top_left"], scale_factor),
        scale_point_to_display(rect["top_right"], scale_factor),
        scale_point_to_display(rect["bottom_right"], scale_factor),
        scale_point_to_display(rect["bottom_left"], scale_factor),
    ]
    # Correct order for hit-testing
    poly = np.array([points[0], points[1], points[3], points[2]])
    return cv2.pointPolygonTest(poly, point, False) >= 0
