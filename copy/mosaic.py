# Combine images in this parent directory into a mosaic image.
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# the images are named alliance-zone01-camera01.jpg
# the images are named alliance-zone01-camera02.jpg
# ...
# the images are named alliance-zone12-camera01.jpg
# the images are named alliance-zone12-camera02.jpg

# we want the grid to be roughly rectangular but have the two cameras above and below each other

# Configuration
NUM_ZONES = 12
GRID_COLS = 4  # Number of zone-pairs per row in the mosaic
IMAGE_BASENAME = "alliance"
IMAGE_EXTENSION = ".jpg"
OUTPUT_FILENAME = "mosaic.jpg"
MAX_DIMENSION = 3840  # Max width/height for 4K

# Determine directories
# Assumes the script is in a subdirectory (e.g., 'copy') and images are in its parent directory
SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_DIR = SCRIPT_DIR

def create_mosaic():
    # Calculate grid rows based on total zones and columns
    # This assumes NUM_ZONES is perfectly divisible by GRID_COLS for a full grid.
    # If not, the grid might have empty spots in the last row.
    GRID_ROWS = (NUM_ZONES + GRID_COLS - 1) // GRID_COLS

    # Get dimensions from the first image (zone01, camera01)
    try:
        first_image_name = f"{IMAGE_BASENAME}-zone01-camera01{IMAGE_EXTENSION}"
        first_image_path = IMAGE_DIR / first_image_name
        with Image.open(first_image_path) as img:
            img_width, img_height = img.size
    except FileNotFoundError:
        print(f"Error: First image {first_image_path} not found. Cannot determine image dimensions.")
        return
    except Exception as e:
        print(f"Error opening first image {first_image_path}: {e}")
        return

    # Calculate mosaic dimensions
    # Each "cell" in the grid contains two images stacked vertically (camera01 above camera02)
    mosaic_width = img_width * GRID_COLS
    mosaic_height = img_height * 2 * GRID_ROWS  # Each zone-pair is 2 images tall

    # Create a new blank image for the mosaic
    mosaic_image = Image.new("RGB", (mosaic_width, mosaic_height))
    print(f"Creating mosaic of size {mosaic_width}x{mosaic_height}...")

    # Font setup for drawing text
    text_color = (255, 255, 255)  # White
    text_position = (10, 10)
    try:
        # Try to load a common font, adjust path if necessary or use a specific one
        font = ImageFont.truetype("arial.ttf", size=int(img_height / 20)) # Adjust size as needed
    except IOError:
        print("Arial font not found, using default font. Text quality may vary.")
        font = ImageFont.load_default(size=int(img_height / 20))


    for zone_idx_0_based in range(NUM_ZONES):
        zone_num = zone_idx_0_based + 1  # Zone numbers are 1-indexed in filenames

        # Calculate grid position for this zone's pair (top-left of camera01)
        row = zone_idx_0_based // GRID_COLS
        col = zone_idx_0_based % GRID_COLS

        # Calculate paste coordinates for camera01
        x_offset = col * img_width
        y_offset_cam1 = row * (img_height * 2) # Each zone-pair takes up 2*img_height vertically

        # Calculate paste coordinates for camera02 (directly below camera01)
        y_offset_cam2 = y_offset_cam1 + img_height

        # Process camera 01
        try:
            cam1_filename = f"{IMAGE_BASENAME}-zone{zone_num:02d}-camera01{IMAGE_EXTENSION}"
            cam1_path = IMAGE_DIR / cam1_filename
            with Image.open(cam1_path) as cam1_img:
                if cam1_img.size != (img_width, img_height):
                    print(f"Warning: Image {cam1_path} has different dimensions. Resizing.")
                    cam1_img = cam1_img.resize((img_width, img_height))

                # Add text to cam1_img
                draw1 = ImageDraw.Draw(cam1_img)
                text1 = f"Zone {zone_num} Cam 1"
                draw1.text(text_position, text1, font=font, fill=text_color)

                mosaic_image.paste(cam1_img, (x_offset, y_offset_cam1))
        except FileNotFoundError:
            print(f"Warning: Image {cam1_path} not found. Skipping.")
            # Optionally, fill this part of the mosaic with a placeholder color/image
            continue
        except Exception as e:
            print(f"Error processing {cam1_path}: {e}. Skipping.")
            continue

        # Process camera 02
        try:
            cam2_filename = f"{IMAGE_BASENAME}-zone{zone_num:02d}-camera02{IMAGE_EXTENSION}"
            cam2_path = IMAGE_DIR / cam2_filename
            with Image.open(cam2_path) as cam2_img:
                if cam2_img.size != (img_width, img_height):
                    print(f"Warning: Image {cam2_path} has different dimensions. Resizing.")
                    cam2_img = cam2_img.resize((img_width, img_height))

                # Add text to cam2_img
                draw2 = ImageDraw.Draw(cam2_img)
                text2 = f"Zone {zone_num} Cam 2"
                draw2.text(text_position, text2, font=font, fill=text_color)

                mosaic_image.paste(cam2_img, (x_offset, y_offset_cam2))
        except FileNotFoundError:
            print(f"Warning: Image {cam2_path} not found. Skipping.")
            # If cam1 was pasted, this leaves a blank space for cam2
        except Exception as e:
            print(f"Error processing {cam2_path}: {e}. Skipping.")

    # Resize mosaic if it exceeds MAX_DIMENSION
    current_width, current_height = mosaic_image.size
    if current_width > MAX_DIMENSION or current_height > MAX_DIMENSION:
        print(f"Mosaic dimensions ({current_width}x{current_height}) exceed {MAX_DIMENSION}px.")
        if current_width > current_height:
            scale_factor = MAX_DIMENSION / current_width
        else:
            scale_factor = MAX_DIMENSION / current_height

        new_width = int(current_width * scale_factor)
        new_height = int(current_height * scale_factor)

        print(f"Resizing mosaic to {new_width}x{new_height}...")
        try:
            # For Pillow 9.1.0+
            mosaic_image = mosaic_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        except AttributeError:
            # Fallback for older Pillow versions
            mosaic_image = mosaic_image.resize((new_width, new_height), Image.LANCZOS)


    # Save the mosaic image
    output_path = IMAGE_DIR / OUTPUT_FILENAME
    try:
        mosaic_image.save(output_path)
        print(f"Mosaic image saved to {output_path}")
    except Exception as e:
        print(f"Error saving mosaic image: {e}")

if __name__ == "__main__":
    create_mosaic()
