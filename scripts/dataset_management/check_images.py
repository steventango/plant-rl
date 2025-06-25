import os
from pathlib import Path

from PIL import Image

images_paths = Path("/data/maria_exp/z8c2/images").glob("*.jpg")

valid_count = 0
invalid_images = []

# Loop through all images matching the pattern
for img_path in images_paths:
    try:
        # Try to open the image with PIL
        with Image.open(img_path) as img:
            # Verify the image by loading it
            img.verify()
            valid_count += 1
            print(f"Valid image: {os.path.basename(img_path)}")
    except Exception as e:
        # If any exception occurs, the image is considered invalid
        invalid_images.append(img_path)
        print(f"Invalid image: {os.path.basename(img_path)}, Error: {str(e)}")

# Print summary
print("\nSummary:")
print(f"Total images checked: {valid_count + len(invalid_images)}")
print(f"Valid images: {valid_count}")
print(f"Invalid images: {len(invalid_images)}")

if invalid_images:
    print("\nList of invalid images:")
    for img in invalid_images:
        print(f"- {img}")
