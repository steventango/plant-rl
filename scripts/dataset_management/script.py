# convert all png files recursively in the current directory to jpg

from pathlib import Path


from PIL import Image


def convert_png_to_jpg(png_file: Path) -> None:
    """Convert a png file to jpg."""
    jpg_file = png_file.with_suffix(".jpg")
    with Image.open(png_file) as img:
        rgb_img = img.convert("RGB")
        rgb_img.save(jpg_file, "JPEG")
    print(f"Converted {png_file} to {jpg_file}")
    return None


def main() -> None:
    """Convert all png files recursively in the current directory to jpg."""
    # Get all png files in the current directory and subdirectories
    png_files = Path("online/E6").rglob("*.png")
    for png_file in png_files:
        # if jpg file already exists, delete the png file
        jpg_file = png_file.with_suffix(".jpg")
        if jpg_file.exists():
            png_file.unlink()
            print(f"Deleted {png_file}")
            continue
        convert_png_to_jpg(png_file)
    return None


if __name__ == "__main__":
    main()
