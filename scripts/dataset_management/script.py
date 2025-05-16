from pathlib import Path

from PIL import Image
from tqdm.contrib.concurrent import thread_map


def convert_png_to_jpg(png_file: Path) -> None:
    """Convert a png file to jpg."""
    jpg_file = png_file.with_suffix(".jpg")
    with Image.open(png_file) as img:
        rgb_img = img.convert("RGB")
        rgb_img.save(jpg_file, "JPEG", quality=90)
    # print(f"Converted {png_file} to {jpg_file}")
    return None


def delete_png_file(png_file: Path) -> None:
    jpg_file = png_file.with_suffix(".jpg")
    if jpg_file.exists():
        png_file.unlink()
        # print(f"Deleted {png_file}")

def main() -> None:
    """Convert all png files recursively in the current directory to jpg."""
    # Get all png files in the current directory and subdirectories
    png_files = sorted(Path("/data/maria_exp").rglob("**/*.png"))
    thread_map(
        convert_png_to_jpg,
        png_files,
        max_workers=64,
        chunksize=1,
    )
    thread_map(
        delete_png_file,
        png_files,
        max_workers=64,
        chunksize=1,
    )


if __name__ == "__main__":
    main()
