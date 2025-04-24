from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


def main():
    datasets = Path("/data/plant-rl/online/E6/P5").glob("Spreadsheet-Poisson*/z*")
    datasets = sorted(datasets)
    datasets = datasets[2:]
    pipeline_version = "v3.3.1"
    for dataset in datasets:
        out_dir = dataset / "processed" / pipeline_version
        out_dir_images = out_dir / "images"
        # stack images
        out_dir_image_paths = sorted(out_dir_images.rglob("*_shape_image.jpg"))
        process_map(
            generate_image_stack,
            out_dir_image_paths,
        )



def generate_image_stack(image_path):
    boxes_path = image_path.parent / (image_path.name.replace("_shape_image.jpg", "_boxes.jpg"))
    masks2_path = image_path.parent / (image_path.name.replace("_shape_image.jpg", "_masks2.jpg"))

    images = []
    for path in [boxes_path, masks2_path, image_path]:
        img = Image.open(path)
        images.append(img)

    # stack images
    stacked_image = Image.new("RGB", (images[0].width * len(images), images[0].height))
    for i, img in enumerate(images):
        stacked_image.paste(img, (i * img.width, 0))

    isoformat = image_path.stem.split("_")[0]
    # add YYYY-MM-DD HH:MM:SS to the top of the image
    draw = ImageDraw.Draw(stacked_image)
    # make font bigger
    font = ImageFont.load_default(size=36)
    center_x = (stacked_image.width) // 2
    draw.text((center_x, 36), isoformat, fill="white", font=font)

    # save stacked image
    stacked_image_path = image_path.parent / (image_path.name.replace("_shape_image.jpg", "_stacked.jpg"))
    stacked_image.save(stacked_image_path)


if __name__ == "__main__":
    main()
