from itertools import chain
from pathlib import Path

from PIL import Image
from tqdm.contrib.concurrent import thread_map


def migrate(path: Path):
    new_path = path.relative_to("data")
    new_path = Path("/data") / new_path
    new_path = new_path.parent / "images" / new_path.name
    new_path = new_path.with_suffix(".jpg")

    if not new_path.parent.exists():
        new_path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.open(path)
    img = img.convert("RGB")
    img.save(new_path, "JPEG", quality=90)
    img.close()

    img = Image.open(new_path)
    img.verify()
    img.close()

    path.unlink()


paths = sorted(
    chain(
        Path("data/online/").glob("**/*.png"),
        Path("data/online/").glob("**/*.jpg"),
    )
)

thread_map(
    migrate,
    paths,
    max_workers=80,
)
