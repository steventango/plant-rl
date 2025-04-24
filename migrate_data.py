from itertools import chain
from pathlib import Path

from PIL import Image
from tqdm.contrib.concurrent import thread_map


def migrate(path: Path, dry_run: bool = False) -> None:
    # new_path = path.relative_to("data")
    # new_path = Path("/data") / path
    new_path = path.parent / "images" / path.name
    new_path = new_path.with_suffix(".jpg")

    if not new_path.parent.exists():
        new_path.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print(f"Dry run: {path} would be migrated to {new_path}")
        return

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
        Path("/data/maria_exp").glob("**/*.png"),
        Path("/data/nazmus_exp").glob("**/*.png"),
        Path("/data/phytochrome_exp").glob("**/*.png"),
    )
)

thread_map(
    migrate,
    paths,
    max_workers=80,
)
