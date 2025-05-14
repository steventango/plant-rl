
import shutil
from itertools import chain
from pathlib import Path

# phytochrome_exp/images/<z11c2>--2023-01-01--22-50-01.jpg
# ->
# phytochrome_exp/z11c2/images/2023-01-01--22-50-01.jpg

paths = chain(
    Path("phytochrome_exp/images/").glob("*.jpg"),
    Path("nazmus_exp/images/").glob("*.jpg"),
)

for path in paths:
    # Get the name of the image
    name = path.stem.split("--")[0]
    # Create the new directory
    new_dir = Path(path.parent.parent / name / path.parent.name)
    new_dir.mkdir(parents=True, exist_ok=True)
    # Move the image to the new directory
    print(f"Moving {path} to {new_dir / path.name}")
    shutil.move(str(path), str(new_dir / path.name))
