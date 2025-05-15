import shutil
from pathlib import Path

paths = sorted(Path("/data/maria_exp").glob("*/*.jpg"))

for path in paths:
    # path: /data/maria_exp/z3c1/zone03cam01-2024-02-05-15-50-01.png
    # new path: /data/maria_exp/z3c1/images/2024-02-05-15-50-01.jpg
    new_path = path.parent / "images" / path.name
    new_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Moving {path} to {new_path}")
    shutil.move(str(path), str(new_path))
