import shutil
from pathlib import Path

paths = Path("/data/maria_exp_png/maria_exp/").glob("**/*.jpg")
new_dir = Path("/data/maria_exp")
new_dir.mkdir(parents=True, exist_ok=True)

for path in paths:
    new_path = new_dir / path.parts[-2] / "images" / path.name
    print(f"Moving {path} to {new_dir / path.name}")
    shutil.move(str(path), str(new_dir / path.name))
