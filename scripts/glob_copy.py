from pathlib import Path
from shutil import copy

for path in sorted(
    Path("results/data/first_exp/z2cR/").glob("2025-02-28T09*_shape_image.png")
):
    print(path)
    out_path = Path(".") / path.name
    copy(path, out_path)
