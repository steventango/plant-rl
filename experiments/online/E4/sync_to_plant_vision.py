import shutil
from datetime import datetime, timedelta
from pathlib import Path

image_dir = Path("data/first_exp/z2cR")

output_dir2 = Path("/workspaces/PlantVision/Pipeline/first_exp/z2cR")
output_dir2.mkdir(exist_ok=True, parents=True)


def round_seconds(obj: datetime) -> datetime:
    if obj.microsecond >= 500_000:
        obj += timedelta(seconds=1)
    return obj.replace(microsecond=0)


def main():
    for image_path in sorted(image_dir.glob("*.png")):
        time = image_path.stat().st_mtime
        time = datetime.fromtimestamp(time)
        time = round_seconds(time)
        timestamp = datetime.isoformat(time)
        timestamp = timestamp.replace(":", "")

        if time.minute % 5 == 0:
            new_path2 = output_dir2 / f"{timestamp}.png"
            shutil.copy(image_path, new_path2)


if __name__ == "__main__":
    main()
