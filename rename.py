import shutil
from datetime import datetime, timedelta
from pathlib import Path

# Define the directory containing the images
image_dirs = [
    Path('old/1/results/online/E4/P0/Spreadsheet/0/images'),
    Path('old/2/results/online/E4/P0/Spreadsheet/0/images'),
    Path('old/3/results/online/E4/P0/Spreadsheet/0/images'),
    Path('results/online/E4/P0/Spreadsheet/0/images'),
]
output_dir = Path('data/first_exp/z2cR')
output_dir.mkdir(exist_ok=True, parents=True)

output_dir2 = Path('/workspaces/PlantVision/Pipeline/first_exp/z2cR')
output_dir2.mkdir(exist_ok=True, parents=True)

def round_seconds(obj: datetime) -> datetime:
    if obj.microsecond >= 500_000:
        obj += timedelta(seconds=1)
    return obj.replace(microsecond=0)

# Iterate over all PNG files in the directory
for image_dir in image_dirs:
    for image_path in sorted(image_dir.glob('*.png')):
        # Get the file creation time
        time = image_path.stat().st_mtime
        time = datetime.fromtimestamp(time)
        time = round_seconds(time)
        timestamp = datetime.isoformat(time)
        timestamp = timestamp.replace(':', '')
        new_path = output_dir / f'{timestamp}.png'

        # Copy the file to data/first_exp with the new name
        if new_path.exists():
            # hash compare the two
            if new_path.stat().st_size != image_path.stat().st_size:
                print(f'{new_path} already exists but is different')
            else:
                pass
        else:
            # print(f'Renamed {image_path} to {new_path}')
            shutil.copy(image_path, new_path)

        if time.minute % 5 == 0:
            new_path2 = output_dir2 / f'{timestamp}.png'
            shutil.copy(image_path, new_path2)
