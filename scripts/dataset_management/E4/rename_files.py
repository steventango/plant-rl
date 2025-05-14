# /workspaces/plant-rl/old/results/online/E4/P0/Spreadsheet/0/images
# rename all the files with isoformat of the date

import os
import shutil
from datetime import datetime
from pathlib import Path

from tqdm import tqdm


def get_file_date(file_path):
    """Get file's modification date as a datetime object"""
    mod_time = os.path.getmtime(file_path)
    return datetime.fromtimestamp(mod_time)

def main():
    # Directory containing the files to rename
    source_dir = Path("/workspaces/plant-rl/old/results/online/E4/P0/Spreadsheet/0/images")

    # Get all image files
    image_files = list(source_dir.glob('**/*.png')) + list(source_dir.glob('**/*.jpg'))
    image_files = sorted(image_files, key=lambda x: x.name)
    print(f"Found {len(image_files)} image files")

    renamed_count = 0
    errors = 0

    # Process all files with a progress bar
    for file_path in tqdm(image_files, desc="Renaming files"):
        try:
            # Get file extension
            file_ext = file_path.suffix

            # Get file's modification date in ISO format
            file_date = get_file_date(file_path)
            file_date = file_date.replace(microsecond=0)
            iso_date = file_date.isoformat().replace(':', '')

            # Create new file name with ISO date
            new_filename = f"{iso_date}{file_ext}"
            new_filepath = file_path.parent / new_filename

            # Check if destination file already exists
            counter = 1
            while new_filepath.exists():
                new_filename = f"{iso_date}_{counter}{file_ext}"
                new_filepath = file_path.parent / new_filename
                counter += 1

            # Rename the file
            shutil.move(file_path, new_filepath)
            renamed_count += 1

        except Exception as e:
            print(f"Error renaming {file_path}: {e}")
            errors += 1

    print(f"Renamed {renamed_count} files. Encountered {errors} errors.")

if __name__ == "__main__":
    main()
