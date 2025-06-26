import os
import shutil

# Define source and destination directories
source_dir = "/data/online/E4/P0.1/z2/images"
destination_dir = "/data/online/E4/P0.2/z2/images"
threshold_filename = "2025-02-24T102103.jpg"

# Ensure destination directory exists
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)
    print(f"Created destination directory: {destination_dir}")

# Get all files in source directory
files = os.listdir(source_dir)

# Filter files that meet or exceed the threshold
files_to_move = [f for f in files if f >= threshold_filename]

# Move the filtered files
moved_count = 0
for filename in files_to_move:
    source_path = os.path.join(source_dir, filename)
    dest_path = os.path.join(destination_dir, filename)

    # Check if it's a file (not a directory)
    if os.path.isfile(source_path):
        shutil.move(source_path, dest_path)
        moved_count += 1
        print(f"Moved: {filename}")

# Print summary
print(
    f"\nOperation complete: {moved_count} files moved from {source_dir} to {destination_dir}"
)
print(f"Files moved had filenames >= {threshold_filename}")
