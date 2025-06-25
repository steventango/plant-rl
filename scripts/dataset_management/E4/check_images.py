import concurrent.futures
import hashlib
import logging
from pathlib import Path

from tqdm import tqdm

first_dir = Path("/workspaces/plant-rl/old/results/online/E4/P0/Spreadsheet/0/images")
second_dir = Path("/data/online/E4/P0/z2/images")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='image_check.log'
)

def calculate_hash(file_path):
    """Calculate the MD5 hash of an image file."""
    try:
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5()
            chunk = f.read(8192)
            while chunk:
                file_hash.update(chunk)
                chunk = f.read(8192)
        return file_hash.hexdigest()
    except Exception as e:
        logging.error(f"Error calculating hash for {file_path}: {e}")
        return None

def main():
    # Get all image files from second directory
    second_dir_images = list(second_dir.glob('**/*.jpg'))
    logging.info(f"Found {len(second_dir_images)} images in second directory")

    # Create hash table for second directory images in parallel
    hash_table = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Map hash calculation to each file
        future_to_file = {executor.submit(calculate_hash, file_path): file_path for file_path in second_dir_images}

        # Process results as they complete with a progress bar
        for future in tqdm(concurrent.futures.as_completed(future_to_file),
                          total=len(future_to_file),
                          desc="Hashing second directory"):
            file_path = future_to_file[future]
            try:
                file_hash = future.result()
                if file_hash:
                    hash_table[file_hash] = file_path
            except Exception as e:
                logging.error(f"Exception for {file_path}: {e}")

    logging.info(f"Created hash table with {len(hash_table)} entries")

    # Process first directory images
    first_dir_images = list(first_dir.glob('**/*.png')) + list(first_dir.glob('**/*.jpg'))
    logging.info(f"Found {len(first_dir_images)} images in first directory")

    removed_count = 0
    not_found_count = 0

    # Add progress bar for first directory image processing
    for img_path in tqdm(first_dir_images, desc="Processing first directory"):
        img_hash = calculate_hash(img_path)
        if img_hash in hash_table:
            # Image exists in second directory, remove from first
            # os.remove(img_path)
            logging.info(f"Removed duplicate: {img_path} (matches {hash_table[img_hash]})")
            removed_count += 1
        else:
            # Image not found in second directory
            logging.warning(f"No match found for: {img_path}")
            not_found_count += 1

    logging.info(f"Summary: {removed_count} images removed, {not_found_count} images not found in second directory")
    print(f"Summary: {removed_count} images removed, {not_found_count} images not found in second directory")

if __name__ == "__main__":
    main()
