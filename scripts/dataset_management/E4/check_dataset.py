import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

dataset_paths = [
    "/data/online/E4/P0.2/z2",
    "/data/online/E4/P1/z2"
]

for dataset_path in dataset_paths:
    core_path = dataset_path + "/core.csv"
    df = pd.read_csv(core_path)
    # check if the image_name file exists on disk
    df["image_exists"] = df["image_name"].apply(lambda x: os.path.exists(os.path.join(dataset_path, "images", x)))
    for image_name, exists in zip(df["image_name"], df["image_exists"]):
        if not exists:
            print(f"Image {image_name} does not exist in {dataset_path}/images")

    print(f"\nChecking for action changes in {dataset_path}...")

    # Create directory for saving comparison images
    comparison_dir = os.path.join(dataset_path, "action_change_comparisons")
    if not os.path.exists(comparison_dir):
        os.makedirs(comparison_dir)

    # Check for action.0 changes between consecutive rows
    if 'action.0' in df.columns:
        action_changes = []
        for i in range(1, len(df)):
            if df['action.0'].iloc[i] != df['action.0'].iloc[i-1]:
                action_changes.append(i)

        print(f"Found {len(action_changes)} action.0 changes")

        # Create side-by-side image comparisons for each change
        for idx in action_changes:
            prev_img_name = df['image_name'].iloc[idx-1]
            curr_img_name = df['image_name'].iloc[idx-1+1]  # Current image

            prev_img_path = os.path.join(dataset_path, "images", prev_img_name)
            curr_img_path = os.path.join(dataset_path, "images", curr_img_name)

            # Check if both images exist
            if os.path.exists(prev_img_path) and os.path.exists(curr_img_path):
                prev_img = Image.open(prev_img_path)
                curr_img = Image.open(curr_img_path)

                # Create the comparison plot
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))

                # Plot the images
                axes[0].imshow(np.array(prev_img))
                axes[0].set_title(f"Before: {prev_img_name}\naction.0: {df['action.0'].iloc[idx-1]}")
                axes[0].axis('off')

                axes[1].imshow(np.array(curr_img))
                axes[1].set_title(f"After: {curr_img_name}\naction.0: {df['action.0'].iloc[idx]}")
                axes[1].axis('off')

                # Add overall title with index information
                plt.suptitle(f"Action.0 change at index {idx}", fontsize=16)

                # Save the figure
                comparison_filename = f"action_change_{idx}_{prev_img_name.split('.')[0]}_to_{curr_img_name.split('.')[0]}.png"
                plt.savefig(os.path.join(comparison_dir, comparison_filename))
                plt.close(fig)
                print(f"  Saved comparison for index {idx}: {comparison_filename}")
            else:
                print(f"  Could not create comparison for index {idx}: image files not found")
    else:
        print("Column 'action.0' not found in dataset")
