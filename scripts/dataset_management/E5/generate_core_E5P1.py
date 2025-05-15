#%%
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from pprint import pprint

ZONE_TO_AGENT = {
    9: "DiscreteRandom",
    2: "ESARSA",
    1: "Spreadsheet-Poisson",
    6: "Spreadsheet-Poisson-slow",
}

RECIPES = {
    "0": [0.1393,0.2667,0.1134,0.0,0.1162,0.2121],
    "1": [0.26865,0.51435,0.2187,0.0,0.2241,0.40905,],
    "2": [0.398,0.762,0.324,0.0,0.332,0.606,],
    "3": [0.657496,1.258824,0.535248,0.0,0.548464,1.001112,],
}

def to_numpy(string_list):
    """Converts a string representation of a list of numbers to a NumPy array."""
    if not isinstance(string_list, str):
        return string_list
    if string_list in list(RECIPES.keys()):
        return np.array(RECIPES[string_list], dtype=float)
    try:
        # Remove the brackets and split by space
        numbers_str = string_list.strip('[]').split()
        # Convert the strings to floats and create a NumPy array
        return np.array([float(num) for num in numbers_str])
    except AttributeError:
        return np.array([])  # Return an empty array for non-string elements


# %%
df = pd.read_csv('/data/online/E5/P1/data.csv')

# Check for NaN values in the DataFrame
print("NaN values per column:")
print(df.isna().sum())


#%%
# drop actions column
df = df[["action", "time", "frame", "environment.zone"]]
# Conert time to UTC
df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
#df = df.sort_values(by='time', ascending=True).reset_index(drop=True)

# %%
#df["action"] = df["action"].apply(to_numpy)
df["action"] = df["action"].apply(to_numpy)
for i in range(len(df["action"][0])):
    df[f"action.{i}"] = df["action"].apply(lambda x: x[i] if isinstance(x, np.ndarray) and len(x) > i else np.nan)
df = df.drop(columns=["action"])
    
#%%

for (name, group_df) in df.groupby("environment.zone"):
    group_df = group_df.reset_index(drop=True)
    #print("\nFirst 5 Entries of Each Column:")
    #print(df.head().to_string(index=False))
    path = Path(f"/data/online/E5/P1/{ZONE_TO_AGENT[name]}/z{name}")
    # images
    image_files = sorted(path.rglob('images/*.jpg'))
    image_file_absolute_paths = [
        str(image_file.absolute()) for image_file in image_files
    ]

    # remove part infront of images/
    image_file_absolute_paths = [
        image_file.split('images/')[1] for image_file in image_file_absolute_paths
    ]

    image_file_df = pd.DataFrame({
        'image_name': image_file_absolute_paths,
    })
    start_image = image_file_df['image_name'][1]
    image_file_df = image_file_df[2:]
    # re-index image_file_df to match group_df
    image_file_df = image_file_df.reset_index(drop=True)
    
    # merge group_df with image_file_df on indexes
    group_df = pd.merge(group_df, image_file_df, left_index=True, right_index=True)
    
    # Create a new row with NaN values except for the image path
    start_row = pd.DataFrame([{col: np.nan if col != 'image_name' else start_image for col in group_df.columns}])
    
    # Concatenate the new row with the existing DataFrame
    group_df = pd.concat([start_row, group_df]).reset_index(drop=True)
    
    '''
    print(f"\nZone {name} shape: {group_df.shape}")
    print("\nFirst 5 rows:")
    print(group_df.head().to_string(index=True))
    print("\nLast 2 rows:")
    print(group_df.tail(2).to_string(index=True))
    print("\n" + "="*80 + "\n")  # Separator for better readability
    '''
    group_df = group_df.drop(columns=["environment.zone"])
    group_df.to_csv(path / 'core.csv', index=False)

# %%
