#%%
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd


# %%
df = pd.read_csv('online/E6/P5/data.csv')


#%%
# drop actions column
df = df.drop(columns=['actions'])

# %%
# groupby "environment.zone"

# blow up "action" column which is a str(numpy.array)
def to_numpy(string_list):
    """Converts a string representation of a list of numbers to a NumPy array."""
    if not isinstance(string_list, str):
        return string_list
    try:
        # Remove the brackets and split by space
        numbers_str = string_list.strip('[]').split()
        # Convert the strings to floats and create a NumPy array
        return np.array([float(num) for num in numbers_str])
    except AttributeError:
        return np.nan  # Or handle non-string elements as needed

#  convert action, area, state columns to numpy arrays to float columns
for col in ['action', 'area', 'state']:
    df[col] = df[col].apply(to_numpy)
    for i in range(len(df[col][0])):
        df[f"{col}.{i}"] = df[col].apply(lambda x: x[i] if isinstance(x, np.ndarray) and len(x) > i else np.nan)
    df = df.drop(columns=[col])
#%%
df
#%%


for (name, group_df) in df.groupby("environment.zone"):
    group_df = group_df.reset_index(drop=True)
    path = Path(f"online/E6/P5/Spreadsheet-Poisson{name}/z{name}")
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
    image_file_df = image_file_df[2:]
    # re-index image_file_df to match group_df
    image_file_df = image_file_df.reset_index(drop=True)


    group_df["time"] = pd.to_datetime(group_df["time"], unit='s')

    # merge group_df with image_file_df on indexes
    group_df = pd.merge(group_df, image_file_df, left_index=True, right_index=True)

    #%%
    print(group_df[["time", "image_name"]].head(1).to_dict())

    #%%
    area_cols = group_df.columns[group_df.columns.str.contains('area')]

    # go wide to long area.* should be rows, keep other columns
    df_long = group_df.melt(id_vars=['frame'], value_vars=area_cols,
                    var_name='plant_id', value_name='area')
    # remove area. from area
    df_long['plant_id'] = df_long['plant_id'].str.replace('area.', '').astype(int) + 1

    # drop the area. columns
    group_df = group_df.drop(columns=area_cols.tolist())

    # sort by frame, then by plant_id
    df_long = df_long.sort_values(by=['frame', 'plant_id'])

    merged_df = pd.merge(df_long, group_df, how='left', left_on='frame', right_on='frame')
    merged_df.to_csv(path / 'raw.csv', index=False)
