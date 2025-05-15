import pandas as pd
from pathlib import Path
import pytz
import warnings
from datetime import datetime, time

def to_time(val):
    if isinstance(val, datetime):
        return val.time()
    elif isinstance(val, time):
        return val
    else:
        warnings.warn(f"Ignoring invalid time entry: {val} (type: {type(val)})", RuntimeWarning)
        return pd.NaT  # or raise ValueError if you'd rather crash


# Define the directories
base_dir = Path("/data/phytochrome_exp")      # directory with z*c* subdirs
recipe_dir = Path("/data/phytochrome_exp/phytochrome_recipes") # directory with z*-prefixed .xlsx files

# Loop through subdirectories in base_dir matching z*c* pattern
for subdir in base_dir.glob("z*c*"):
    if not subdir.is_dir():
        continue

    # Load raw.csv from the subdirectory
    raw_csv_path = subdir / "raw.csv"
    if not raw_csv_path.exists():
        continue
    raw_df = pd.read_csv(raw_csv_path)

    # Extract z* from the subdirectory name
    z_part = subdir.name.split('c')[0]  # Assumes format like 'z12c3' -> 'z12'

    # Find corresponding xlsx file in dir_x
    matching_xlsx_files = list(recipe_dir.glob(f"{z_part}*.xlsx"))
    if not matching_xlsx_files:
        print(f"No XLSX file found for {z_part}")
        continue

    xlsx_path = matching_xlsx_files[0]
    recipe_df = pd.read_excel(xlsx_path)

    # Now df_csv and df_xlsx are available for use
    print(f"Loaded {raw_csv_path} and {xlsx_path}")
    

    # Convert raw_df time from string to datetime (UTC)
    raw_df["time"] = pd.to_datetime(raw_df["time"], utc=True)
    raw_df = raw_df.sort_values("time")

    # Rename recipe_df columns before processing
    recipe_df = recipe_df.rename(
        columns=
        {
            "Blue": "action.0", 
            "Cool_White": "action.1",
            "Warm_White": "action.2",
            "Orange_Red": "action.3",
            "Red": "action.4",
            "Far_Red": "action.5",
        }
    )
    recipe_df["Time"] = recipe_df["Time"].apply(to_time)
    

    # Align recipe_df["Time"] with dates from raw_df
    tz = pytz.timezone("America/Edmonton")
    recipe_records = []

    for date in raw_df["time"].dt.date.unique():
        full_timestamps = [
            tz.localize(datetime.combine(date, t)) for t in recipe_df["Time"]
        ]
        full_timestamps = pd.Series(full_timestamps).dt.tz_convert("UTC")

        recipe_day = pd.DataFrame({
            "time_recipe": full_timestamps,
            "action.0": recipe_df["action.0"].values,
            "action.1": recipe_df["action.1"].values,
            "action.2": recipe_df["action.2"].values,
            "action.3": recipe_df["action.3"].values,
            "action.4": recipe_df["action.4"].values,
            "action.5": recipe_df["action.5"].values,})
        recipe_records.append(recipe_day)

    # Combine all dated recipe records
    recipe_times = pd.concat(recipe_records).sort_values("time_recipe")

    # Merge the recipe info into raw_df using backward fill on time
    merged_df = pd.merge_asof(
        raw_df,
        recipe_times,
        left_on="time",
        right_on="time_recipe",
        direction="backward"
    ).drop(columns=["time_recipe"])
    column_order = ["time", "frame", "action.0", "action.1", "action.2", "action.3", "action.4", "action.5", "image_name"]
    merged_df = merged_df[column_order]  # Reorder columns
    
    action_cols = ["action.0", "action.1", "action.2", "action.3", "action.4", "action.5"]
    merged_df[action_cols] = merged_df[action_cols].fillna(0)

    merged_df.to_csv(subdir / "core.csv", index=False)
