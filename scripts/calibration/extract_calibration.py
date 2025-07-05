# %%

# /workspaces/plant-rl/src/environments/PlantGrowthChamber/configs/Plant Chamber Full Calibration.xlsx
import json
import os

import pandas as pd

from utils.calibration import load_and_clean_data

# Load the calibration data
# one sheet per zone
# transpose the sheet
calibration_file = (
    "/workspaces/plant-rl/scripts/calibration/Plant Chamber Full Calibration.xlsx"
)

maximum = {}
safe_maximum = {}

for zone in range(1, 13):
    zone_str = f"zone{zone:02}"
    df = pd.read_excel(
        calibration_file,
        sheet_name=zone_str,
        header=None,
    )
    df = df.transpose()
    df.columns = df.iloc[0]  # Set the first row as the header
    df = df[1:]  # Remove the first row
    df.reset_index(drop=True, inplace=True)

    # Get Far Red integrals
    _, integrals = load_and_clean_data(
        os.path.join(
            os.path.dirname(calibration_file),
            f"RL_FarRedIntensityz{zone}.txt",
        )
    )
    print(integrals)

    df["far_red"] = df["Action"].map(integrals)  # type: ignore

    # Rename columns to lower snake case
    df.columns = [col.replace(" ", "_").lower() for col in df.columns]

    # Convert the DataFrame to a dictionary with lists as values
    df_dict = df.to_dict(orient="list")  # type: ignore
    # if entire column is NaN, set it to None
    df_dict = {k: None if all(pd.isna(v)) else v for k, v in df_dict.items()}
    print(df_dict)

    actions = df_dict.keys()

    actions_without_action = [action for action in actions if action not in ["action"]]

    # Store the maximum values for each action
    for action in actions_without_action:
        if action not in maximum:
            maximum[action] = 0
        values = df_dict[action]
        if values is None:
            continue
        # remove NaN values
        values = [v for v in values if not pd.isna(v)]
        max_value = max(values)
        maximum[action] = max(maximum[action], max_value)
        if action not in safe_maximum:
            safe_maximum[action] = max_value
        else:
            safe_maximum[action] = min(safe_maximum[action], max_value)

    # Save maximum to a JSON file
    # /workspaces/plant-rl/src/environments/PlantGrowthChamber/configs/calibration.json
    max_file_path = "/workspaces/plant-rl/src/environments/PlantGrowthChamber/configs/calibration.json"
    with open(max_file_path, "w") as max_file:
        out = {
            "maximum": maximum,
            "safe_maximum": safe_maximum,
        }
        json.dump(out, max_file, indent=4)

    # Update the calibrations in alliance-zone*.json
    alliance_zone_file = f"/workspaces/plant-rl/src/environments/PlantGrowthChamber/configs/alliance-{zone_str}.json"
    with open(alliance_zone_file, "r") as file:
        alliance_zone = json.load(file)
    alliance_zone["zone"]["calibration"] = df_dict
    with open(alliance_zone_file, "w") as file:
        json.dump(alliance_zone, file, indent=4)

# %%
