# %%

# /workspaces/plant-rl/src/environments/PlantGrowthChamber/configs/Plant Chamber Full Calibration.xlsx
import json
import os

import numpy as np
import pandas as pd


def get_far_red_integrals(zone: int, calibration_file_path: str) -> pd.Series:
    """Reads the spectral data for a given zone, calculates the integral for each action,
    and returns a Series with the integrals indexed by action."""
    spectral_file = os.path.join(
        os.path.dirname(calibration_file_path), f"RL_FarRedIntensityz{zone}.txt"
    )
    if not os.path.exists(spectral_file):
        print(
            f"Warning: Spectral file for zone {zone} not found. Falling back to zone 12."
        )
        spectral_file = os.path.join(
            os.path.dirname(calibration_file_path), "RL_FarRedIntensityz12.txt"
        )

    df = pd.read_csv(spectral_file, sep="\t", header=0)

    # Clean the DataFrame
    df = df.iloc[:, :-1]  # Drop last column
    df = df.drop(df.columns[1], axis=1)  # Drop second column
    df = df.drop(df.columns[range(1, df.shape[1], 2)], axis=1)  # Drop odd columns

    # Rename columns
    action_values = [
        0.10,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.85,
        0.90,
        0.95,
        1.00,
    ]
    df.columns = ["Wavelength"] + action_values

    # Drop 0.10
    df = df.drop(columns=[0.10])
    action_values.remove(0.10)

    # Calculate integrals
    integrals = {}
    for action in action_values:
        integrals[action] = np.trapz(df[action], df["Wavelength"])

    return pd.Series(integrals)


# Load the calibration data
# one sheet per zone
# transpose the sheet
calibration_file = (
    "/workspaces/plant-rl/scripts/calibration/Plant Chamber Full Calibration.xlsx"
)

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
    far_red_integrals = get_far_red_integrals(zone, calibration_file)
    df["Far Red"] = df["Action"].map(far_red_integrals).fillna(0)  # type: ignore

    # Convert the DataFrame to a dictionary with lists as values
    df_dict = df.to_dict(orient="list")  # type: ignore
    print(df_dict)

    # Update the calibrations in alliance-zone*.json
    alliance_zone_file = f"/workspaces/plant-rl/src/environments/PlantGrowthChamber/configs/alliance-{zone_str}.json"
    with open(alliance_zone_file, "r") as file:
        alliance_zone = json.load(file)
    alliance_zone["zone"]["calibration"] = df_dict
    with open(alliance_zone_file, "w") as file:
        json.dump(alliance_zone, file, indent=4)

# %%
