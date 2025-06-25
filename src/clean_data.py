from pathlib import Path

import pandas as pd

from utils.metrics import UnbiasedExponentialMovingAverage as UEMA


def clean_area(df, a=0.2, b=0.2):
    df["clean_area"] = df["area"].copy()
    df["mean"] = 0.0
    prev_area = 0
    for _plant_id, plant_df in df.groupby("plant_id"):
        uema = UEMA(alpha=0.1)
        for j, (i, row) in enumerate(plant_df.iterrows()):
            area = row["area"]
            mean = uema.compute()
            df["mean"].at[i] = mean.item()
            if j > 36 and (area < (1 - a) * mean or area > (1 + b) * mean):
                df.at[i, "clean_area"] = prev_area
            else:
                prev_area = area
            if area > 0:
                uema.update(area)
    return df


def main():
    datasets = Path("/data").glob("nazmus_exp/z11c1")
    datasets = sorted(datasets)

    pipeline_version = "v3.6.0"
    a = 0.1
    b = 0.3
    for dataset in datasets:
        csv_path = dataset / "processed" / pipeline_version / "all.csv"
        cleaned_csv_path = dataset / "processed" / pipeline_version / "clean.csv"

        df = pd.read_csv(csv_path)

        # if has nan print warning and fillna with 0
        if df["area"].isna().any():
            print(f"Warning: NaN values found in {csv_path}. Filling with 0.")
            df["area"] = df["area"].fillna(0)

        df = clean_area(df, a=a, b=b)

        df.to_csv(cleaned_csv_path, index=False)
        print(f"Cleaned data saved to {cleaned_csv_path}")


if __name__ == "__main__":
    main()
