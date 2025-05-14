import pandas as pd

all_df = pd.read_csv("/data/nazmus_exp/z11c1/processed/v3.6.0/all.csv")

raw_df = pd.read_csv("/data/nazmus_exp/z11c1/raw.csv")

# join on frame, replace all_df["time"] with raw_df["time"]
all_df["time"] = all_df.merge(raw_df, on="image_name", how="left")["time_y"]

# to csv
all_df.to_csv("/data/nazmus_exp/z11c1/processed/v3.6.0/all.csv", index=False)
