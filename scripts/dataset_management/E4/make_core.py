# %%
import numpy as np
import pandas as pd

df = pd.read_csv("/data/online/E4/P1/z2/images/processed/v1.0.0/E4_policy.csv")
df["light_on"] = df["blue"] < 0.5


# %%
# drop index, blue, brightness columns
df = df.drop(columns=["Unnamed: 0", "blue", "brightness"])

# %%
# localize timestamp
df["time"] = pd.to_datetime(df["time"])
df["time"] = df["time"].dt.tz_localize("America/Edmonton")
# %%
# convert to UTC
df["time"] = df["time"].dt.tz_convert("UTC")
# %%
threshold_datetime = (
    pd.to_datetime("2025-02-24T102103")
    .tz_localize("America/Edmonton")
    .tz_convert("UTC")
)
df = df[df["time"] >= threshold_datetime]

# %%

light_on_action_map = {
    False: [0, 0, 0, 0, 0, 0],
    True: [0.199, 0.381, 0.162, 0, 0.166, 0.303],
}
df["actions"] = df["light_on"].map(light_on_action_map)
for i in range(6):
    df[f"action.{i}"] = df["actions"].apply(lambda x: x[i])
df = df.drop(columns=["actions", "light_on"])

# %%


df["image_name"] = (
    df["time"].dt.tz_convert("America/Edmonton").dt.strftime("%Y-%m-%dT%H%M%S") + ".jpg"
)

# %%

# split into two dataframes
split_datetime = (
    pd.to_datetime("2025-03-03T222605")
    .tz_localize("America/Edmonton")
    .tz_convert("UTC")
)
df1 = df[df["time"] < split_datetime].copy()
df1["frame"] = np.arange(len(df1))
df2 = df[df["time"] >= split_datetime].copy()
df2["frame"] = np.arange(len(df2))


# save df1 and df2 to csv
df1.to_csv("/data/online/E4/P0.2/z2/core.csv", index=False)

df2.to_csv("/data/online/E4/P1/z2/core.csv", index=False)

# %%
