# %%
import pandas as pd

# %%
df_online = pd.read_csv("/workspaces/plant-rl/experiments/online/E6/P3/data.csv")
print(df_online)

# %%
df_offline = pd.read_csv("/workspaces/plant-rl/experiments/offline/E6/P3/data.csv")
print(df_offline)

# %%
print(df_online.columns)

# %%
# join the two dataframes on the timestamp column
df_combined = pd.merge(
    df_online,
    df_offline,
    on=["steps", "environment.zone"],
    suffixes=("_online", "_offline"),
    how="left",
)


# replace state, area, reward with the offline values
for column in ["state", "area", "reward"]:
    df_combined[column + "_online"] = df_combined[column + "_offline"]
    df_combined = df_combined.drop(column + "_offline", axis=1)

# drop the suffixes from the column names
df_combined.columns = df_combined.columns.str.replace("_online", "")

df_combined = df_combined[df_combined["environment.zone"].isin([1, 2])].reset_index(
    drop=True
)

# save the combined dataframe to a csv file
df_combined.to_csv(
    "/workspaces/plant-rl/experiments/offline/E6/P3/data_combined.csv", index=False
)
# %%
