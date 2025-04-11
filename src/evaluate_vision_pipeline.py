# plot the pivot table
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.dates import AutoDateLocator, DateFormatter

methods = [
    "baseline",
    "grounded-sam2",
]


# Set the style
sns.set_theme(style="whitegrid")
fig, axs = plt.subplots(len(methods), 1, figsize=(20, 10), sharex=True, sharey=True)
# Set the date format
date_format = DateFormatter("%Y-%m-%d %H:%M")
plt.gca().xaxis.set_major_formatter(date_format)
plt.gca().xaxis.set_major_locator(AutoDateLocator())

for method, ax in zip(methods, axs):
    df = pd.read_csv(f"tmp/{method}/all.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # pivot table col:plant_id, index:timestamp, values:area
    df_pivot = df.pivot_table(
        index="timestamp",
        columns="plant_id",
        values="area",
        aggfunc="mean",
    )


    # Plot the data
    for col in df_pivot.columns:
        ax.plot(df_pivot.index, df_pivot[col], label=col)
    ax.set_title(f"Plant Area Over Time ({method})")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Area")
    ax.tick_params(axis='x', rotation=45)


    # calculate a consistency metric
    # the area of each plant should be consistent over time
    # it is the average difference in plant area between each timestamp
    def calculate_consistency(df_pivot):
        # calculate the difference between each timestamp
        diff = df_pivot.diff().abs()
        # calculate the mean of the differences
        consistency = diff.mean()
        return consistency

    # calculate the consistency metric
    consistency = calculate_consistency(df_pivot)
    # save the consistency metric to a csv file
    consistency.to_csv(f"tmp/{method}/consistency.csv")

    # final number is the consistency metric averaged over all plants
    consistency_final = consistency.mean()
    print(f"Consistency Metric: {consistency_final:.2f}")

fig.tight_layout()
fig.savefig(f"tmp/plant_area_over_time.png")
