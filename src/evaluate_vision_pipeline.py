# plot the pivot table
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.dates import AutoDateLocator, DateFormatter

methods = [
    "baseline",
]

for method in methods:
    df = pd.read_csv(f"tmp/{method}/all.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # pivot table col:plant_id, index:timestamp, values:area
    df_pivot = df.pivot_table(
        index="timestamp",
        columns="plant_id",
        values="area",
        aggfunc="mean",
    )

    # Set the style
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(20, 10))
    # Set the date format
    date_format = DateFormatter("%Y-%m-%d %H:%M")
    plt.gca().xaxis.set_major_formatter(date_format)
    plt.gca().xaxis.set_major_locator(AutoDateLocator())

    # Plot the data
    for col in df_pivot.columns:
        plt.plot(df_pivot.index, df_pivot[col], label=col)
    plt.legend()
    plt.title("Plant Area Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Area")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("tmp/baseline/plant_area_over_time.png")

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
    consistency.to_csv("tmp/baseline/consistency.csv")
    # plot the consistency metric
    plt.figure(figsize=(20, 10))
    plt.plot(consistency.index, consistency.values)
    plt.title("Consistency Metric")
    plt.xlabel("Timestamp")
    plt.ylabel("Consistency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("tmp/baseline/consistency_metric.png")

    # final number is the consistency metric averaged over all plants
    consistency_final = consistency.mean()
    print(f"Consistency Metric: {consistency_final:.2f}")
