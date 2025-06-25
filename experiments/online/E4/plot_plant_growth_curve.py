# %%
# %%
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

image_dir = Path("data/first_exp/z2cR")

times = []
blue = []
brightness = []


def process_image(image):
    try:
        img = Image.open(image)
        img = np.array(img)
        blue_ratio_img = img[:, :, 2] / (img.sum(axis=2) + 1e-10)
        blue_value = blue_ratio_img.mean()
        brightness_value = img.sum() / img.size
        time = datetime.fromisoformat(image.stem)
        return time, blue_value, brightness_value
    except Exception:
        return None


# results = process_map(process_image, sorted(image_dir.glob("*.png")))

# for result in results:
#     if result is not None:
#         time, blue_value, brightness_value = result
#         times.append(time)
#         blue.append(blue_value)
#         brightness.append(brightness_value)

# df = pd.DataFrame(
#     {
#         "time": times,
#         "blue": blue,
#         "brightness": brightness,
#     }
# )

# df["time"] = pd.to_datetime(df["time"])
# # drop time before 2025
# df = df[df["time"] > "2025-01-01"]

# df.plot(x="time", y="blue", marker=".", figsize=(9, 6), label="Blue Channel")
# plt.legend()
# # plt.show()


# #%%
# df["light_on"] = df["blue"] < .5

# # %%
# df.to_csv("data/first_exp/z2cR.csv", index=False)

# %%
# import pandas as pd

# df = pd.read_csv("data/first_exp/z2cR.csv")
# df["time"] = pd.to_datetime(df["time"])

# import matplotlib.pyplot as plt

# # filter data for Mar 1st
# # cond = ((df["time"].dt.month == 3) & (df["time"].dt.day == 1)) | ((df["time"].dt.month == 3) & (df["time"].dt.day == 2))
# # df = df[cond]

# df.plot(x="time", y="blue", marker=".", figsize=(72, 6), label="Blue Channel")
# # show 9 am 0 minutes as xticks
# cond = (df["time"].dt.minute == 0)
# plt.xticks(
#     ticks=df["time"][cond],
#     # format MM-DD HH:MM
#     labels=df["time"][cond].dt.strftime("%m-%d %H:%M"),
#     rotation=90,
# )

# %%


df = pd.read_csv(Path(__file__).parent / "new_area_over_time.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df

# %%
# if value < 10, set to nan, for non timestamp columns
for column in df.columns:
    if column == "timestamp":
        continue
    df.loc[df[column] < 10, column] = pd.NA

df = df.dropna()

# %%

# %%
# plot all columns as lines onto the same plot


def plot_segments(df, x_column, y_column, ax, color, label=None, gap_indices=None):
    """
    Plot a column as a line with possible breaks for large time gaps.

    Args:
        df: DataFrame containing the data
        x_column: Column name for x-axis (typically timestamp)
        y_column: Column name for y-axis to plot
        ax: Matplotlib axis to plot on
        color: Color for the line
        label: Label for the legend (default: None)
        gap_indices: List of indices where gaps occur (default: None)
    """
    # If no gaps, plot as a single line
    if not gap_indices:
        sns.lineplot(
            data=df,
            x=x_column,
            y=y_column,
            ax=ax,
            label=label,
            legend=False,
            color=color,
        )
    else:
        # Plot segments separately
        start_idx = 0
        for gap_idx in gap_indices:
            segment = df.loc[start_idx : gap_idx - 1]
            if not segment.empty:
                sns.lineplot(
                    data=segment,
                    x=x_column,
                    y=y_column,
                    ax=ax,
                    label=label if start_idx == 0 else None,
                    legend=False,
                    color=color,
                )
            start_idx = gap_idx

        # Plot the final segment
        final_segment = df.loc[start_idx:]
        if not final_segment.empty:
            sns.lineplot(
                data=final_segment,
                x=x_column,
                y=y_column,
                ax=ax,
                label=None,
                color=color,
                legend=False,
            )


# Create subplots with shared x-axis
fig, axs = plt.subplots(2, 1, figsize=(60, 6), sharex=True)
ax = axs[1]
# Sort DataFrame by timestamp to ensure proper ordering
df = df.sort_values("timestamp")

# %%

# Calculate time differences between consecutive points
df["time_diff"] = df["timestamp"].diff()

# Identify large gaps (e.g., gaps > 6 hours)
threshold = pd.Timedelta(minutes=6)
large_gaps = df["time_diff"] > threshold

# %%
# Get the indices where large gaps occur
gap_indices = df.index[large_gaps].tolist()

palette = sns.color_palette("hls", len(df.columns) - 2)

for column in df.columns:
    if column in ["timestamp", "time_diff"]:
        continue
    if column == "iqm":
        color = "black"
    else:
        color = palette.pop(0)
    plot_segments(
        df, "timestamp", column, ax, color, label=column, gap_indices=gap_indices
    )

area_columns = [
    column for column in df.columns if column not in ["timestamp", "time_diff"]
]
ax.set_ylim(0, df[area_columns].quantile(0.999).max())
ax.set_ylabel("Area (mm$^2$)")
ax.set_title("Area Over Time")
# plt.legend()
# %%


df2 = pd.read_csv(
    Path(__file__).parent.parent.parent.parent / "data/first_exp/z2cR.csv"
)
df2["time"] = pd.to_datetime(df2["time"])
df2["light_on"] = df2["light_on"].astype(int)

# %%


# Sort DataFrame by time to ensure proper ordering
df2 = df2.sort_values("time")

# Calculate time differences between consecutive points
df2["time_diff"] = df2["time"].diff()

# Identify large gaps (e.g., gaps > 1 hour)
threshold = pd.Timedelta(minutes=6)
large_gaps_df2 = df2["time_diff"] > threshold

# Get the indices where large gaps occur
gap_indices_df2 = df2.index[large_gaps_df2].tolist()

# Use plot_segments to plot light_on data
plot_segments(
    df2,
    "time",
    "light_on",
    axs[0],
    color="blue",
    label="Light On",
    gap_indices=gap_indices_df2,
)
axs[0].set_title("Policy")
axs[0].set_ylabel("Action (Light On)")
axs[0].set_ylim(-0.1, 1.1)  # For binary data, setting appropriate y-limits

# Find common time range for both plots
min_time = max(df["timestamp"].min(), df2["time"].min())
max_time = min(df["timestamp"].max(), df2["time"].max())

# Set both axes to the same range
for ax in axs:
    ax.set_xlim(min_time, max_time)

# Format the x-axis dates nicely
fig.autofmt_xdate()

# add major grid lines for every day
for ax in axs:
    ax.xaxis.grid(True, which="major")
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))

    ax.xaxis.grid(True, which="minor")
    import matplotlib.dates as mdates

    ax.xaxis.set_minor_locator(
        mdates.HourLocator(byhour=[1, 2, 9, 13, 14, 16, 17, 18, 19, 21], interval=1)
    )
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H"))
    ax.tick_params(axis="x", which="minor", labelrotation=90)


# Set x-label only on the bottom subplot
plt.xlabel("Time")

# rotate x-axis labels
plt.xticks(rotation=90)

# Adjust layout to make room for x-axis labels
plt.tight_layout()

# %%
# plt.show()
fig.savefig(Path(__file__).parent / "area_over_time_and_policy.png")
# fig.show()
# plt.savefig("area_over_time_and_policy.png")

# %%
