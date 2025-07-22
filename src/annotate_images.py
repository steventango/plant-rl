import io
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# Set non-interactive backend before importing pyplot
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm.contrib.concurrent import process_map

# Use non-interactive backend to avoid X server errors
matplotlib.use("Agg")


def main():
    datasets = Path("/data/online/E10/P1").glob("**/alliance-zone*")
    datasets = sorted(datasets)
    pipeline_version = "v3.6.1"

    for dataset in datasets:
        out_dir = dataset / "processed" / pipeline_version
        out_dir_images = out_dir / "images"
        df = pd.read_csv(out_dir / "all.csv")

        # Group by time and plant_id to ensure areas match with correct plants
        plant_df = (
            df.groupby(["time", "plant_id"])
            .agg(
                area=("area", "first"),
            )
            .reset_index()
        )

        # Create a pivot table with time as index and plant_id as columns
        pivot_df = plant_df.pivot(index="time", columns="plant_id", values="area")

        # Ensure all plant_ids are present in each row (fill NaN with 0)
        pivot_df = pivot_df.fillna(0)

        print(pivot_df.head())

        # Get the image name for each timestamp
        image_names = df.groupby("time")["image_name"].first()

        # Get the timestamps we're working with
        timestamps = pivot_df.index[:]

        image_paths = process_map(
            annotate_image,
            timestamps,
            [out_dir_images] * len(timestamps),
            [pivot_df] * len(timestamps),
            [image_names] * len(timestamps),
            chunksize=1,
        )
        print(image_paths[-1])


def annotate_image(timestamp, out_dir_images, pivot_df, image_names):
    # Get the image name for this timestamp
    image_name = image_names.loc[timestamp]
    image_path = out_dir_images / image_name
    isoformat_og = image_path.stem.split("_")[0]
    masks2_path = image_path.parent / (isoformat_og + "_masks2.jpg")

    image = Image.open(masks2_path)

    # add YYYY-MM-DD HH:MM:SS to the top of the image
    draw = ImageDraw.Draw(image)
    # make font bigger
    font = ImageFont.load_default(size=36)
    x = image.width - (image.width) // 4
    # convert isoformat to Edmonton timezone
    edmonton_tz = ZoneInfo("America/Edmonton")
    dt = datetime.fromisoformat(isoformat_og).astimezone(edmonton_tz)
    isoformat = dt.strftime("%Y-%m-%d %H:%M:%S")
    draw.text((x, 10), isoformat, fill="white", font=font)

    # Calculate mean area directly from pivot_df for this timestamp
    mean_area = pivot_df.loc[timestamp].mean()
    text = f"Mean Area: {mean_area:.1f} mm²"
    draw.text((10, 10), text, fill="white", font=font)

    # Create a new time series figure
    time_series_fig = plt.figure(figsize=(20, 15))

    # Get plant IDs from pivot_df
    plant_ids = pivot_df.columns.tolist()
    num_plants = len(plant_ids)

    # Calculate layout parameters
    cols = 6
    rows = 3
    max_plants = rows * cols

    # Create a sliding window for the time series
    # Convert all timestamps to datetime objects with Edmonton timezone
    edmonton_tz = ZoneInfo("America/Edmonton")
    all_times = []
    for time_str in pivot_df.index:
        dt_time = datetime.fromisoformat(time_str).astimezone(edmonton_tz)
        all_times.append(dt_time)

    # Find the index of the current timestamp
    current_time_idx = pivot_df.index.get_loc(timestamp)

    # Create a sliding window centered on the current timestamp
    window_size = 720  # Show 720 timestamps before and after current timestamp
    if current_time_idx - window_size < 0:
        start_idx = 0
        end_idx = 2 * window_size
    elif current_time_idx + window_size >= len(pivot_df.index):
        start_idx = len(pivot_df.index) - 2 * window_size
        end_idx = current_time_idx + window_size + 1
    else:
        start_idx = current_time_idx - window_size
        end_idx = current_time_idx + window_size + 1
    start_idx = max(0, start_idx)
    end_idx = min(len(pivot_df.index), end_idx)

    # Get the windowed data
    windowed_times = all_times[start_idx:end_idx]
    windowed_df = pivot_df.iloc[start_idx:end_idx]

    # Create grid spec for layout: 3 rows x 6 columns for plants, plus 1 row for mean
    gs = time_series_fig.add_gridspec(rows + 1, cols, height_ratios=[1, 1, 1, 0.8])

    # Create individual plant plots in a grid
    num_plant_plots = min(num_plants, max_plants)  # Cap at max_plants

    for i in range(num_plant_plots):
        row = i // cols
        col = i % cols

        # Create a subplot in the figure
        ax = time_series_fig.add_subplot(gs[row, col])

        # Plot the plant data
        plant_id = plant_ids[i]
        plant_areas = windowed_df[plant_id].values
        ax.plot(windowed_times, plant_areas, label=f"Plant {plant_id}")
        ax.set_title(f"Plant {plant_id}", pad=5)
        ax.grid(True)

        # Only show x labels for bottom row
        if row == rows - 1:
            ax.set_xlabel("Time")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        else:
            ax.set_xticklabels([])

        # Only show y labels for leftmost column
        if col == 0:
            ax.set_ylabel("Area (mm²)")

        # Add vertical line at current time
        if timestamp in windowed_df.index:
            current_idx = windowed_df.index.get_loc(timestamp)
            current_time = windowed_times[current_idx]
            ax.axvline(
                x=current_time,
                color="red",
                linestyle="--",
                linewidth=2,
            )

    # Create the mean area plot that spans all columns at the bottom
    ax_mean = time_series_fig.add_subplot(gs[rows, :])

    # Plot mean data
    mean_areas = windowed_df.mean(axis=1).values
    ax_mean.plot(
        windowed_times,
        mean_areas,
        "b-",
        linewidth=2,
        label="Mean Plant Area",
    )

    # Add vertical line at current time
    if timestamp in windowed_df.index:
        current_idx = windowed_df.index.get_loc(timestamp)
        current_time = windowed_times[current_idx]
        ax_mean.axvline(
            x=current_time,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Current Time",
        )

    # Set formatting
    ax_mean.set_xlabel("Time")
    ax_mean.set_ylabel("Mean Area (mm²)")
    ax_mean.set_title("Mean Plant Area Over Time")
    ax_mean.grid(True)
    ax_mean.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    plt.setp(ax_mean.xaxis.get_majorticklabels(), rotation=45)
    ax_mean.legend(loc="upper right")

    plt.tight_layout()

    # Convert the matplotlib figure to a PIL Image
    buf = io.BytesIO()
    time_series_fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(time_series_fig)
    buf.seek(0)
    plot_img = Image.open(buf)

    # Resize the plot to fit the width of the original image
    plot_width = image.width
    plot_height = int(plot_img.height * (plot_width / plot_img.width))

    # Ensure height is even for video encoding compatibility
    if plot_height % 2 != 0:
        plot_height += 1

    plot_img = plot_img.resize((plot_width, plot_height))

    # Create a new image with space for the plot at the bottom
    new_height = image.height + plot_height

    # Ensure the total height is even (for video encoding compatibility)
    if new_height % 2 != 0:
        new_height += 1

    new_image = Image.new("RGB", (image.width, new_height), (255, 255, 255))

    # Paste the original image and the plot
    new_image.paste(image, (0, 0))
    new_image.paste(plot_img, (0, image.height))

    # save annotated image
    image_path = image_path.parent / (isoformat_og + "_annotated.jpg")
    new_image.save(image_path)
    return image_path


if __name__ == "__main__":
    main()
