from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    datasets = Path("/data").glob("nazmus_exp/z11c1")
    datasets = sorted(datasets)
    pipeline_version = "v3.6.0"

    a = 0.1
    b = 0.3
    for dataset in datasets:
        csv_path = dataset / "processed" / pipeline_version / "clean.csv"
        plot_path = dataset / "processed" / pipeline_version / "plots" / "area.jpg"

        plot_path.parent.mkdir(exist_ok=True, parents=True)
        df = pd.read_csv(csv_path)

        # Convert time column to datetime if it's not already
        # Check if time column contains datetime strings
        if isinstance(df["time"].iloc[0], str) and any(
            c in df["time"].iloc[0] for c in ["-", "/", "T", ":"]
        ):
            df["time"] = pd.to_datetime(df["time"])
        # If time column is numeric and represents days, convert to datetime
        elif pd.api.types.is_numeric_dtype(df["time"]):
            # Assuming time values are days since some reference point
            # Create a datetime range
            base_date = datetime(2025, 4, 1)  # Use a reference date
            df["time"] = pd.to_datetime(
                [base_date + pd.Timedelta(days=t) for t in df["time"]]
            )

        plant_ids = df["plant_id"].unique()
        n_plants = len(plant_ids)
        # Add one for the overall mean subplot
        fig, axes = plt.subplots(
            n_plants + 1, 1, figsize=(20, 4 * (n_plants + 1)), sharex=True
        )
        if n_plants == 1:
            axes = [axes]

        # Plot mean area over all plants at the top
        ax_mean = axes[0]
        # Group by time and compute mean of area and clean_area, ensure proper sorting
        mean_df = (
            df.groupby("time").agg({"area": "mean", "clean_area": "mean"}).reset_index()
        )
        ax_mean.plot(
            mean_df["time"], mean_df["area"], label="Mean Area", color="tab:blue"
        )
        ax_mean.plot(
            mean_df["time"],
            mean_df["clean_area"],
            label="Mean Cleaned Area",
            color="tab:green",
        )
        ax_mean.set_title(f"Mean Area Over All Plants ({dataset.name})")
        ax_mean.set_ylabel("Mean Area")
        ax_mean.legend()

        # Configure x-axis with 1 major tick per day and 1 minor tick per hour
        ax_mean.xaxis.set_major_locator(mdates.DayLocator())
        ax_mean.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax_mean.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
        ax_mean.tick_params(axis="x", rotation=45)
        ax_mean.grid(True, which="major", axis="x", linestyle="-")
        ax_mean.grid(True, which="minor", axis="x", linestyle=":", alpha=0.5)
        sns.despine(ax=ax_mean)

        # Individual plant subplots
        # sort plant_ids by final area
        plant_ids = sorted(
            plant_ids,
            key=lambda x: df[df["plant_id"] == x]["area"].iloc[-1],  # type: ignore
            reverse=True,
        )
        for ax, plant_id in zip(axes[1:], plant_ids, strict=False):
            plant_df = df[df["plant_id"] == plant_id]
            ax.plot(
                plant_df["time"],
                plant_df["mean"],
                label="Mean",
                color="tab:orange",
                linestyle="--",
            )
            ax.plot(
                plant_df["time"],
                plant_df["area"],
                label=f"Plant {plant_id} Area",
                alpha=0.5,
            )
            ax.plot(
                plant_df["time"],
                plant_df["clean_area"],
                label="Cleaned Area",
                color="tab:green",
            )
            mean = plant_df["mean"]
            ax.fill_between(
                plant_df["time"],
                (1 - a) * mean,
                (1 + b) * mean,
                color="tab:orange",
                alpha=0.2,
                label=f"Mean - {a * 100}%, Mean + {b * 100}%",
            )
            ax.set_title(f"Plant {plant_id} Area")
            ax.set_ylabel("Area")
            ax.legend()

            # Apply same x-axis tick settings to all subplots
            ax.xaxis.set_major_locator(mdates.DayLocator())
            ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
            ax.tick_params(axis="x", rotation=45)
            ax.grid(True, which="major", axis="x", linestyle="-")
            ax.grid(True, which="minor", axis="x", linestyle=":", alpha=0.5)
            sns.despine(ax=ax)

        plt.xlabel("Time")
        plt.tight_layout()
        plt.savefig(plot_path)
        print(plot_path)


if __name__ == "__main__":
    main()
