#!/usr/bin/env python3
"""
Plot mean clean area vs time for each experiment and zone.
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(
        description="Plot mean clean area vs time for each experiment and zone."
    )
    parser.add_argument(
        "--parquet",
        "-p",
        default="/data/offline/cleaned_offline_dataset_continuous.parquet",
        help="Path to parquet file",
    )
    parser.add_argument(
        "--out",
        "-o",
        default="clean_area_plots",
        help="Output directory or prefix for plots",
    )
    parser.add_argument(
        "--show", action="store_true", help="Show the plots interactively"
    )
    args = parser.parse_args()

    try:
        logging.info(f"Reading parquet: {args.parquet}")
        df = pl.read_parquet(args.parquet)
        logging.info(f"Rows after reading: {df.shape[0]}")
        if df.shape[0] > 0:
            logging.info(
                f"Time range in data: {df['time'].min()} to {df['time'].max()}"
            )
    except Exception as e:
        logging.error(f"Failed to read parquet: {e}")
        sys.exit(1)

    # Check if required columns exist
    required_cols = ["experiment", "zone", "time", "mean_clean_area"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logging.error("Missing required columns: %s", missing_cols)
        logging.info("Available columns: %s", df.columns)
        sys.exit(1)

    # Group by experiment, zone, time and compute average mean_clean_area
    logging.info("Grouping data by experiment, zone, and time...")
    grouped = (
        df.group_by(["experiment", "zone", "time"])
        .agg(pl.col("mean_clean_area").mean().alias("avg_mean_clean_area"))
        .sort(["experiment", "zone", "time"])
    )

    # Get unique experiments
    experiments = sorted(grouped["experiment"].unique())

    # Create output directory if it doesn't exist
    out_path = Path(args.out)
    if len(experiments) > 1 or out_path.suffix:
        # If multiple experiments or out has extension, treat as directory
        out_path.mkdir(parents=True, exist_ok=True)
        out_prefix = out_path / "clean_area_E"
        out_suffix = ".png"
    else:
        # Single experiment, treat as prefix
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_prefix = str(out_path) + "_E"
        out_suffix = ".png"

    sns.set(style="whitegrid")

    for exp in experiments:
        logging.info(f"Plotting for experiment {exp}")
        exp_data = grouped.filter(pl.col("experiment") == exp)

        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=exp_data.to_pandas(),
            x="time",
            y="avg_mean_clean_area",
            hue="zone",
            palette="tab10",
        )

        plt.title(f"Mean Clean Area vs Time for Experiment {exp}")
        plt.xlabel("Time")
        plt.ylabel("Average Mean Clean Area")
        plt.xticks(rotation=45)
        plt.tight_layout()

        out_file = f"{out_prefix}{exp}{out_suffix}"
        plt.savefig(out_file, dpi=200)
        logging.info("Saved plot to %s", out_file)

        if args.show:
            plt.show()
        else:
            plt.close()


if __name__ == "__main__":
    main()
