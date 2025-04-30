from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image


def main():
    datasets = Path("/data/plant-rl/online/E6/P5").glob("Spreadsheet-Poisson*/z*")
    pipeline_version = "v3.3.0"
    for dataset in datasets:
        csv_path = dataset / "processed" / pipeline_version / "all.csv"
        plot_path = dataset / "processed" / pipeline_version / "plots" / "area.pdf"

        # Plot the data
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 14), sharex=True)

        plot_path.parent.mkdir(exist_ok=True, parents=True)
        df = pd.read_csv(csv_path)
        df["time"] = pd.to_datetime(df["time"])
        # pivot table col:plant_id, index:timestamp, values:area
        df_pivot = df.pivot_table(
            index="time",
            columns="plant_id",
            values="area_y",
            aggfunc="mean",
        )
        for col in df_pivot.columns:
            areas = df_pivot[col]
            ax1.plot(df_pivot.index, areas, label=col)
        ax1.set_title(f"Plant Area Over Time ({dataset.name})")
        ax1.set_ylabel("Area")
        ax1.legend()
        ax1.tick_params(axis="x", rotation=45)
        sns.despine(ax=ax1)

        # Second subplot: plot action.0 column
        if "action.0" in df.columns:
            ax2.plot(df["time"], df["action.0"], label="action.0", color="tab:orange")
            ax2.set_ylabel("action.0")
            ax2.set_title("Action 0 Over Time")
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, "action.0 column not found", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_xlabel("Timestamp")
        ax2.tick_params(axis="x", rotation=45)
        sns.despine(ax=ax2)
        plt.tight_layout()
        plt.savefig(plot_path)
        print(plot_path)


if __name__ == "__main__":
    main()
