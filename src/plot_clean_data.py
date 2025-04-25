from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    datasets = Path("/data/plant-rl/online/E6/P5").glob("Spreadsheet-Poisson*/z*")
    datasets = sorted(datasets)
    datasets = datasets[2:]
    pipeline_version = "v3.3.1"
    
    a = 0.1
    b = 0.3
    for dataset in datasets:
        csv_path = dataset / "processed" / pipeline_version / "clean.csv"
        plot_path = dataset / "processed" / pipeline_version / "plots" / "area.jpg"

        plot_path.parent.mkdir(exist_ok=True, parents=True)
        df = pd.read_csv(csv_path)
        df["time"] = pd.to_datetime(df["time"])
        plant_ids = df["plant_id"].unique()
        n_plants = len(plant_ids)
        # Add one for the overall mean subplot
        fig, axes = plt.subplots(n_plants + 1, 1, figsize=(20, 4 * (n_plants + 1)), sharex=True)
        if n_plants == 1:
            axes = [axes]
        # Plot mean area over all plants at the top
        ax_mean = axes[0]
        # Group by time and compute mean of area and clean_area
        mean_df = df.groupby("time").agg({"area": "mean", "clean_area": "mean"}).reset_index()
        ax_mean.plot(mean_df["time"], mean_df["area"], label="Mean Area", color="tab:blue")
        ax_mean.plot(
            mean_df["time"], mean_df["clean_area"], label="Mean Cleaned Area", color="tab:green", linestyle="--"
        )
        ax_mean.set_title(f"Mean Area Over All Plants ({dataset.name})")
        ax_mean.set_ylabel("Mean Area")
        ax_mean.legend()
        ax_mean.tick_params(axis="x", rotation=45)
        sns.despine(ax=ax_mean)
        # Individual plant subplots
        # sort plant_ids by final area
        plant_ids = sorted(plant_ids, key=lambda x: df[df["plant_id"] == x]["area"].iloc[-1], reverse=True)
        for ax, plant_id in zip(axes[1:], plant_ids):
            plant_df = df[df["plant_id"] == plant_id]
            ax.plot(plant_df["time"], plant_df["area"], label=f"Plant {plant_id} Area", alpha=0.5)
            ax.plot(plant_df["time"], plant_df["clean_area"], label="Cleaned Area", color="tab:green", linestyle="--")
            ax.plot(plant_df["time"], plant_df["mean"], label="Mean", color="tab:orange", linestyle="--")
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
            ax.tick_params(axis="x", rotation=45)
            sns.despine(ax=ax)
        plt.xlabel("Time")
        plt.tight_layout()
        plt.savefig(plot_path)
        print(plot_path)


if __name__ == "__main__":
    main()
