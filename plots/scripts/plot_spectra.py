from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot spectra grouped by color with overlaid zones."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("scripts/calibration/spectra"),
        help="Directory containing files named like zone1_blue.txt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots/spectra.png"),
        help="Output PNG path",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.58,
        help="Line alpha for overlaid traces",
    )
    return parser.parse_args()


def read_spectrum(path: Path) -> tuple[list[float], list[float]]:
    wavelengths: list[float] = []
    intensities: list[float] = []
    with path.open() as file_handle:
        next(file_handle, None)
        for line in file_handle:
            parts = line.strip().split()
            if len(parts) >= 2:
                wavelengths.append(float(parts[0]))
                intensities.append(float(parts[1]))
    return wavelengths, intensities


def build_file_index(input_dir: Path) -> dict[tuple[int, str], Path]:
    pattern = re.compile(r"zone(\d+)_(blue|red|white)\.txt$")
    files: dict[tuple[int, str], Path] = {}
    for path in input_dir.glob("zone*_*.txt"):
        match = pattern.match(path.name)
        if match:
            zone = int(match.group(1))
            color = match.group(2)
            files[(zone, color)] = path
    return files


def wavelength_to_rgb(wavelength_nm: float) -> tuple[float, float, float]:
    wavelength = float(wavelength_nm)
    if wavelength < 380 or wavelength > 780:
        return (0.2, 0.2, 0.2)

    if wavelength < 440:
        red = -(wavelength - 440) / (440 - 380)
        green = 0.0
        blue = 1.0
    elif wavelength < 490:
        red = 0.0
        green = (wavelength - 440) / (490 - 440)
        blue = 1.0
    elif wavelength < 510:
        red = 0.0
        green = 1.0
        blue = -(wavelength - 510) / (510 - 490)
    elif wavelength < 580:
        red = (wavelength - 510) / (580 - 510)
        green = 1.0
        blue = 0.0
    elif wavelength < 645:
        red = 1.0
        green = -(wavelength - 645) / (645 - 580)
        blue = 0.0
    else:
        red = 1.0
        green = 0.0
        blue = 0.0

    if wavelength < 420:
        factor = 0.3 + 0.7 * (wavelength - 380) / (420 - 380)
    elif wavelength <= 700:
        factor = 1.0
    else:
        factor = 0.3 + 0.7 * (780 - wavelength) / (780 - 700)

    return (red * factor, green * factor, blue * factor)


def add_em_color_bar(axis: plt.Axes) -> None:
    x_min, x_max = axis.get_xlim()
    wavelengths = np.linspace(x_min, x_max, 600)
    colors = np.array([wavelength_to_rgb(wavelength) for wavelength in wavelengths])[np.newaxis, :, :]

    color_axis = axis.inset_axes([0.0, -0.26, 1.0, 0.08])
    color_axis.imshow(colors, aspect="auto", extent=(x_min, x_max, 0, 1), interpolation="nearest")
    color_axis.set_yticks([])
    color_axis.set_xticks([])
    for spine in color_axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)


def main() -> None:
    args = parse_args()
    files = build_file_index(args.input_dir)

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "figure.titlesize": 18,
        }
    )

    all_zones = sorted({zone for zone, _ in files})
    colors = ["red", "white", "blue"]

    if len(all_zones) < 1:
        raise ValueError(f"No matching zone files found in {args.input_dir}")

    row_groups = [
        ("All", all_zones),
        ("1/5/9", [zone for zone in [1, 5, 9] if zone in all_zones]),
        ("2/6/10", [zone for zone in [2, 6, 10] if zone in all_zones]),
        ("3/7/11", [zone for zone in [3, 7, 11] if zone in all_zones]),
        ("4/8/12", [zone for zone in [4, 8, 12] if zone in all_zones]),
    ]

    zone_palette = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]
    line_styles = ["-", "--", ":", "-."]
    zone_line_colors = {zone: zone_palette[index % len(zone_palette)] for index, zone in enumerate(all_zones)}
    zone_styles = {zone: line_styles[index % len(line_styles)] for index, zone in enumerate(all_zones)}

    fig, axes = plt.subplots(len(row_groups), len(colors), figsize=(18, 20), sharex=True, sharey=True)

    for row_index, (group_label, zones) in enumerate(row_groups):
        for column_index, color in enumerate(colors):
            axis = axes[row_index, column_index]
            for zone in zones:
                spectrum_path = files.get((zone, color))
                if spectrum_path is None:
                    continue

                x_values, y_values = read_spectrum(spectrum_path)
                axis.plot(
                    x_values,
                    y_values,
                    color=zone_line_colors[zone],
                    linestyle=zone_styles[zone],
                    linewidth=1.8,
                    alpha=args.alpha,
                    label=f"Zone {zone}",
                )

            if row_index == 0:
                axis.set_title(color.capitalize())
            if row_index == len(row_groups) - 1:
                axis.set_xlabel("Wavelength (nm)")
            if column_index == 0:
                axis.set_ylabel(f"Intensity\n{group_label}")
            if axis.get_legend_handles_labels()[0]:
                axis.legend(frameon=False)
            if row_index == len(row_groups) - 1:
                add_em_color_bar(axis)

    fig.suptitle("Zone Spectra Overlaid by Color and Subgroup", fontsize=14)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(args.output)


if __name__ == "__main__":
    main()
