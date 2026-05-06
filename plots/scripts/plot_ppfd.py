#!/usr/bin/env python3
"""Plot PPFD heatmaps (Elec, Elec-105) from ppfd.csv."""

import os
import sys
import numpy as np
import pandas as pd

# ensure src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)
import matplotlib.pyplot as plt
import seaborn as sns


def read_ppfd(path):
    df = pd.read_csv(path, skip_blank_lines=True)
    df = df.dropna(how="all")
    # Ensure numeric
    df["ZONE"] = df["ZONE"].astype(int)
    df["PPFD_ELEC"] = pd.to_numeric(df["PPFD_ELEC"])
    return df


def pivot_by_color(df, value_col):
    # Keep color order R,W,B
    order = ["R", "W", "B"]
    pivot = df.pivot(index="ZONE", columns="COLOR", values=value_col)
    pivot = pivot.reindex(columns=order)
    pivot = pivot.sort_index()
    return pivot


def make_heatmaps(df, out_path):
    elec = pivot_by_color(df, "PPFD_ELEC")
    elec_minus_105 = elec - 105.0

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    sns.set(style="whitegrid", font_scale=1.0)
    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax_hist = fig.add_subplot(gs[1, :])

    delta = elec.values.astype(float) - 105.0
    max_abs_global = float(np.nanmax(np.abs(delta)))
    vmin_global = 105.0 - max_abs_global
    vmax_global = 105.0 + max_abs_global

    annot_elec = elec.copy().astype(object)
    for r in elec.index:
        for c in elec.columns:
            annot_elec.at[r, c] = f"{int(round(elec.at[r, c]))}"

    sns.heatmap(
        elec,
        annot=annot_elec,
        fmt="",
        cmap="RdYlBu_r",
        center=105,
        vmin=vmin_global,
        vmax=vmax_global,
        cbar=False,
        ax=ax0,
    )
    ax0.set_title("PPFD (ELEC)")

    maxabs_e105 = float(np.nanmax(np.abs(elec_minus_105.values.astype(float))))
    sns.heatmap(
        elec_minus_105,
        annot=True,
        fmt="+.0f",
        cmap="RdYlBu_r",
        center=0,
        vmin=-maxabs_e105,
        vmax=maxabs_e105,
        cbar=False,
        ax=ax1,
    )
    ax1.set_title("PPFD (ELEC) - 105")

    # Histogram of PPFD (ELEC) - 105 by color
    colors_map = {"R": "tab:red", "W": "grey", "B": "tab:blue"}
    combined_vals = []
    for color in ["R", "W", "B"]:
        vals = elec_minus_105[color].dropna().values.astype(float)
        if vals.size > 0:
            combined_vals.append(vals)
    if combined_vals:
        combined = np.concatenate(combined_vals)
        minv = int(np.floor(np.nanmin(combined)))
        maxv = int(np.ceil(np.nanmax(combined)))
        if maxv == minv:
            minv -= 1
            maxv += 1
        bin_edges = np.arange(minv, maxv + 1, 1)
    else:
        minv, maxv = -5, 5
        bin_edges = np.arange(minv, maxv + 1, 1)

    means_elec105 = {}
    for color in ["R", "W", "B"]:
        vals105 = elec_minus_105[color].dropna().values.astype(float)
        if vals105.size > 0:
            ax_hist.hist(
                vals105,
                bins=bin_edges,
                alpha=0.6,
                label=color,
                color=colors_map[color],
                align="left",
            )
            q1 = float(np.percentile(vals105, 25))
            q3 = float(np.percentile(vals105, 75))
            iqr = q3 - q1
            if iqr == 0:
                filtered = vals105
            else:
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                filtered = vals105[(vals105 >= lower) & (vals105 <= upper)]
                if filtered.size == 0:
                    filtered = vals105
            means_elec105[color] = float(np.mean(filtered))
        else:
            means_elec105[color] = float("nan")
    ax_hist.axvline(0, color="k", linewidth=0.8)
    for color in ["R", "W", "B"]:
        m = means_elec105[color]
        if not np.isnan(m):
            ax_hist.axvline(m, color=colors_map[color], linestyle="--", linewidth=1.2)
    ax_hist.set_title("PPFD (ELEC) - 105 Distribution")
    ax_hist.set_xlabel("PPFD (ELEC) - 105")
    ax_hist.set_ylabel("Count")
    leg = ax_hist.legend(title="Color", frameon=False)
    if leg is not None:
        leg.set_frame_on(False)
    try:
        xticks = np.arange(minv, maxv + 1, 1)
        ax_hist.set_xticks(xticks)
        ax_hist.set_xlim(minv - 0.5, maxv + 0.5)
    except Exception:
        pass

    for a in [ax0, ax1, ax_hist]:
        a.grid(False)

    try:
        fig.colorbar(ax0.collections[0], ax=[ax0], label="PPFD")
    except Exception:
        pass
    try:
        fig.colorbar(ax1.collections[0], ax=[ax1], label="Δ PPFD")
    except Exception:
        pass

    print("Means (PPFD ELEC - 105) by color (after IQR outlier exclusion):")
    for c in ["R", "W", "B"]:
        v = means_elec105.get(c, float("nan"))
        if np.isnan(v):
            print(f"  {c}: NaN")
        else:
            print(f"  {c}: {v:.2f}")

    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


def main():
    src = "scripts/calibration/ppfd.csv"
    out = "plots/ppfd.png"
    df = read_ppfd(src)
    make_heatmaps(df, out)


if __name__ == "__main__":
    main()
