"""Analyze the intensity/PPFD vs power relationship for the E18/P0.1 sweep.

Per-row alignment: within a single CSV row, ``action.*`` / ``calibrated_action.*``
and ``power`` are paired (both refer to the same 5-min interval). The
``agent_action`` column lags by one schedule step relative to ``action.*``
because it is the wrapper's *next* polled scalar.

Two data anomalies handled here:

1. **Zone02 lightbar stuck.** After the first repeat (step 119+),
   ``action.*`` and ``calibrated_action.*`` froze at the s_max values from
   step 115 while ``agent_action`` advanced. Those rows are dropped.
2. **Zone02 pre-experiment first row.** The first recorded row captured a
   stale 69 W reading from before the lightbar was commanded off. Dropped.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

DATA_ROOT = Path("/data/plant-rl/online/E18/P0.1")
OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

ZONES = {
    "zone01": DATA_ROOT / "Sequence1/alliance-zone01/raw_2026-05-14.csv",
    "zone02": DATA_ROOT / "Sequence2/alliance-zone02/raw_2026-05-14.csv",
}

USECOLS = [
    "time",
    "frame",
    "steps",
    "action.0",
    "action.1",
    "action.2",
    "action.3",
    "action.4",
    "action.5",
    "calibrated_action.0",
    "calibrated_action.1",
    "calibrated_action.2",
    "calibrated_action.3",
    "calibrated_action.4",
    "calibrated_action.5",
    "agent_action",
    "power",
    "voltage",
    "current",
]

BALANCED_SHARES = np.array([19.5, 71.53, 7.82, 0.0, 6.15, 0.0])
TOTAL_BALANCED = BALANCED_SHARES.sum()

S_MAX = 90.0 / 71.53
LEVELS_PER_REPEAT = 21
LEVEL_S = np.arange(LEVELS_PER_REPEAT) * S_MAX / (LEVELS_PER_REPEAT - 1)
LEVEL_PPFD = LEVEL_S * TOTAL_BALANCED

# safe_min activation thresholds (s value) for each balanced-105 channel
ACTIVATION = {
    "cool_white": 5 / 71.53,
    "blue": 5 / 19.50,
    "warm_white": 5 / 7.82,
    "red": 5 / 6.15,
}

COLORS = {"zone01": "tab:blue", "zone02": "tab:orange"}
MARKERS = {0: "o", 1: "x", 2: "^", 3: "s"}


def load_zone(label: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=USECOLS)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    df["req_ppfd"] = df[[f"action.{i}" for i in range(5)]].sum(axis=1)
    df["drive"] = df[[f"calibrated_action.{i}" for i in range(6)]].sum(axis=1)

    # Detect "stuck lightbar": req_ppfd no longer tracks agent_action × 105.
    # Normal wrapper lag is at most one schedule step (~6.6 PPFD); anything
    # beyond that means the actuator stopped responding.
    expected = df["agent_action"] * TOTAL_BALANCED
    stuck_mask = (df["req_ppfd"] - expected).abs() > 15
    if bool(stuck_mask.any()):
        first_stuck = int(df.index[stuck_mask][0])
        print(
            f"  {label}: stuck at row {first_stuck} "
            f"(t={df.loc[first_stuck, 'time']}); dropping from there"
        )
        df = df.iloc[:first_stuck].copy()

    # Drop pre-experiment stale readings: lights-off action (drive ≈ 0) but
    # power well above baseline means the lightbar hadn't transitioned yet.
    pre_exp = (df["drive"] < 0.01) & (df["power"] > 20)
    if bool(pre_exp.any()):
        n = int(pre_exp.sum())
        print(f"  {label}: dropping {n} pre-experiment stale row(s)")
        df = df.loc[~pre_exp].copy()

    df = df.dropna(subset=["power"]).reset_index(drop=True)
    return df


def assign_level(df: pd.DataFrame) -> pd.DataFrame:
    req = df["req_ppfd"].to_numpy()
    diffs = np.abs(req[:, None] - LEVEL_PPFD[None, :])
    idx = diffs.argmin(axis=1)
    df["level_idx"] = idx
    df["s"] = LEVEL_S[idx]
    df["nominal_ppfd"] = LEVEL_PPFD[idx]
    df["snap_err"] = req - df["nominal_ppfd"]
    return df


def assign_repeat(df: pd.DataFrame) -> pd.DataFrame:
    diffs = df["level_idx"].diff().fillna(0)
    df["repeat"] = (diffs <= -10).cumsum().astype(int)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["repeat", "level_idx"], as_index=False).agg(
        s=("s", "first"),
        nominal_ppfd=("nominal_ppfd", "first"),
        mean_power=("power", "mean"),
        std_power=("power", "std"),
        n=("power", "size"),
        mean_voltage=("voltage", "mean"),
        mean_current=("current", "mean"),
        mean_drive=("drive", "mean"),
    )
    return pd.DataFrame(grouped)


def plot_power_vs_x(
    summaries: dict[str, pd.DataFrame], x_col: str, x_label: str, fname: str
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, summ in summaries.items():
        for rep_val, sub in summ.groupby("repeat"):
            rep = int(rep_val)  # type: ignore[arg-type]
            sub = sub.sort_values(x_col)
            ax.errorbar(
                sub[x_col],
                sub["mean_power"],
                yerr=sub["std_power"],
                fmt=MARKERS.get(rep, "s"),
                color=COLORS[label],
                label=f"{label} r{rep + 1} (n={int(sub['n'].sum())})",
                capsize=3,
                alpha=0.85,
            )
    threshold_x = {"nominal_ppfd": TOTAL_BALANCED, "s": 1.0}[x_col]
    for name, s in ACTIVATION.items():
        ax.axvline(
            s * threshold_x, color="gray", linestyle="--", linewidth=0.7, alpha=0.5
        )
        ax.text(
            s * threshold_x,
            ax.get_ylim()[1] * 0.95,
            f" {name} on",
            rotation=90,
            fontsize=7,
            color="gray",
            alpha=0.8,
            va="top",
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel("plug power (W)")
    ax.set_title(f"E18/P0.1: plug power vs {x_label}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname)
    plt.close(fig)


def main() -> None:
    summaries: dict[str, pd.DataFrame] = {}
    rows: dict[str, pd.DataFrame] = {}
    for label, path in ZONES.items():
        print(f"loading {label}: {path}")
        df = load_zone(label, path)
        df = assign_level(df)
        df = assign_repeat(df)
        rows[label] = df
        summaries[label] = summarize(df)

    plot_power_vs_x(
        summaries, "nominal_ppfd", "nominal balanced PPFD", "power_vs_ppfd.pdf"
    )
    plot_power_vs_x(summaries, "s", "intensity scalar s", "power_vs_s.pdf")

    print("\n--- Model fits over LEDs-on region only (drive > 0, levels 2..20) ---")
    print(
        "    Below that (levels 0,1) every channel is safe_min-gated; power = baseline."
    )

    def model_linear(x: np.ndarray, a: float, b: float) -> np.ndarray:
        return a + b * x

    def model_quadratic(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        return a + b * x + c * x**2

    def model_power_law(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        # P = a + b * PPFD^c. Captures concave/convex curvature with one shape param.
        return a + b * np.power(np.clip(x, 1e-6, None), c)

    def model_lin_plus_exp(
        x: np.ndarray, a: float, b: float, c: float, tau: float
    ) -> np.ndarray:
        # Linear in PPFD plus a decaying overhead from sub-spectrum drive.
        return a + b * x + c * np.exp(-x / max(tau, 1e-6))

    candidates = [
        ("linear         (2p)", model_linear, [7.0, 0.5], ["a", "b"]),
        ("quadratic      (3p)", model_quadratic, [7.0, 0.5, -1e-4], ["a", "b", "c"]),
        ("power-law      (3p)", model_power_law, [5.0, 0.5, 1.0], ["a", "b", "c"]),
        (
            "lin+exp decay  (4p)",
            model_lin_plus_exp,
            [3.0, 0.45, 5.0, 30.0],
            ["a", "b", "c", "tau"],
        ),
    ]

    # Baseline: the levels where every channel is gated below safe_min and
    # the lightbar draws nothing — power = chamber idle load.
    baselines: dict[str, float] = {}
    for label, summ in summaries.items():
        off = summ[summ["level_idx"] <= 1]
        baselines[label] = float(off["mean_power"].mean()) if len(off) else float("nan")

    fit_results: dict[str, dict[str, tuple[np.ndarray, float]]] = {}
    for label, summ in summaries.items():
        on = summ[summ["level_idx"] >= 2]
        x = on["nominal_ppfd"].to_numpy()
        y = on["mean_power"].to_numpy()
        print(
            f"\n{label}  baseline={baselines[label]:.3f} W   "
            f"fit n={len(x)} levels (drive > 0)"
        )
        fit_results[label] = {}
        for name, fn, p0, pnames in candidates:
            try:
                popt, _ = curve_fit(fn, x, y, p0=p0, maxfev=20000)
            except Exception as e:
                print(f"  {name}: fit failed ({e})")
                continue
            pred = fn(x, *popt)
            rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
            param_str = ", ".join(
                f"{n}={v:.4g}" for n, v in zip(pnames, popt, strict=False)
            )
            print(f"  {name}: RMSE={rmse:.3f} W  |  {param_str}")
            fit_results[label][name] = (popt, rmse)

    print("\nBaseline (level 0, lights off) per zone:")
    for label, summ in summaries.items():
        b0 = summ[summ["level_idx"] == 0]
        if len(b0):
            print(
                f"  {label}: power={b0['mean_power'].mean():.3f} W "
                f"(n={int(b0['n'].sum())})"
            )

    # --- pooled fit across both zones (LEDs-on only) ---
    pooled_on = pd.concat(
        [
            summ[summ["level_idx"] >= 2].assign(zone=label)
            for label, summ in summaries.items()
        ],
        ignore_index=True,
    )
    x_pool = pooled_on["nominal_ppfd"].to_numpy()
    y_pool = pooled_on["mean_power"].to_numpy()
    pooled_baseline = float(np.mean(list(baselines.values())))
    print(f"\n--- Pooled fit  (zone01 + zone02, n={len(x_pool)} LEDs-on points) ---")
    print(f"    pooled baseline = {pooled_baseline:.3f} W")
    pooled_fits: dict[str, tuple[np.ndarray, float]] = {}
    for name, fn, p0, pnames in candidates:
        try:
            popt, _ = curve_fit(fn, x_pool, y_pool, p0=p0, maxfev=20000)
        except Exception as e:
            print(f"  {name}: fit failed ({e})")
            continue
        pred = fn(x_pool, *popt)
        rmse = float(np.sqrt(np.mean((y_pool - pred) ** 2)))
        param_str = ", ".join(
            f"{n}={v:.4g}" for n, v in zip(pnames, popt, strict=False)
        )
        print(f"  {name}: RMSE={rmse:.3f} W  |  {param_str}")
        pooled_fits[name] = (popt, rmse)

    # --- pooled power-law fit plot ---
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, summ in summaries.items():
        on = summ[summ["level_idx"] >= 2]
        off = summ[summ["level_idx"] <= 1]
        ax.scatter(
            on["nominal_ppfd"],
            on["mean_power"],
            color=COLORS[label],
            s=30,
            zorder=10,
            label=f"{label}",
        )
        ax.scatter(
            off["nominal_ppfd"],
            off["mean_power"],
            color=COLORS[label],
            s=30,
            marker="x",
            zorder=10,
            alpha=0.7,
        )
    ppfd_on_min = float(pooled_on["nominal_ppfd"].min())
    ppfd_max = float(pooled_on["nominal_ppfd"].max())
    x_grid = np.linspace(ppfd_on_min, ppfd_max, 200)
    ax.hlines(
        pooled_baseline,
        0,
        ppfd_on_min,
        color="dimgray",
        linewidth=2,
        label=f"baseline = {pooled_baseline:.2f} W",
    )
    popt_pl, rmse_pl = pooled_fits["power-law      (3p)"]
    a_pl, b_pl, c_pl = popt_pl
    ax.plot(
        x_grid,
        model_power_law(x_grid, *popt_pl),
        color="tab:red",
        linewidth=2,
        label=f"P = {a_pl:.2f} + {b_pl:.3f}·PPFD^{c_pl:.3f}   (RMSE={rmse_pl:.2f} W)",
    )
    ax.set_xlabel("nominal balanced PPFD")
    ax.set_ylabel("plug power (W)")
    ax.set_title("E18/P0.1 pooled power-law fit")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fits_pooled.pdf")
    plt.close(fig)

    # --- power-vs-drive: does the zone gap close when we plot on the actuator axis? ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for label, summ in summaries.items():
        on = summ[summ["level_idx"] >= 2]
        axes[0].scatter(
            on["mean_drive"],
            on["mean_power"],
            color=COLORS[label],
            s=30,
            label=label,
            alpha=0.85,
        )
        axes[1].scatter(
            on["mean_drive"],
            on["mean_current"],
            color=COLORS[label],
            s=30,
            label=label,
            alpha=0.85,
        )
    axes[0].set_xlabel("total drive (Σ calibrated_action)")
    axes[0].set_ylabel("plug power (W)")
    axes[0].set_title("Power vs drive — same actuator axis")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=9)
    axes[1].set_xlabel("total drive (Σ calibrated_action)")
    axes[1].set_ylabel("plug current (A)")
    axes[1].set_title("Current vs drive")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=9)
    fig.suptitle(
        "If zones overlap here, the power-vs-PPFD gap is calibration; "
        "if not, it's hardware"
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "calibration_vs_hardware.pdf")
    plt.close(fig)

    # --- voltage check ---
    print("\nMean supply voltage (V) during LEDs-on rows:")
    for label, summ in summaries.items():
        on = summ[summ["level_idx"] >= 2]
        print(f"  {label}: {on['mean_voltage'].mean():.2f} V")

    # --- power per drive at matched drive levels ---
    print("\nPower @ matched total-drive levels (interpolated):")
    drive_grid = np.linspace(0.4, 2.0, 9)
    rows_disp: list[str] = []
    rows_disp.append("  drive    z01_power  z02_power   Δ(W)   z01_I   z02_I   ΔI(A)")
    z1on = summaries["zone01"][summaries["zone01"]["level_idx"] >= 2].sort_values(
        "mean_drive"
    )
    z2on = summaries["zone02"][summaries["zone02"]["level_idx"] >= 2].sort_values(
        "mean_drive"
    )
    for d in drive_grid:
        p1 = float(np.interp(d, z1on["mean_drive"], z1on["mean_power"]))
        p2 = float(np.interp(d, z2on["mean_drive"], z2on["mean_power"]))
        i1 = float(np.interp(d, z1on["mean_drive"], z1on["mean_current"]))
        i2 = float(np.interp(d, z2on["mean_drive"], z2on["mean_current"]))
        rows_disp.append(
            f"  {d:5.2f}   {p1:8.2f}   {p2:8.2f}  {p1 - p2:+6.2f}  "
            f"{i1:6.3f}  {i2:6.3f}  {i1 - i2:+6.3f}"
        )
    print("\n".join(rows_disp))

    # --- overlay: candidate models on top of zone01 data ---
    fig, ax = plt.subplots(figsize=(9, 5.5))
    summ_z = summaries["zone01"]
    on_z = summ_z[summ_z["level_idx"] >= 2]
    off_z = summ_z[summ_z["level_idx"] <= 1]
    ax.scatter(
        on_z["nominal_ppfd"],
        on_z["mean_power"],
        color="black",
        s=30,
        zorder=10,
        label="zone01 data (LEDs on)",
    )
    ax.scatter(
        off_z["nominal_ppfd"],
        off_z["mean_power"],
        color="black",
        s=30,
        marker="x",
        zorder=10,
        label="zone01 baseline (gated)",
    )
    ppfd_on_min = float(on_z["nominal_ppfd"].min())
    ppfd_max = float(summ_z["nominal_ppfd"].max())
    x_grid = np.linspace(ppfd_on_min, ppfd_max, 200)
    # baseline segment up to first LEDs-on threshold
    ax.hlines(
        baselines["zone01"],
        0,
        ppfd_on_min,
        color="dimgray",
        linewidth=2,
        label=f"baseline = {baselines['zone01']:.2f} W",
    )
    model_fns = {
        "linear         (2p)": model_linear,
        "quadratic      (3p)": model_quadratic,
        "power-law      (3p)": model_power_law,
        "lin+exp decay  (4p)": model_lin_plus_exp,
    }
    cmap = plt.get_cmap("tab10")
    for i, (name, fn) in enumerate(model_fns.items()):
        if name not in fit_results["zone01"]:
            continue
        popt, rmse = fit_results["zone01"][name]
        ax.plot(
            x_grid,
            fn(x_grid, *popt),
            color=cmap(i),
            label=f"{name.strip()}  RMSE={rmse:.2f} W",
            linewidth=1.5,
        )
    for _ch, s in ACTIVATION.items():
        ax.axvline(
            s * TOTAL_BALANCED, color="gray", linestyle="--", linewidth=0.7, alpha=0.4
        )
    ax.set_xlabel("nominal balanced PPFD")
    ax.set_ylabel("plug power (W)")
    ax.set_title("E18/P0.1 zone01: candidate fits (LEDs-on region only)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fits_zone01.pdf")
    plt.close(fig)

    # --- residual plots for each candidate (zone01) ---
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for ax, (name, fn) in zip(axes.flat, model_fns.items(), strict=False):
        if name not in fit_results["zone01"]:
            ax.set_visible(False)
            continue
        popt, rmse = fit_results["zone01"][name]
        for label, summ in summaries.items():
            on = summ[summ["level_idx"] >= 2]
            xs = on["nominal_ppfd"].to_numpy()
            ys = on["mean_power"].to_numpy()
            ax.scatter(
                xs, ys - fn(xs, *popt), color=COLORS[label], label=label, alpha=0.85
            )
        ax.axhline(0, color="black", linewidth=0.6)
        for _ch, s in ACTIVATION.items():
            ax.axvline(
                s * TOTAL_BALANCED,
                color="gray",
                linestyle="--",
                linewidth=0.7,
                alpha=0.4,
            )
        ax.set_title(f"{name.strip()}   zone01 RMSE={rmse:.2f} W", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    for ax in axes[-1, :]:
        ax.set_xlabel("nominal balanced PPFD")
    for ax in axes[:, 0]:
        ax.set_ylabel("residual (W)")
    fig.suptitle(
        "Residuals vs PPFD for each candidate fit (zone01 params applied to both zones)"
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fit_residuals.pdf")
    plt.close(fig)

    csv_dir = Path("/tmp/e18_p0_1_summaries")
    csv_dir.mkdir(exist_ok=True)
    for label, summ in summaries.items():
        summ.to_csv(csv_dir / f"summary_{label}.csv", index=False)
    long_rows = []
    for label, summ in summaries.items():
        s = summ.copy()
        s["zone"] = label
        long_rows.append(s)
    pd.concat(long_rows, ignore_index=True).to_csv(
        csv_dir / "summary_long.csv", index=False
    )
    print(f"\nfigures (PDF) written to {OUT_DIR}; CSV summaries written to {csv_dir}")


if __name__ == "__main__":
    main()
