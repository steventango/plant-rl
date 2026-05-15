"""Post-deploy analysis of the E18/P0.2 12 h shakedown.

Reads each zone's raw CSV (skipping the giant `state.*` image-array
columns), plots actions/power/voltage/current over time, and flags
anything anomalous in Z2's power record - including any signature of a
`put_action` aiohttp timeout that left the chamber in a stale-action
state for one or more env-steps.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_ROOT = Path("/data/plant-rl/online/E18/P0.2")
FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

ZONES = {
    "Z1 power-law ramp": DATA_ROOT
    / "SequencePowerLawRamp1/alliance-zone01/raw_2026-05-15.csv",
    "Z2 within-day parabola": DATA_ROOT
    / "SequenceParabolic2/alliance-zone02/raw_2026-05-15.csv",
    "Z3 constant 105": DATA_ROOT / "Constant3/alliance-zone03/raw_2026-05-15.csv",
}
ZONE_COLOR = {
    "Z1 power-law ramp": "tab:blue",
    "Z2 within-day parabola": "tab:orange",
    "Z3 constant 105": "tab:green",
}

BASE_COLS = [
    "time",
    "frame",
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
    "steps",
    "env_time",
]
POWER_COLS = ["power", "voltage", "current"]


def P_model(ppfd: np.ndarray) -> np.ndarray:
    """E18/P0.1 pooled fit, lights-on; baseline 7.21 W when zero."""
    return np.where(
        ppfd > 0,
        9.71 + 0.164 * np.power(np.maximum(ppfd, 1e-9), 1.19),
        7.21,
    )


def load_zone(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    cols = BASE_COLS + [c for c in POWER_COLS if c in header]
    df = pd.read_csv(path, usecols=cols)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    df["req_ppfd"] = df[[f"action.{i}" for i in range(5)]].sum(axis=1)
    df["drive"] = df[[f"calibrated_action.{i}" for i in range(6)]].sum(axis=1)
    df["predicted_power"] = P_model(df["req_ppfd"].to_numpy())
    if "power" in df.columns:
        df["residual"] = df["power"] - df["predicted_power"]
    return df


def cadence_stats(df: pd.DataFrame, label: str) -> None:
    deltas = df["time"].diff().dt.total_seconds()
    print(f"\n=== {label} ===")
    print(f"  n_rows: {len(df)}; span: {df['time'].iloc[0]} -> {df['time'].iloc[-1]}")
    print(
        f"  step deltas (s):  median={deltas.median():.1f}  "
        f"min={deltas.min():.1f}  max={deltas.max():.1f}  "
        f"std={deltas.std():.2f}"
    )
    # Flag only ANOMALOUS gaps (anything not on the typical 5-min cadence)
    typical_delta = float(deltas.median())
    abnormal = deltas[(deltas > typical_delta + 30) | (deltas < typical_delta - 30)]
    if len(abnormal):
        print(
            f"  anomalous step gaps ({len(abnormal)} rows, expected ~{typical_delta:.0f}s):"
        )
        for t, d in zip(df.loc[abnormal.index, "time"], abnormal, strict=False):
            print(f"    {t}  gap={d:.0f}s")
    if "power" in df.columns:
        n_nan_power = df["power"].isna().sum()
        print(f"  NaN power: {n_nan_power}")
    print(
        f"  unique req_ppfd: {sorted(df['req_ppfd'].round(1).unique().tolist())[:15]}"
    )
    on = df[df["req_ppfd"] > 0.5]
    if len(on) and "residual" in df.columns:
        print(f"  power vs predicted (lights-on, n={len(on)}):")
        print(f"    mean residual = {on['residual'].mean():+.2f} W")
        print(f"    abs residual median = {on['residual'].abs().median():.2f} W")
        print(f"    abs residual p95 = {on['residual'].abs().quantile(0.95):.2f} W")


def plot_zone(df: pd.DataFrame, label: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    t = df["time"]

    # Top: requested PPFD + spectrum components
    ch_names = ["blue", "cool_white", "warm_white", "orange_red", "red", "far_red"]
    ch_colors = [
        "tab:blue",
        "tab:cyan",
        "goldenrod",
        "tab:orange",
        "tab:red",
        "tab:brown",
    ]
    for i, (name, color) in enumerate(zip(ch_names, ch_colors, strict=False)):
        col = f"action.{i}"
        if df[col].max() < 1e-6:
            continue
        axes[0].plot(t, df[col], color=color, linewidth=0.6, label=name)
    axes[0].plot(
        t,
        df["req_ppfd"],
        color="black",
        linewidth=1.0,
        alpha=0.85,
        label="total req PPFD",
    )
    axes[0].set_ylabel("requested PPFD\nµmol m⁻² s⁻¹")
    axes[0].set_title(f"{label} — actions emitted on the real chamber")
    axes[0].legend(fontsize=8, loc="upper right", ncol=4)
    axes[0].grid(alpha=0.3)

    # Middle: power: measured vs model
    if "power" in df.columns:
        axes[1].plot(
            t,
            df["power"],
            color=ZONE_COLOR[label],
            linewidth=0.7,
            label="measured power",
        )
    axes[1].plot(
        t,
        df["predicted_power"],
        color="gray",
        linestyle="--",
        linewidth=0.8,
        alpha=0.85,
        label="P(req_ppfd) model",
    )
    axes[1].set_ylabel("plug power (W)")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].grid(alpha=0.3)

    # Bottom: residual + voltage/current overlay
    axes[2].axhline(0, color="black", linewidth=0.6)
    if "residual" in df.columns:
        axes[2].plot(
            t, df["residual"], color="tab:red", linewidth=0.5, label="power − model"
        )
    axes[2].set_ylabel("residual (W)")
    axes[2].set_xlabel("time (UTC)")
    axes[2].legend(fontsize=8, loc="upper right")
    axes[2].grid(alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()
    out = FIG_DIR / f"p02_{label.split()[0].lower()}_overview.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


def detect_stale_actions(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """A `put_action` timeout silently re-uses the previous action on the
    lightbar. Telltale: the recorded `action.*` changes but the measured
    `power` stays consistent with the *previous* action's PPFD for a
    step. We flag rows where |residual| is large relative to the typical
    on-zone residual."""
    if df["req_ppfd"].max() < 0.5 or "residual" not in df.columns:
        return df.iloc[:0]
    on = df[df["req_ppfd"] > 0.5].copy()
    sigma = on["residual"].abs().median() * 3.5 + 2.0  # robust threshold
    big = on[on["residual"].abs() > sigma].copy()
    if len(big):
        print(f"\n  [{label}] {len(big)} rows with |residual| > {sigma:.2f} W:")
        print(
            big[["time", "req_ppfd", "power", "predicted_power", "residual"]]
            .head(15)
            .to_string(index=False)
        )
    return big


def main() -> None:
    data = {}
    for label, path in ZONES.items():
        if not path.exists():
            print(f"!! missing {path}; skipping {label}")
            continue
        print(f"loading {label}: {path}")
        df = load_zone(path)
        data[label] = df
        cadence_stats(df, label)
        plot_zone(df, label)
        detect_stale_actions(df, label)

    # Combined overlay: power vs time across all zones
    fig, ax = plt.subplots(figsize=(11, 4))
    for label, df in data.items():
        if "power" not in df.columns:
            continue
        ax.plot(
            df["time"],
            df["power"],
            color=ZONE_COLOR[label],
            linewidth=0.7,
            label=label,
            alpha=0.85,
        )
    ax.set_xlabel("time (UTC)")
    ax.set_ylabel("plug power (W)")
    ax.set_title("E18/P0.2 — plug power per zone")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "p02_power_all_zones.pdf")
    plt.close(fig)
    print("\nwrote combined power plot")


if __name__ == "__main__":
    main()
