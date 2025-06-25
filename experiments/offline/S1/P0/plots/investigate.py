# %%
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append("/workspaces/plant-rl/src")

from utils.metrics import UnbiasedExponentialMovingAverage as uema

df = pd.read_csv("area.csv")
xs = df["xs"].values
ys = df["ys"].values

# %%
# %%


def calc_ema(xs, alpha):
    ema = uema(alpha=alpha)
    ema_values = np.zeros_like(xs)
    for i, x in enumerate(xs):
        ema.update(x)
        ema_values[i] = ema.compute().item()
    return ema_values


# plot various EMA values
plt.plot(xs, ys, label="raw")

for alpha in [0.8, 0.06]:
    plt.plot(xs, calc_ema(ys, alpha), label=f"alpha={alpha}")

plt.xlabel("Day Time [Hours]")
plt.ylabel("Area")
plt.legend()

# %%
# proposed reward function: difference between current and previous EMA
plt.plot(xs, (calc_ema(ys, 0.2) - calc_ema(ys, 0.06)) / 150, label="difference")
plt.xlabel("Day Time [Hours]")
plt.ylabel("Reward")
plt.legend()

# %%
