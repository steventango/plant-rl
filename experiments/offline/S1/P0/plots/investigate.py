#%%
import numpy as np
import pandas as pd

df = pd.read_csv('area.csv')
xs = df['xs'].values
ys = df['ys'].values

#%%
import matplotlib.pyplot as plt


# %%
import sys
sys.path.append('/workspaces/plant-rl/src')

from utils.metrics import UnbiasedExponentialMovingAverage as uema

def calc_ema(xs, alpha):
    ema = uema(alpha=alpha)
    ema_values = np.zeros_like(xs)
    for i, x in enumerate(xs):
        ema.update(x)
        ema_values[i] = ema.compute().item()
    return ema_values


# plot various EMA values
plt.plot(xs, ys, label='raw')

for alpha in [.8, .06]:
    plt.plot(xs, calc_ema(ys, alpha), label=f'alpha={alpha}')

plt.xlabel('Day Time [Hours]')
plt.ylabel('Area')
plt.legend()

# %%
# proposed reward function: difference between current and previous EMA
plt.plot(xs, (calc_ema(ys, .2) - calc_ema(ys, .06)) / 150, label='difference')
plt.xlabel('Day Time [Hours]')
plt.ylabel('Reward')
plt.legend()

# %%
