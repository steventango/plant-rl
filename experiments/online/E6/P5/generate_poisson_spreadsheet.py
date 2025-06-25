from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def generate_policy(filename, interval = 5, lam = 3):
    days = 21
    reference_spectrum = np.array([0.398, 0.762, 0.324, 0.000, 0.332, 0.606])
    red = np.array([0.000, 0.324, 0.324, 0.000, 1.000, 0.606])
    blue = np.array([1.000, 0.324, 0.324, 0.000, 0.000, 0.000])
    action_map = {
        0: reference_spectrum * 0.350,
        1: reference_spectrum * 0.675,
        2: reference_spectrum,
        3: reference_spectrum * 1.652,
        4: red * 0.675,
        5: red,
        6: red * 1.652,
        7: blue * 0.675,
        8: blue,
        9: blue * 1.652,
    }
    data = []
    total_minutes = 0
    prev_action = np.zeros_like(reference_spectrum)
    day = 0
    while total_minutes < days * 24 * 60:
        delta = timedelta(minutes=total_minutes)
        day = delta.days
        if day >= days:
            break
        time_str = (datetime.min + delta).strftime("%H:%M:%S")
        new_action = action_map[np.random.choice(10)]
        if len(data):
            time_step_prev = (datetime.min + delta - timedelta(seconds=1)).strftime("%H:%M:%S")
            data.append([day, time_step_prev, *prev_action, 1.0])
        data.append([day, time_str, *new_action, 1.0])
        prev_action = new_action
        interval_step = np.random.poisson(lam) + 1
        total_minutes += interval_step * interval
    data.append([day, "23:59:59", *prev_action, 1.0])

    df = pd.DataFrame(
        data, columns=["Day", "Time", "Blue", "Cool_White", "Warm_White", "Orange_Red", "Red", "Far_Red", "Scaling"]
    )
    save_path = Path(__file__).parent / filename
    df.to_excel(save_path, index=False)


if __name__ == "__main__":
    for zone in [1,2,6,9]:
        generate_policy(f"poisson{zone}.xlsx", interval=10, lam=2)
