from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def generate_policy(filename, interval = 10, lam = 3):
    days = 28
    reference_action = np.array([0.398, 0.762, 0.324, 0.000, 0.332, 0.606])
    data = []
    total_minutes = 0
    prev_action = np.zeros_like(reference_action)
    day = 0
    intensity = np.random.choice([0.675, 1])

    while total_minutes < days * 24 * 60:
        delta = timedelta(minutes=total_minutes)
        day = delta.days
        if day >= days:
            break
        time_str = (datetime.min + delta).strftime("%H:%M:%S")
        new_action = reference_action * intensity
        if len(data):
            time_step_prev = (datetime.min + delta - timedelta(seconds=1)).strftime("%H:%M:%S")
            data.append([day, time_step_prev, *prev_action, 1.0])
        data.append([day, time_str, *new_action, 1.0])
        prev_action = new_action
        poisson_sample = min(np.random.poisson(lam), 5)
        interval_minutes = interval * (poisson_sample + 1)
        total_minutes += interval_minutes
        if intensity == 1:
            intensity = 0.675
        else:
            intensity = 1
    data.append([day, "23:59:59", *prev_action, 1.0])

    df = pd.DataFrame(
        data, columns=["Day", "Time", "Blue", "Cool_White", "Warm_White", "Orange_Red", "Red", "Far_Red", "Scaling"]
    )
    save_path = Path(__file__).parent / filename
    df.to_excel(save_path, index=False)


if __name__ == "__main__":
    for zone in [1]:
        generate_policy(f"poisson{zone}.xlsx", interval=10, lam=3)
