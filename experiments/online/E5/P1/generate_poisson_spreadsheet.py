from datetime import timedelta, datetime
from pathlib import Path

import numpy as np
import pandas as pd


def generate_policy():
    days = 21
    lam = 3
    reference_action = np.array([0.199, 0.381, 0.162, 0.000, 0.166, 0.303])
    data = []
    total_minutes = 0
    prev_action = np.zeros_like(reference_action)
    day = 0
    while total_minutes < days * 24 * 60:
        delta = timedelta(minutes=total_minutes)
        day = delta.days
        if day >= days:
            break
        time_str = (datetime.min + delta).strftime("%H:%M:%S")
        new_intensity = np.random.choice([0, 0.5, 1, 1.5])
        new_action = reference_action * new_intensity
        if len(data):
            time_step_prev = (datetime.min + delta - timedelta(seconds=1)).strftime("%H:%M:%S")
            data.append([day, time_step_prev, *prev_action, 1.0])
        data.append([day, time_str, *new_action, 1.0])
        prev_action = new_action
        interval_step = np.random.poisson(lam) + 1
        total_minutes += interval_step * 5
    data.append([day, "23:59:59", *prev_action, 1.0])

    df = pd.DataFrame(
        data, columns=["Day", "Time", "Blue", "Cool_White", "Warm_White", "Orange_Red", "Red", "Far_Red", "Scaling"]
    )
    save_path = Path(__file__).parent / "poisson.xlsx"
    df.to_excel(save_path, index=False)


if __name__ == "__main__":
    generate_policy()
