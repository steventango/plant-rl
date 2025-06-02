from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def generate_policy(filename, interval = 10, lam = 3):
    days = 28
    reference_action = np.array([0.398, 0.762, 0.324, 0.000, 0.332, 0.606])
    data = []
    day = 0

    intensity = np.random.choice([0.675, 1])

    # For each day, add structured entries for day/night transitions
    for day in range(days):
        prev_action = np.zeros_like(reference_action)

        # Generate poisson-based entries during daytime (9:00-21:00)
        current_minutes = 9 * 60  # Start at 9:00 AM

        while current_minutes < 21 * 60:  # Until 9:00 PM
            # Get time string
            delta = timedelta(minutes=current_minutes)
            time_str = (datetime.min + delta).strftime("%H:%M:%S")

            # Add previous action's end time
            time_step_prev = (datetime.min + delta - timedelta(seconds=1)).strftime("%H:%M:%S")
            data.append([day, time_step_prev, *prev_action, 1.0])

            # Flip intensity
            if intensity == 1:
                intensity = 0.675
            else:
                intensity = 1

            # Apply new action
            new_action = reference_action * intensity
            data.append([day, time_str, *new_action, 1.0])
            prev_action = new_action

            # Calculate next interval
            poisson_sample = min(np.random.poisson(lam), 5)
            interval_minutes = interval * (poisson_sample + 1)
            current_minutes += interval_minutes

            # Don't exceed 21:00
            if current_minutes >= 21 * 60:
                break

        # Add 21:00 entry (start of night)
        data.append([day, "20:59:59", *prev_action, 1.0])
        data.append([day, "21:00:00", 0, 0, 0, 0, 0, 0, 1.0])

    # Add final day's end
    data.append([days - 1, "23:59:59", 0, 0, 0, 0, 0, 0, 1.0])

    df = pd.DataFrame(
        data, columns=["Day", "Time", "Blue", "Cool_White", "Warm_White", "Orange_Red", "Red", "Far_Red", "Scaling"]
    )
    save_path = Path(__file__).parent / filename
    df.to_excel(save_path, index=False)


if __name__ == "__main__":
    for zone in [1]:
        generate_policy(f"poisson{zone}.xlsx", interval=10, lam=3)
