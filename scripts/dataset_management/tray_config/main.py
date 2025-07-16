import tkinter as tk
from itertools import chain
from pathlib import Path

from app import TrayConfigApp

# Flag to update existing config files instead of creating new ones
UPDATE_CONFIG = True  # When True, updates configs in PlantGrowthChamber directory

# Base directories containing the datasets
BASE_DIRS = chain(
    Path("/data/online/E10/P0").rglob("*"),
    [Path("/data/online/A0/P0/MotionTrackingControllerExp6")],
)


def main():
    root = tk.Tk()
    TrayConfigApp(root, BASE_DIRS, UPDATE_CONFIG)
    root.mainloop()


if __name__ == "__main__":
    main()
