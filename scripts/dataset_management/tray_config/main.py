import tkinter as tk
from pathlib import Path

from app import TrayConfigApp

# Flag to update existing config files instead of creating new ones
UPDATE_CONFIG = True  # When True, updates configs in PlantGrowthChamber directory

# Base directories containing the datasets
BASE_DIRS = Path("/data/online/E10/P0").rglob("*")


def main():
    root = tk.Tk()
    TrayConfigApp(root, BASE_DIRS, UPDATE_CONFIG)
    root.mainloop()


if __name__ == "__main__":
    main()
