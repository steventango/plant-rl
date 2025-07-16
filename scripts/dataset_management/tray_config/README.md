# Tray Configuration Tool

This tool provides a graphical interface for configuring trays in plant growth chambers.

## Overview

The Tray Configuration Tool allows users to:

1. View images from plant growth chambers
2. Mark tray boundaries by selecting points on the image
3. Specify tray dimensions (rows and columns)
4. Save the configurations to zone configuration files

## Structure

The tool is organized as follows:

- `__init__.py`: Entry point that defines constants and the main function
- `main.py`: Executable script to launch the application
- `app.py`: Main application class (TrayConfigApp)
- `utils.py`: Utility functions for file handling, zone identification, etc.
- `visualization.py`: Functions for visualizing trays and handling image display

## Usage

To run the tool:

```bash
python -m scripts.dataset_management.tray_config.main
```

### Controls

- **Left-click**: Select a point for tray corners
- **Right-click**: Select an existing tray for editing or deletion
- **Save & Next Dataset**: Save current configuration and move to next dataset
- **Reset Points**: Clear current point selection
- **Delete Tray**: Remove the currently selected tray
- **Cancel Edit**: Cancel current tray editing operation

## Configuration

Modify the following constants in `__init__.py` to customize behavior:

- `UPDATE_CONFIG`: When True, updates configs in PlantGrowthChamber directory
- `BASE_DIRS`: List of base directories containing datasets to process
