# Tray Configuration Tool - Web Interface

A web-based tool for configuring plant trays in growth chamber images. Uses a browser interface that works in containerized and remote environments without requiring X11 or a display server.

## Features

- **Browser-based UI**: Works in any modern web browser, no X11 or display server required
- **Point-and-click configuration**: Select 4 corner points to define each tray
- **Visual feedback**: See all configured trays overlaid on the image
- **Tray editing**: Right-click any tray to edit or delete it
- **Batch processing**: Process multiple datasets sequentially
- **Persistent storage**: Saves configurations to zone config files

## Quick Start

### 1. Start the Server

```bash
cd /workspaces/plant-rl/scripts/dataset_management/tray_config
python main.py
```

You should see:
```
============================================================
Tray Configuration Tool - Web Interface
============================================================

Using default dataset paths from configuration...
Config mode: PlantGrowthChamber

Starting web server on http://0.0.0.0:5000

Once started, open your browser to:
  http://localhost:5000

Press Ctrl+C to stop the server
============================================================
 * Running on http://0.0.0.0:5000
```

### 2. Open the UI

1. Open your browser
2. Navigate to: **http://localhost:5000**
3. The first dataset should load automatically

### 3. Configure Trays

#### Adding a Tray

1. **Set dimensions** (top controls):
   - Rows (n_tall): Number of rows in the tray
   - Columns (n_wide): Number of columns in the tray

2. **Click 4 points** on the image in order:
   - Top-left corner
   - Top-right corner
   - Bottom-left corner
   - Bottom-right corner

3. **Automatic save**: After the 4th point, the tray is saved automatically
   - Green outline appears around the tray
   - Label shows "Tray N: rows×columns"
   - Tray count updates

4. **Add more trays**: Simply click 4 more points for the next tray

#### Editing a Tray

1. **Right-click** on any green tray outline
2. The tray's points will appear as blue dots
3. Click 4 new points to update the tray
4. Or click **"Cancel Edit"** to abort

#### Deleting a Tray

1. **Right-click** on the tray to select it
2. Click **"Delete Tray"** button
3. Confirm the deletion

### 4. Save and Continue

- Click **"Save & Next Dataset"** for each dataset
- When all datasets are processed, you'll see: "All datasets have been processed"
- Press **Ctrl+C** in the terminal to stop the server

### Example Session

```
1. Start server: python main.py
2. Open browser: http://localhost:5000
3. See: "alliance-zone01 (1/12)"
4. Set: Rows=3, Columns=6
5. Click 4 corner points
6. Tray appears in green
7. Add more trays if needed
8. Click "Save & Next Dataset"
9. Repeat for next dataset
10. Done when you see "All datasets processed"
```

## Advanced Usage

### Command-line Options

```bash
# Custom port
python main.py --port 8080

# Specific paths
python main.py /data/online/E14/P0/* /data/online/E15/P0/*

# Debug mode
python main.py --debug /data/online/E14/P0/*

# Local config mode (save to dataset dirs instead of PlantGrowthChamber)
python main.py --local-config /data/online/E14/P0/*

# Combine options
python main.py --port 8080 --debug /data/online/E14/P0/*
```

### UI Controls

- **Left-click**: Select corner points for tray boundaries
- **Right-click**: Select an existing tray for editing or deletion
- **Save & Next Dataset**: Save current configuration and move to next dataset
- **Reset Points**: Clear current point selection
- **Delete Tray**: Remove the currently selected tray
- **Cancel Edit**: Cancel current tray editing operation

### Tips

- Use the latest image in each dataset (automatically loaded)
- Zoom in browser (Ctrl +/-) if you need to see details better
- You can access from any device on the network at `http://<server-ip>:5000`
- Check the terminal for error messages if something goes wrong
- Status bar at bottom shows which point to select (1/4, 2/4, etc.)

## Configuration

Edit `main.py` to change default settings:

- `UPDATE_CONFIG`: Set to `True` to update zone configs in `PlantGrowthChamber` directory (default)
- `BASE_DIRS`: Default directories to search for datasets (must contain `images/` subdirectory)

Example:
```python
BASE_DIRS = chain(
    Path("/data/online/E14/P0").rglob("*"),
    Path("/data/online/E15/P0").rglob("*"),
)
```

## Technical Details

### Architecture

**Backend (Flask)**
- RESTful JSON API
- In-memory server-side state
- PIL/Pillow for image processing
- Default port: 5000 (configurable)

**Frontend (HTML/JS)**
- HTML5 Canvas for rendering
- Vanilla JavaScript (no frameworks)
- Responsive CSS
- Async HTTP requests (Fetch API)

### File Structure

```
tray_config/
├── main.py              # Entry point with CLI
├── web_app.py           # Flask application
├── tray_config_utils.py # Utility functions
├── visualization.py     # Image processing utilities
├── templates/
│   └── index.html       # Web UI
└── README.md            # This file
```

### API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/` | Serve main UI page |
| GET | `/api/init` | Initialize and load first dataset |
| GET | `/api/image` | Get current dataset image |
| GET | `/api/status` | Get current status |
| POST | `/api/add_tray` | Add a new tray configuration |
| POST | `/api/update_tray` | Update an existing tray |
| POST | `/api/delete_tray` | Delete a tray |
| POST | `/api/save_and_next` | Save current config and load next dataset |

### Browser Compatibility

Requires a modern browser with Canvas API support:
- Chrome/Chromium
- Firefox
- Safari
- Edge

## Advantages Over tkinter

1. **No X11 Required**: Works in headless environments and dev containers
2. **Remote Access**: Access from any device on the network
3. **Better Performance**: Canvas rendering is optimized for image display
4. **Modern UI**: Responsive design with better UX
5. **Easy Deployment**: Only needs Python and Flask, no GUI dependencies

## Troubleshooting

### Port Already in Use
```bash
# Use a different port
python main.py --port 8080
```

Or kill the process using port 5000:
```bash
lsof -ti:5000 | xargs kill -9
```

### Can't Connect to Server
- Check that the server is running in the terminal
- Verify firewall settings allow connections on the port
- Try accessing via `http://localhost:5000` instead of `0.0.0.0`
- If running in a container, ensure port forwarding is configured

### Images Not Loading
- Ensure dataset directories have an `images/` subdirectory
- Check that images are in supported formats (.png, .jpg, .jpeg)
- Check server logs in terminal for errors
- Verify image files exist and are readable

### No Image Appears
- Check that dataset has `images/` subdirectory
- Check terminal logs for errors
- Verify image file format (.png, .jpg, .jpeg)

### Dependencies Missing
```bash
pip install flask pillow
```

## Migration from tkinter

The tool was previously tkinter-based. The web version provides the same functionality with these improvements:

- Works without X11/display server
- Accessible remotely via network
- Better performance with Canvas API
- Modern, responsive UI
- Easier deployment in containers

The original tkinter app (`app.py`) is preserved for reference.

## Command-line Options

```
usage: main.py [-h] [--port PORT] [--host HOST] [--debug] [--local-config]
               [paths ...]

positional arguments:
  paths            Paths to search for datasets (can use wildcards)

optional arguments:
  -h, --help       show this help message and exit
  --port PORT      Port to run the web server on (default: 5001)
  --host HOST      Host to bind the server to (default: 0.0.0.0)
  --debug          Run server in debug mode
  --local-config   Save configs locally in dataset dirs instead of
                   PlantGrowthChamber
```

## Dependencies

- `flask` - Web framework
- `pillow` - Image processing
- `numpy` - Numerical operations
- `opencv-python` (cv2) - Point-in-polygon testing

## License

Part of the plant-rl project.
