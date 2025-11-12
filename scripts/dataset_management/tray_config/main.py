#!/usr/bin/env python3
"""
Tray Configuration Tool - Web Interface
Entry point with command-line argument support
"""

import argparse
from itertools import chain
from pathlib import Path

from web_app import run_app

# Default configuration
UPDATE_CONFIG = True  # When True, updates configs in PlantGrowthChamber directory
DEFAULT_BASE_DIRS = chain(
    Path("/data/online/E14/P0").rglob("*"),
)


def main():
    """Run the web-based tray configuration tool"""
    parser = argparse.ArgumentParser(
        description="Tray Configuration Tool - Web Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default configuration
  python main.py

  # Custom port
  python main.py --port 8080

  # Specific paths
  python main.py /data/online/E14/P0/* /data/online/E15/P0/*

  # Debug mode with custom paths
  python main.py --debug /data/online/E14/P0/*

  # Local config mode
  python main.py --local-config /data/online/E14/P0/*
        """,
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5001,
        help="Port to run the web server on (default: 5001)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run server in debug mode",
    )
    parser.add_argument(
        "--local-config",
        action="store_true",
        help="Save configs locally in dataset dirs instead of PlantGrowthChamber",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=str,
        help="Paths to search for datasets (can use wildcards). If not provided, uses default from script.",
    )

    args = parser.parse_args()

    # Build list of base directories
    if args.paths:
        base_dirs = []
        for path_str in args.paths:
            path = Path(path_str).expanduser()
            if "*" in str(path):
                # Handle glob patterns
                parent = path.parent
                pattern = path.name
                base_dirs.extend(parent.rglob(pattern))
            else:
                base_dirs.append(path)

        if not base_dirs:
            print("Error: No valid paths provided")
            return 1
    else:
        # Use default BASE_DIRS
        base_dirs = DEFAULT_BASE_DIRS

    update_config = not args.local_config

    print("=" * 60)
    print("Tray Configuration Tool - Web Interface")
    print("=" * 60)
    if args.paths:
        print(f"\nSearching {len(args.paths)} path(s) for datasets...")
    else:
        print("\nUsing default dataset paths from configuration...")
    print(f"Config mode: {'PlantGrowthChamber' if update_config else 'Local'}")
    print(f"\nStarting web server on http://{args.host}:{args.port}")
    print("\nOnce started, open your browser to:")
    print(f"  http://localhost:{args.port}")
    if args.debug:
        print("\n⚠️  Running in DEBUG mode")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)

    run_app(
        base_dirs,
        update_config=update_config,
        host=args.host,
        port=args.port,
        debug=args.debug,
    )

    return 0


if __name__ == "__main__":
    exit(main())
