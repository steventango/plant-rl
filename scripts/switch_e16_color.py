#!/usr/bin/env python3

import argparse
import re
import subprocess
from pathlib import Path

COLOR_TO_PHASE = {
    "white": "P3",
    "red": "P4",
    "blue": "P5",
}

DEFAULT_SERVICES = [f"zone{i}" for i in range(1, 13)]


def update_compose(
    compose_path: Path, experiment: str, color: str, dry_run: bool
) -> int:
    phase = COLOR_TO_PHASE[color]
    text = compose_path.read_text()

    changed = 0
    updated_lines: list[str] = []
    zone_index = 0

    for line in text.splitlines(keepends=True):
        if "command: uv run python src/main_real.py -e " not in line:
            updated_lines.append(line)
            continue

        zone_index += 1
        if zone_index > 12:
            updated_lines.append(line)
            continue

        pattern = rf"experiments/online/{re.escape(experiment)}/P[^/]+/(?:Constant|Random){zone_index}\.json"
        replacement = (
            f"experiments/online/{experiment}/{phase}/Constant{zone_index}.json"
        )
        new_line, n = re.subn(pattern, replacement, line)

        if n:
            changed += 1
            updated_lines.append(new_line)
        else:
            updated_lines.append(line)

    if changed == 0:
        raise RuntimeError(
            "No zone command paths were updated. Check compose file format or experiment name."
        )

    if not dry_run:
        compose_path.write_text("".join(updated_lines))

    print(
        f"Switched {changed} zone command(s) in {compose_path} to {experiment}/{phase} ({color})."
    )
    return changed


def run_compose_apply(compose_path: Path, down_first: bool) -> None:
    compose_cmd = ["docker", "compose", "-f", str(compose_path)]

    if down_first:
        down_cmd = [*compose_cmd, "down", *DEFAULT_SERVICES]
        print("Running:", " ".join(down_cmd))
        subprocess.run(down_cmd, check=True)

    up_cmd = [*compose_cmd, "up", "-d", *DEFAULT_SERVICES]
    print("Running:", " ".join(up_cmd))
    subprocess.run(up_cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Switch E16 zone configs in compose.yml between white/red/blue phases."
    )
    parser.add_argument("color", choices=["white", "red", "blue"])
    parser.add_argument(
        "--compose",
        type=Path,
        default=Path("compose.yml"),
        help="Path to compose file (default: compose.yml)",
    )
    parser.add_argument(
        "--experiment",
        default="E16",
        help="Experiment id under experiments/online (default: E16)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-apply",
        action="store_true",
        help="Only update compose.yml and do not run docker compose up -d",
    )
    parser.add_argument(
        "--down-first",
        action="store_true",
        help="Run docker compose down zone1..zone12 before up",
    )
    args = parser.parse_args()

    if args.dry_run and (args.no_apply or args.down_first):
        parser.error("--dry-run cannot be combined with --no-apply/--down-first")

    if args.down_first and args.no_apply:
        parser.error("--down-first cannot be used with --no-apply")

    update_compose(args.compose, args.experiment, args.color, args.dry_run)

    if not args.dry_run and not args.no_apply:
        run_compose_apply(args.compose, args.down_first)


if __name__ == "__main__":
    main()
