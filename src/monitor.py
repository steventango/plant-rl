import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import wandb
from wandb.sdk.wandb_alerts import AlertLevel

log_dir = Path("/data/logs")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "monitor.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class SystemMonitor:
    """Monitor system resources using Wandb's built-in system metrics collection."""

    def __init__(
        self,
        memory_threshold: float = 0.95,
        disk_threshold: float = 0.95,
        entity: str = "plant-rl",
        project: str = "monitor",
        disk_paths: tuple = ("/", "/data"),
    ):
        """
        Initialize the system monitor.

        Args:
            memory_threshold: Memory usage threshold for alerts (default: 0.95)
            disk_threshold: Disk usage threshold for alerts (default: 0.95)
            entity: Wandb entity name (default: "plant-rl")
            project: Wandb project name (default: "monitor")
            disk_paths: Tuple of disk paths to monitor (default: ("/", "/data"))
        """
        assert 0.0 <= memory_threshold <= 1.0, (
            "memory_threshold must be between 0 and 1"
        )
        assert 0.0 <= disk_threshold <= 1.0, "disk_threshold must be between 0 and 1"
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold
        self.disk_paths = disk_paths

        # Initialize wandb
        try:
            self.wandb_run = wandb.init(
                entity=entity,
                project=project,
                settings=wandb.Settings(
                    x_stats_disk_paths=disk_paths,
                ),
            )
            logger.info("Wandb initialized successfully")
        except Exception as e:
            logger.exception(f"Failed to initialize Wandb: {e}")
            raise

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics from wandb's built-in system monitoring."""
        try:
            api = wandb.Api()
            run = api.run(self.wandb_run.path)
            system_metrics_history = run.history(stream="system")
            if system_metrics_history.empty:
                logger.warning("No system metrics found in wandb history")
                return {}
            system_metrics = system_metrics_history.iloc[-1]

            logger.info("Latest system metrics:")
            logger.info(system_metrics)

            return system_metrics
        except Exception as e:
            logger.exception(f"Failed to get system metrics from wandb: {e}")
            return {}

    def send_wandb_alert(self, title: str, text: str):
        """Send an alert using Wandb's alert system."""
        try:
            self.wandb_run.alert(
                title=title,
                text=text,
                level=AlertLevel.WARN,
            )
            logger.warning(f"ALERT: {title}")
            logger.warning(f"Details: {text}")
        except Exception as e:
            logger.exception(f"Failed to send Wandb alert: {e}")

    def check_memory(self, metrics: Dict[str, Any]):
        """Check memory usage and send wandb alert if threshold exceeded."""
        memory_percent = metrics.get("memory_percent", 0)

        if memory_percent == 0:
            logger.warning("Memory metrics not available from wandb yet")
            return

        if memory_percent >= self.memory_threshold:
            title = f"High Memory Usage Alert - {memory_percent * 100:.1f}%"
            text = (
                f"Memory usage has exceeded {self.memory_threshold * 100}% threshold.\n"
                f"Current system memory usage: {memory_percent * 100:.1f}%\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )

            self.send_wandb_alert(title, text)
            logger.warning(f"Memory usage high: {memory_percent * 100:.1f}%")
        else:
            logger.info(f"Memory usage normal: {memory_percent * 100:.1f}%")

    def check_disk(self, metrics: Dict[str, Any]):
        """Check disk usage and send wandb alert if threshold exceeded for any disk."""
        # Find all disk metrics
        disk_metrics = {
            k: v
            for k, v in metrics.items()
            if k.startswith("disk_") and k.endswith("_percent")
        }

        if not disk_metrics:
            logger.warning("No disk metrics available from wandb yet")
            return

        # Check each disk individually
        high_usage_disks = []
        all_disk_status = []

        for disk_name, usage in disk_metrics.items():
            # Convert disk name back to path for display
            sanitized_path = disk_name.replace("disk_", "").replace("_percent", "")
            disk_path = (
                "/"
                if sanitized_path == "root"
                else f"/{sanitized_path.replace('_', '/')}"
            )

            if usage == 0:
                logger.warning(
                    f"Disk metrics not available for {disk_path} from wandb yet"
                )
                continue

            # Log status for each disk
            if usage >= self.disk_threshold:
                high_usage_disks.append((disk_path, usage))
                logger.warning(f"Disk usage high on {disk_path}: {usage:.1f}%")
            else:
                logger.info(f"Disk usage normal on {disk_path}: {usage:.1f}%")

            all_disk_status.append(f"{disk_path}: {usage:.1f}%")

        # Send alert if any disk exceeds threshold
        if high_usage_disks:
            # Sort by usage percentage (highest first)
            high_usage_disks.sort(key=lambda x: x[1], reverse=True)

            highest_disk_path, highest_usage = high_usage_disks[0]

            title = f"High Disk Usage Alert - {len(high_usage_disks)} disk(s) above threshold"

            # Build detailed text with all disk usages
            all_disk_details = "\n".join(all_disk_status)
            high_usage_details = "\n".join(
                [f"{path}: {usage:.1f}%" for path, usage in high_usage_disks]
            )

            text = (
                f"Disk usage has exceeded {self.disk_threshold}% threshold on {len(high_usage_disks)} disk(s).\n"
                f"Highest usage: {highest_disk_path} at {highest_usage:.1f}%\n\n"
                f"Disks above threshold:\n{high_usage_details}\n\n"
                f"All monitored disks:\n{all_disk_details}\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )

            self.send_wandb_alert(title, text)
        else:
            # Log summary of all disks when none exceed threshold
            if all_disk_status:
                max_usage = max(disk_metrics.values())
                logger.info(
                    f"All disks within threshold - highest usage: {max_usage:.1f}%"
                )

    def run_monitoring_cycle(self):
        """Run one complete monitoring cycle."""
        logger.info("Running monitoring cycle...")

        try:
            # Get system metrics (simplified since wandb handles the actual monitoring)
            metrics = self.get_system_metrics()

            if not metrics:
                logger.warning("No metrics available for this cycle")
                return

            # Check memory monitoring status
            self.check_memory(metrics)

            # Check disk monitoring status
            self.check_disk(metrics)

        except Exception as e:
            logger.exception(f"Error during monitoring cycle: {e}")

    def run(self, interval_seconds: int = 60):
        """
        Run the monitoring loop.

        Args:
            interval_seconds: Time interval between checks in seconds (default: 60)
        """
        logger.info(f"Starting system monitor with {interval_seconds}s interval")
        logger.info(f"Memory threshold: {self.memory_threshold}%")
        logger.info(f"Disk threshold: {self.disk_threshold}%")

        try:
            while True:
                self.run_monitoring_cycle()
                logger.info(f"Sleeping for {interval_seconds} seconds...")
                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.exception(f"Monitoring loop error: {e}")
            raise
        finally:
            try:
                wandb.finish()
                logger.info("Wandb session closed")
            except Exception as e:
                logger.exception(f"Error closing Wandb session: {e}")


def main():
    """Main function to run the system monitor."""

    parser = argparse.ArgumentParser(
        description="System Resource Monitor with Wandb Alerts"
    )
    parser.add_argument(
        "--memory-threshold",
        type=float,
        default=0.95,
        help="Memory usage threshold for alerts (default: 0.95)",
    )
    parser.add_argument(
        "--disk-threshold",
        type=float,
        default=0.95,
        help="Disk usage threshold for alerts (default: 0.95)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Check interval in seconds (default: 60)",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default="plant-rl",
        help="Wandb entity name (default: plant-rl)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="monitor",
        help="Wandb project name (default: monitor)",
    )
    parser.add_argument(
        "--disk-paths",
        type=str,
        nargs="+",
        default=["/", "/data"],
        help="Disk paths to monitor (default: / /data)",
    )

    args = parser.parse_args()

    # Create and run the monitor
    monitor = SystemMonitor(
        memory_threshold=args.memory_threshold,
        disk_threshold=args.disk_threshold,
        entity=args.entity,
        project=args.project,
        disk_paths=tuple(args.disk_paths),
    )

    monitor.run(interval_seconds=args.interval)


if __name__ == "__main__":
    main()
