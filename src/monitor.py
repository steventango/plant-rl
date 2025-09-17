import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

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

        # State tracking for alerts
        self.memory_alert_active = False
        self.disk_alert_active = {path: False for path in disk_paths}

    def get_system_metrics(self) -> pd.Series:
        """Get current system metrics from wandb's built-in system monitoring."""
        try:
            api = wandb.Api()
            run = api.run(self.wandb_run.path)
            system_metrics_history = run.history(stream="system")
            if system_metrics_history.empty:
                logger.warning("No system metrics found in wandb history")
                return pd.Series()
            system_metrics = system_metrics_history.iloc[-1]

            logger.info("Latest system metrics:")
            logger.info(system_metrics)

            return system_metrics
        except Exception as e:
            logger.exception(f"Failed to get system metrics from wandb: {e}")
            return pd.Series()

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

    def check_memory(self, metrics: pd.Series):
        """Check memory usage and send wandb alert if threshold breached."""
        memory_percent = metrics.get("system.memory_percent", 0)

        if memory_percent == 0:
            logger.warning("Memory metrics not available from wandb yet")
            return

        if memory_percent is not None and memory_percent >= self.memory_threshold * 100:
            if not self.memory_alert_active:
                title = f"High Memory Usage Alert - {memory_percent:.1f}%"
                text = (
                    f"Memory usage has exceeded {self.memory_threshold * 100:.0f}% threshold.\n"
                    f"Current system memory usage: {memory_percent:.1f}%\n"
                    f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
                self.send_wandb_alert(title, text)
                logger.warning(f"Memory usage high: {memory_percent:.1f}%")
                self.memory_alert_active = True
            else:
                logger.info(
                    f"Memory usage still high: {memory_percent:.1f}% (alert already sent)"
                )
        else:
            if self.memory_alert_active:
                logger.info(
                    f"Memory usage back to normal: {memory_percent:.1f}% (alert reset)"
                )
            else:
                logger.info(f"Memory usage normal: {memory_percent:.1f}%")
            self.memory_alert_active = False

    def check_disk(self, metrics: pd.Series):
        """Check disk usage and send wandb alert if threshold breached for any disk."""
        disk_metrics = {
            k: v
            for k, v in metrics.items()
            if isinstance(k, str) and "disk." in k and ".usagePercent" in k
        }

        if not disk_metrics:
            logger.warning("No disk metrics available from wandb yet")
            return

        all_disk_status = []

        for disk_name, usage in disk_metrics.items():
            disk_path = disk_name.replace("system.disk.", "").replace(
                ".usagePercent", ""
            )
            if usage == 0:
                logger.warning(
                    f"Disk metrics not available for {disk_path} from wandb yet"
                )
                continue

            if usage >= self.disk_threshold * 100:
                if not self.disk_alert_active.get(disk_path, False):
                    title = f"High Disk Usage Alert - {disk_path} {usage:.1f}%"
                    text = (
                        f"Disk usage has exceeded {self.disk_threshold * 100:.0f}% threshold on {disk_path}.\n"
                        f"Current usage: {usage:.1f}%\n"
                        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    self.send_wandb_alert(title, text)
                    logger.warning(f"Disk usage high on {disk_path}: {usage:.1f}%")
                    self.disk_alert_active[disk_path] = True
                else:
                    logger.info(
                        f"Disk usage still high on {disk_path}: {usage:.1f}% (alert already sent)"
                    )
            else:
                if self.disk_alert_active.get(disk_path, False):
                    logger.info(
                        f"Disk usage back to normal on {disk_path}: {usage:.1f}% (alert reset)"
                    )
                else:
                    logger.info(f"Disk usage normal on {disk_path}: {usage:.1f}%")
                self.disk_alert_active[disk_path] = False

            all_disk_status.append(f"{disk_path}: {usage:.1f}%")

        # Optionally log summary of all disks
        if all_disk_status:
            max_usage = max(disk_metrics.values())
            logger.info(
                f"Disk usage summary: {', '.join(all_disk_status)}; highest usage: {max_usage:.1f}%"
            )

    def run_monitoring_cycle(self):
        """Run one complete monitoring cycle."""
        logger.info("Running monitoring cycle...")

        try:
            # Get system metrics (simplified since wandb handles the actual monitoring)
            metrics = self.get_system_metrics()

            if metrics.empty:
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
        logger.info(f"Memory threshold: {self.memory_threshold * 100}%")
        logger.info(f"Disk threshold: {self.disk_threshold * 100}%")

        try:
            # Initial delay to allow Wandb to start collecting metrics
            logger.info(
                "Waiting for Wandb to initialize and collect initial metrics..."
            )
            time.sleep(30)
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
