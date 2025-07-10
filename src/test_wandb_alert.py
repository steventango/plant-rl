import logging

import wandb
from utils.logger import WandbAlertHandler

# Get the root logger
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s:%(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()


def main():
    wandb_run = wandb.init(
        entity="plant-rl",
        project="main",
        settings=wandb.Settings(
            x_stats_disk_paths=(
                "/",
                "/data",
            ),  # So wandb alerts when data dir is near full
        ),
    )

    # Set up logger
    handler = WandbAlertHandler(wandb_run)
    logger.addHandler(handler)

    logger.debug("This is a debug alert to test wandb alerts.")

    try:
        logger.warning("This is a test alert from the wandb_alert script.")
        raise ValueError("This is a test error to trigger an alert.")
    except Exception as e:
        logger.exception(e)
        raise e
    finally:
        logger.info("This is an info alert to test wandb alerts.")

    wandb_run.finish()


if __name__ == "__main__":
    main()
# %%
