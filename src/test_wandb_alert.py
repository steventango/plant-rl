# %%

import logging

import wandb
from utils.logger import WandbAlertLogger


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
    logger = WandbAlertLogger("plant-rl", wandb_run)
    logger.setLevel(logging.DEBUG)
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(console_handler)

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
