import logging
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from utils.logger import WandbAlertHandler, log
from wandb.sdk.wandb_alerts import AlertLevel


@pytest.fixture
def mock_wandb_run():
    run = MagicMock()
    return run


@pytest.fixture
def logger():
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s] %(levelname)s:%(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    return logger


def test_wandb_alert_handler_debug(mock_wandb_run, logger):
    handler = WandbAlertHandler(mock_wandb_run)
    logger.addHandler(handler)
    logger.debug("Test debug message")
    mock_wandb_run.alert.assert_not_called()


def test_wandb_alert_handler_info(mock_wandb_run, logger):
    handler = WandbAlertHandler(mock_wandb_run)
    logger.addHandler(handler)
    logger.info("Test info message")
    mock_wandb_run.alert.assert_called_once_with(
        title="Test info message",
        text="",
        level=AlertLevel.INFO,
        wait_duration=1,
    )


def test_wandb_alert_handler_warning(mock_wandb_run, logger):
    handler = WandbAlertHandler(mock_wandb_run)
    logger.addHandler(handler)
    logger.warning("Test warning message")
    mock_wandb_run.alert.assert_called_once_with(
        title="Test warning message",
        text="",
        level=AlertLevel.WARN,
        wait_duration=1,
    )


def test_wandb_alert_handler_error(mock_wandb_run, logger):
    handler = WandbAlertHandler(mock_wandb_run)
    logger.addHandler(handler)
    logger.error("Test error message")
    mock_wandb_run.alert.assert_called_once_with(
        title="Test error message",
        text="",
        level=AlertLevel.ERROR,
        wait_duration=1,
    )


def test_wandb_alert_handler_exception(mock_wandb_run, logger):
    handler = WandbAlertHandler(mock_wandb_run)
    logger.addHandler(handler)
    try:
        raise ValueError("Test exception")
    except ValueError as e:
        logger.exception(e)
    mock_wandb_run.alert.assert_called_once_with(
        title="Test exception",
        text="""```Traceback (most recent call last):
  File "/workspaces/plant-rl/tests/utils/test_logger.py", line 77, in test_wandb_alert_handler_exception
    raise ValueError("Test exception")
ValueError: Test exception```""",
        level=AlertLevel.ERROR,
        wait_duration=1,
    )


def test_log_function_with_mock_env(mock_wandb_run):
    env = MagicMock()
    glue = MagicMock()
    s = {"state_key": "state_value"}
    a = {"action_key": "action_value"}
    info = {"info_key": "info_value"}
    log(env, glue, mock_wandb_run, s, a, info, is_mock_env=True)
    mock_wandb_run.log.assert_called()


def test_log_function_with_real_env(mock_wandb_run):
    env = MagicMock()
    env.time.minute = 0
    env.image = np.zeros((10, 10, 3), dtype=np.uint8)
    env.images = {"warped": env.image}
    env.detections.xyxy = np.array([[0, 0, 1, 1]])
    env.detections.confidence = np.array([0.5])
    env.detections.class_id = np.array([0])
    env.detections.mask = np.zeros((1, 1), dtype=bool)
    glue = MagicMock()
    s = {"state_key": "state_value"}
    a = {"action_key": "action_value"}
    info = {"info_key": "info_value"}
    log(env, glue, mock_wandb_run, s, a, info, is_mock_env=False)
    mock_wandb_run.log.assert_called()


def test_log_function_with_dataframe(mock_wandb_run):
    env = MagicMock()
    glue = MagicMock()
    s = {"state_key": "state_value"}
    a = {"action_key": "action_value"}
    info = {"dataframe": pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})}
    log(env, glue, mock_wandb_run, s, a, info, is_mock_env=True)
    mock_wandb_run.log.assert_called()


def test_log_function_with_reward_terminal_return_episode(mock_wandb_run):
    env = MagicMock()
    glue = MagicMock()
    s = {"state_key": "state_value"}
    a = {"action_key": "action_value"}
    info = {"info_key": "info_value"}
    r = 1.0
    t = True
    episodic_return = 10.0
    episode = 1
    log(
        env,
        glue,
        mock_wandb_run,
        s,
        a,
        info,
        is_mock_env=True,
        r=r,
        t=t,
        episodic_return=episodic_return,
        episode=episode,
    )
    mock_wandb_run.log.assert_called()
