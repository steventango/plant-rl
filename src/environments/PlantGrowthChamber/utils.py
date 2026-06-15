from datetime import datetime

import aiohttp
import numpy as np
import pandas as pd
from aiohttp_retry import ExponentialRetry, RetryClient

# Per-step budgets. LAN_RETRIES=2 → one retry; per-attempt timeout is budget / 2.
LAN_RETRIES = 2
IMAGE_BUDGET_S = 6
ACTION_BUDGET_S = 2
CV_REQUEST_TIMEOUT_S = 50
GET_OBSERVATION_TIMEOUT_S = IMAGE_BUDGET_S + CV_REQUEST_TIMEOUT_S


async def _create_session(*, attempts: int, total_timeout: float) -> RetryClient:
    retry_options = ExponentialRetry(
        attempts=attempts,
        start_timeout=0,
        max_timeout=0,
        factor=1,
        statuses={500, 502, 503, 504, 429},
    )
    timeout = aiohttp.ClientTimeout(total=total_timeout)
    connector = aiohttp.TCPConnector(limit=10, ttl_dns_cache=300)
    return RetryClient(
        client_session=aiohttp.ClientSession(timeout=timeout, connector=connector),
        retry_options=retry_options,
        raise_for_status=True,
    )


async def create_image_session() -> RetryClient:
    """Camera GET: 2 attempts, each capped at half the image budget."""
    return await _create_session(
        attempts=LAN_RETRIES, total_timeout=IMAGE_BUDGET_S / LAN_RETRIES
    )


async def create_action_session() -> RetryClient:
    """Lightbar PUT: 2 attempts, each capped at half the action budget."""
    return await _create_session(
        attempts=LAN_RETRIES, total_timeout=ACTION_BUDGET_S / LAN_RETRIES
    )


async def create_cv_session() -> RetryClient:
    """CV pipeline: single attempt."""
    return await _create_session(attempts=1, total_timeout=CV_REQUEST_TIMEOUT_S)


def get_one_hot_time_observation(local_time: datetime):
    one_hot = np.zeros(13, dtype=np.float64)
    index = max(0, min(12, local_time.hour - 9))
    one_hot[index] = 1.0
    return one_hot


def mean_clean_area(df: pd.DataFrame) -> float:
    if df.empty or "clean_area" not in df.columns:
        return 0.0
    return float(df["clean_area"].mean())


def hours_normalized(local_time: datetime) -> float:
    hours_since_start = (local_time.hour - 9) + ((local_time.minute - 30) / 60)
    return float(np.clip(hours_since_start / 11.0, 0, 1))
