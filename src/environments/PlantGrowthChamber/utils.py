from datetime import datetime

import aiohttp
import numpy as np
import pandas as pd
from aiohttp_retry import ExponentialRetry, RetryClient


async def _create_session(*, attempts: int, total_timeout: float) -> RetryClient:
    retry_options = ExponentialRetry(
        attempts=attempts,
        start_timeout=1,
        max_timeout=4,
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


async def create_action_session() -> RetryClient:
    """Fast-fail session for LAN endpoints (lightbar PUT and camera GET).

    Lightbar steady-state response is ~600-800 ms (lightbar.py:50-60 does
    12 I2C writes x 50 ms sleep = 600 ms server-side); cameras on the same
    LAN respond comparably. Aggressive retry budget so a stuck call can't
    push env.step past its 60 s/cycle budget - a dropped call is re-emitted
    on the next env.step anyway.
    """
    return await _create_session(attempts=2, total_timeout=15)


async def create_cv_session() -> RetryClient:
    """Generous-timeout session for the CV pipeline only.

    CV detect/propagate can take ~30 s when multiple chambers compete for
    one CV server, so per-request timeout must accommodate that. No retries
    within an env.step (attempts=1) - if CV doesn't respond, the previous
    frame's df is reused via the asyncio.wait_for fallback on
    get_observation.
    """
    return await _create_session(attempts=1, total_timeout=45)


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
