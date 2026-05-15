from datetime import datetime

import aiohttp
import numpy as np
from aiohttp_retry import ExponentialRetry, RetryClient


async def create_action_session():
    """Fast-fail session for LAN endpoints (lightbar PUT and camera GET).

    Lightbar steady-state response is ~600-800 ms (lightbar.py:50-60 does
    12 I2C writes x 50 ms sleep = 600 ms server-side); cameras on the same
    LAN respond comparably. Aggressive retry budget so a stuck call can't
    push env.step past its 60 s/cycle budget - a dropped call is re-emitted
    on the next env.step anyway.
    """
    retry_options = ExponentialRetry(
        attempts=2,
        start_timeout=1,
        max_timeout=4,
        factor=1,
        statuses={500, 502, 503, 504, 429},
    )
    # 15 s session default is a safety net; callers override per-request
    # (put_action uses 2 s, _fetch_image uses 10 s).
    timeout = aiohttp.ClientTimeout(total=15)
    connector = aiohttp.TCPConnector(limit=10, ttl_dns_cache=300)
    return RetryClient(
        client_session=aiohttp.ClientSession(timeout=timeout, connector=connector),
        retry_options=retry_options,
        raise_for_status=True,
    )


async def create_cv_session():
    """Generous-timeout session for the CV pipeline only.

    CV detect/propagate can take ~30 s when multiple chambers compete for
    one CV server, so per-request timeout must accommodate that. No retries
    within an env.step (attempts=1) - if CV doesn't respond, the previous
    frame's df is reused via the asyncio.wait_for fallback on
    get_observation.
    """
    retry_options = ExponentialRetry(
        attempts=1,
        start_timeout=1,
        max_timeout=4,
        factor=1,
        statuses={500, 502, 503, 504, 429},
    )
    # 45 s session default; CVPipelineClient.detect/propagate override
    # per-request to 35 s so the call returns before get_observation's
    # 50 s asyncio.wait_for ceiling fires.
    timeout = aiohttp.ClientTimeout(total=45)
    connector = aiohttp.TCPConnector(limit=10, ttl_dns_cache=300)
    return RetryClient(
        client_session=aiohttp.ClientSession(timeout=timeout, connector=connector),
        retry_options=retry_options,
        raise_for_status=True,
    )


def get_one_hot_time_observation(local_time: datetime):
    one_hot = np.zeros(13, dtype=np.float64)
    index = max(0, min(12, local_time.hour - 9))
    one_hot[index] = 1.0
    return one_hot
