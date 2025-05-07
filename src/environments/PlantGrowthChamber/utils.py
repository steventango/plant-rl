from datetime import datetime

import aiohttp
import numpy as np
from aiohttp_retry import ExponentialRetry, RetryClient

_session = None


async def get_session():
    global _session
    if _session is not None:
        return _session
    # Configure retry options with exponential backoff
    retry_options = ExponentialRetry(
        attempts=3,  # Maximum 3 retry attempts
        start_timeout=0.5,  # Start with 0.5s delay
        max_timeout=10,  # Maximum 10s delay
        factor=2,  # Double the delay each retry
        statuses={500, 502, 503, 504, 429},  # Retry on server errors and rate limiting
    )

    # Create RetryClient with retry options
    timeout = aiohttp.ClientTimeout(total=60)
    connector = aiohttp.TCPConnector(limit=10, ttl_dns_cache=300)
    _session = RetryClient(
        client_session=aiohttp.ClientSession(timeout=timeout, connector=connector),
        retry_options=retry_options,
        raise_for_status=True,  # Automatically raise for HTTP errors
    )
    return _session


def get_one_hot_time_observation(local_time: datetime):
    one_hot = np.zeros(13, dtype=np.float64)
    index = max(0, min(12, local_time.hour - 9))
    one_hot[index] = 1.0
    return one_hot
