from datetime import datetime

import aiohttp
import numpy as np
from aiohttp_retry import ExponentialRetry, RetryClient


async def create_session():
    # Configure retry options with exponential backoff
    retry_options = ExponentialRetry(
        attempts=9,  # Maximum 9 retry attempts
        start_timeout=0.1,  # Start with 0.1s delay
        max_timeout=30,  # Maximum 30s delay
        factor=2,  # Double the delay each retry
        statuses={500, 502, 503, 504, 429},  # Retry on server errors and rate limiting
    )

    # Create RetryClient with retry options
    timeout = aiohttp.ClientTimeout(total=60)
    connector = aiohttp.TCPConnector(limit=10, ttl_dns_cache=300)
    # Create and return a new session instance directly
    session = RetryClient(
        client_session=aiohttp.ClientSession(timeout=timeout, connector=connector),
        retry_options=retry_options,
        raise_for_status=True,  # Automatically raise for HTTP errors
    )
    return session


def get_one_hot_time_observation(local_time: datetime):
    one_hot = np.zeros(13, dtype=np.float64)
    index = max(0, min(12, local_time.hour - 9))
    one_hot[index] = 1.0
    return one_hot
