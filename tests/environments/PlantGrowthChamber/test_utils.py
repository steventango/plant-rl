from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

from environments.PlantGrowthChamber.utils import get_one_hot_time_observation

ZERO = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
ONE = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
TWO = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
THREE = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
FOUR = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
FIVE = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
SIX = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
SEVEN = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
EIGHT = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32)
NINE = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=np.float32)
TEN = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=np.float32)
ELEVEN = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=np.float32)
TWELVE = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32)

tz = ZoneInfo("America/Edmonton")



def test_get_one_hot_time_observation_zero():
    local_time = datetime(2025, 5, 7, 9, 0, 0, tzinfo=tz)
    observation = get_one_hot_time_observation(local_time)
    np.testing.assert_equal(observation, ZERO)
    local_time = datetime(2025, 5, 7, 9, 0, 1, tzinfo=tz)
    observation = get_one_hot_time_observation(local_time)
    np.testing.assert_equal(observation, ZERO)
    local_time = datetime(2025, 5, 7, 9, 5, 0, tzinfo=tz)
    observation = get_one_hot_time_observation(local_time)
    np.testing.assert_equal(observation, ZERO)
    local_time = datetime(2025, 5, 7, 9, 10, 0, tzinfo=tz)
    observation = get_one_hot_time_observation(local_time)
    np.testing.assert_equal(observation, ZERO)
    local_time = datetime(2025, 5, 7, 9, 20, 0, tzinfo=tz)
    observation = get_one_hot_time_observation(local_time)
    np.testing.assert_equal(observation, ZERO)
    local_time = datetime(2025, 5, 7, 9, 30, 0, tzinfo=tz)
    observation = get_one_hot_time_observation(local_time)
    np.testing.assert_equal(observation, ZERO)
    local_time = datetime(2025, 5, 7, 9, 40, 0, tzinfo=tz)
    observation = get_one_hot_time_observation(local_time)
    np.testing.assert_equal(observation, ZERO)
    local_time = datetime(2025, 5, 7, 9, 50, 0, tzinfo=tz)
    observation = get_one_hot_time_observation(local_time)
    np.testing.assert_equal(observation, ZERO)
    local_time = datetime(2025, 5, 7, 9, 59, 59, tzinfo=tz)
    observation = get_one_hot_time_observation(local_time)
    np.testing.assert_equal(observation, ZERO)


def test_get_one_hot_time_observation_one():
    local_time = datetime(2025, 5, 7, 10, 0, 0, tzinfo=tz)
    observation = get_one_hot_time_observation(local_time)
    np.testing.assert_equal(observation, ONE)
    local_time = datetime(2025, 5, 7, 10, 0, 1, tzinfo=tz)
    observation = get_one_hot_time_observation(local_time)
    np.testing.assert_equal(observation, ONE)
    local_time = datetime(2025, 5, 7, 10, 59, 59, tzinfo=tz)
    observation = get_one_hot_time_observation(local_time)
    np.testing.assert_equal(observation, ONE)


def test_get_one_hot_time_observation_two():
    local_time = datetime(2025, 5, 7, 11, 0, 0, tzinfo=tz)
    observation = get_one_hot_time_observation(local_time)
    np.testing.assert_equal(observation, TWO)
    local_time = datetime(2025, 5, 7, 11, 0, 1, tzinfo=tz)
    observation = get_one_hot_time_observation(local_time)
    np.testing.assert_equal(observation, TWO)
    local_time = datetime(2025, 5, 7, 11, 59, 59, tzinfo=tz)
    observation = get_one_hot_time_observation(local_time)
    np.testing.assert_equal(observation, TWO)


def test_get_one_hot_time_observation_three():
    local_time = datetime(2025, 5, 7, 12, 0, 0, tzinfo=tz)
    observation = get_one_hot_time_observation(local_time)
    np.testing.assert_equal(observation, THREE)
    local_time = datetime(2025, 5, 7, 12, 0, 1, tzinfo=tz)
    observation = get_one_hot_time_observation(local_time)
    np.testing.assert_equal(observation, THREE)
    local_time = datetime(2025, 5, 7, 12, 59, 59, tzinfo=tz)
    observation = get_one_hot_time_observation(local_time)
    np.testing.assert_equal(observation, THREE)


def test_get_one_hot_time_observation_four():
    local_time = datetime(2025, 5, 7, 13, 0, 0, tzinfo=tz)
    observation = get_one_hot_time_observation(local_time)
    np.testing.assert_equal(observation, FOUR)
    local_time = datetime(2025, 5, 7, 13, 0, 1, tzinfo=tz)
    observation = get_one_hot_time_observation(local_time)
    np.testing.assert_equal(observation, FOUR)
    local_time = datetime(2025, 5, 7, 13, 59, 59, tzinfo=tz)
    observation = get_one_hot_time_observation(local_time)
    np.testing.assert_equal(observation, FOUR)


def test_get_one_hot_time_observation_five():
    local_time = datetime(2025, 5, 7, 14, 0, 0, tzinfo=tz)
    observation = get_one_hot_time_observation(local_time)
    np.testing.assert_equal(observation, FIVE)
    local_time = datetime(2025, 5, 7, 14, 0, 1, tzinfo=tz)
    observation = get_one_hot_time_observation(local_time)
    np.testing.assert_equal(observation, FIVE)
    local_time = datetime(2025, 5, 7, 14, 59, 59, tzinfo=tz)
    observation = get_one_hot_time_observation(local_time)
    np.testing.assert_equal(observation, FIVE)


def test_get_one_hot_time_observation_six():
    local_time = datetime(2025, 5, 7, 15, 0, 0, tzinfo=tz)
    observation = get_one_hot_time_observation(local_time)
    np.testing.assert_equal(observation, SIX)
    local_time = datetime(2025, 5, 7, 15, 0, 1, tzinfo=tz)
    observation = get_one_hot_time_observation(local_time)
    np.testing.assert_equal(observation, SIX)
    local_time = datetime(2025, 5, 7, 15, 59, 59, tzinfo=tz)
    observation = get_one_hot_time_observation(local_time)
    np.testing.assert_equal(observation, SIX)


def test_get_one_hot_time_observation_seven():
    local_time = datetime(2025, 5, 7, 16, 0, 0, tzinfo=tz)
    observation = get_one_hot_time_observation(local_time)
    np.testing.assert_equal(observation, SEVEN)
    local_time = datetime(2025, 5, 7, 16, 0, 1, tzinfo=tz)
    observation = get_one_hot_time_observation(local_time)
    np.testing.assert_equal(observation, SEVEN)
    local_time = datetime(2025, 5, 7, 16, 59, 59, tzinfo=tz)
    observation = get_one_hot_time_observation(local_time)
    np.testing.assert_equal(observation, SEVEN)


def test_get_one_hot_time_observation_eight():
    local_time = datetime(2025, 5, 7, 17, 0, 0, tzinfo=tz)
    observation = get_one_hot_time_observation(local_time)
    np.testing.assert_equal(observation, EIGHT)
    local_time = datetime(2025, 5, 7, 17, 0, 1, tzinfo=tz)
    observation = get_one_hot_time_observation(local_time)
    np.testing.assert_equal(observation, EIGHT)
    local_time = datetime(2025, 5, 7, 17, 59, 59, tzinfo=tz)
    observation = get_one_hot_time_observation(local_time)
    np.testing.assert_equal(observation, EIGHT)


def test_get_one_hot_time_observation_nine():
    local_time = datetime(2025, 5, 7, 18, 0, 0, tzinfo=tz)
    observation = get_one_hot_time_observation(local_time)
    np.testing.assert_equal(observation, NINE)
    local_time = datetime(2025, 5, 7, 18, 0, 1, tzinfo=tz)
    observation = get_one_hot_time_observation(local_time)
    np.testing.assert_equal(observation, NINE)
    local_time = datetime(2025, 5, 7, 18, 59, 59, tzinfo=tz)
    observation = get_one_hot_time_observation(local_time)
    np.testing.assert_equal(observation, NINE)


def test_get_one_hot_time_observation_ten():
    local_time = datetime(2025, 5, 7, 19, 0, 0, tzinfo=tz)
    observation = get_one_hot_time_observation(local_time)
    np.testing.assert_equal(observation, TEN)
    local_time = datetime(2025, 5, 7, 19, 0, 1, tzinfo=tz)
    observation = get_one_hot_time_observation(local_time)
    np.testing.assert_equal(observation, TEN)
    local_time = datetime(2025, 5, 7, 19, 59, 59, tzinfo=tz)
    observation = get_one_hot_time_observation(local_time)
    np.testing.assert_equal(observation, TEN)


def test_get_one_hot_time_observation_eleven():
    local_time = datetime(2025, 5, 7, 20, 0, 0, tzinfo=tz)
    observation = get_one_hot_time_observation(local_time)
    np.testing.assert_equal(observation, ELEVEN)
    local_time = datetime(2025, 5, 7, 20, 0, 1, tzinfo=tz)
    observation = get_one_hot_time_observation(local_time)
    np.testing.assert_equal(observation, ELEVEN)
    local_time = datetime(2025, 5, 7, 20, 59, 59, tzinfo=tz)
    observation = get_one_hot_time_observation(local_time)
    np.testing.assert_equal(observation, ELEVEN)


def test_get_one_hot_time_observation_twelve():
    local_time = datetime(2025, 5, 7, 21, 0, 0, tzinfo=tz)
    observation = get_one_hot_time_observation(local_time)
    np.testing.assert_equal(observation, TWELVE)
    local_time = datetime(2025, 5, 7, 21, 0, 1, tzinfo=tz)
    observation = get_one_hot_time_observation(local_time)
    np.testing.assert_equal(observation, TWELVE)
    local_time = datetime(2025, 5, 7, 21, 59, 59, tzinfo=tz)
    observation = get_one_hot_time_observation(local_time)
    np.testing.assert_equal(observation, TWELVE)


def test_get_one_hot_time_observation_hours_before_9():
    for hour in range(0, 9):
        local_time = datetime(2025, 5, 7, hour, 0, 0, tzinfo=tz)
        observation = get_one_hot_time_observation(local_time)
        np.testing.assert_equal(observation, ZERO)

    local_time = datetime(2025, 5, 7, 8, 59, 0, tzinfo=tz)
    observation = get_one_hot_time_observation(local_time)
    np.testing.assert_equal(observation, ZERO)


def test_get_one_hot_time_observation_hours_after_21():
    for hour in range(22, 24):
        local_time = datetime(2025, 5, 7, hour, 0, 0, tzinfo=tz)
        observation = get_one_hot_time_observation(local_time)
        np.testing.assert_equal(observation, TWELVE)
