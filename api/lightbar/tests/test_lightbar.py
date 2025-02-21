from functools import lru_cache

import numpy as np
import pytest

from ..app.lightbar import Lightbar
from .mock_lightbar import MockLightbar


@lru_cache(maxsize=None)
def get_lightbar():
    return MockLightbar(0x69)


@pytest.fixture
def lightbar():
    return get_lightbar()


class TestLightbar:
    def test_step(self, lightbar: Lightbar):
        lightbar.step(np.ones(6))
        assert lightbar.i2c.data[0x69][3][0] == [0, 0x06, 0, 0, 0x55, 0x05]
        assert lightbar.i2c.data[0x69][3][1] == [0, 0x0A, 0, 0, 0x55, 0x05]
        assert lightbar.i2c.data[0x69][3][2] == [0, 0x1E, 0, 0, 0x55, 0x05]
        assert lightbar.i2c.data[0x69][3][3] == [0, 0x1A, 0, 0, 0x55, 0x05]
        assert lightbar.i2c.data[0x69][3][4] == [0, 0x16, 0, 0, 0x55, 0x05]
        assert lightbar.i2c.data[0x69][3][5] == [0, 0x0E, 0, 0, 0x55, 0x05]

    def test_set_duty_cycle(self, lightbar: Lightbar):
        lightbar.set_duty_cycle(np.ones(6, dtype=np.int32) * 1365)
        assert lightbar.i2c.data[0x69][3][0] == [0, 0x06, 0, 0, 0x55, 0x05]
        assert lightbar.i2c.data[0x69][3][1] == [0, 0x0A, 0, 0, 0x55, 0x05]
        assert lightbar.i2c.data[0x69][3][2] == [0, 0x1E, 0, 0, 0x55, 0x05]
        assert lightbar.i2c.data[0x69][3][3] == [0, 0x1A, 0, 0, 0x55, 0x05]
        assert lightbar.i2c.data[0x69][3][4] == [0, 0x16, 0, 0, 0x55, 0x05]
        assert lightbar.i2c.data[0x69][3][5] == [0, 0x0E, 0, 0, 0x55, 0x05]

    def test_get_command_array(self, lightbar: Lightbar):
        command_array = lightbar.get_command_array(0, 1365)
        assert command_array == [0, 0x06, 0, 0, 0x55, 0x05]

    def test_set_half_bar_pwm(self, lightbar: Lightbar):
        lightbar.set_half_bar_pwm(0, 1365)
        assert lightbar.i2c.data[0x69][3][0] == [0, 0x06, 0, 0, 0x55, 0x05]

    def test_ensure_safety_limits(self, lightbar: Lightbar):
        action = np.ones(6)
        action = lightbar.ensure_safety_limits(action)
        assert action.shape == (6,)
        np.testing.assert_array_equal(action, np.ones(6) / 3)

    def test_convert_to_duty_cycle_zeros(self, lightbar: Lightbar):
        action = np.zeros(6)
        duty_cycle = lightbar.convert_to_duty_cycle(action)
        assert duty_cycle.shape == (6,)
        np.testing.assert_array_equal(duty_cycle, np.zeros(6))

    def test_convert_to_duty_cycle_ones(self, lightbar: Lightbar):
        action = np.ones(6)
        duty_cycle = lightbar.convert_to_duty_cycle(action)
        assert duty_cycle.shape == (6,)
        np.testing.assert_array_equal(duty_cycle, np.ones(6) * 4095)

    def test_convert_to_duty_cycle_half(self, lightbar: Lightbar):
        action = np.ones(6) * 1 / 3
        duty_cycle = lightbar.convert_to_duty_cycle(action)
        assert duty_cycle.shape == (6,)
        np.testing.assert_array_equal(duty_cycle, np.ones(6) * 1365)


def test_get_lightbar_singleton():
    lightbar1 = get_lightbar()
    lightbar2 = get_lightbar()
    assert lightbar1 is lightbar2
