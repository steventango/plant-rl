from functools import lru_cache

import numpy as np
import pytest

from ..app.lightbar import Lightbar
from ..app.zones import Zone
from .mock_lightbar import MockLightbar


@lru_cache(maxsize=None)
def get_lightbar():
    return MockLightbar(Zone(0x69, 0x71))


@pytest.fixture
def lightbar():
    return get_lightbar()


class TestLightbar:
    def test_step(self, lightbar: Lightbar):
        lightbar.step(np.ones((2, 6)))
        for address in [0x69, 0x71]:
            assert lightbar.i2c.data[address][3][0] == [0, 0x06, 0, 0, 0x55, 0x05]
            assert lightbar.i2c.data[address][3][1] == [0, 0x0A, 0, 0, 0x55, 0x05]
            assert lightbar.i2c.data[address][3][2] == [0, 0x1E, 0, 0, 0x55, 0x05]
            assert lightbar.i2c.data[address][3][3] == [0, 0x1A, 0, 0, 0x55, 0x05]
            assert lightbar.i2c.data[address][3][4] == [0, 0x16, 0, 0, 0x55, 0x05]
            assert lightbar.i2c.data[address][3][5] == [0, 0x0E, 0, 0, 0x55, 0x05]

    def test_set_duty_cycle(self, lightbar: Lightbar):
        lightbar.set_duty_cycle(np.ones((2, 6), dtype=np.int32) * 1365)
        for address in [0x69, 0x71]:
            assert lightbar.i2c.data[address][3][0] == [0, 0x06, 0, 0, 0x55, 0x05]
            assert lightbar.i2c.data[address][3][1] == [0, 0x0A, 0, 0, 0x55, 0x05]
            assert lightbar.i2c.data[address][3][2] == [0, 0x1E, 0, 0, 0x55, 0x05]
            assert lightbar.i2c.data[address][3][3] == [0, 0x1A, 0, 0, 0x55, 0x05]
            assert lightbar.i2c.data[address][3][4] == [0, 0x16, 0, 0, 0x55, 0x05]
            assert lightbar.i2c.data[address][3][5] == [0, 0x0E, 0, 0, 0x55, 0x05]

    def test_get_command_array(self, lightbar: Lightbar):
        command_array = lightbar.get_command_array(0, 1365)
        assert command_array == [0, 0x06, 0, 0, 0x55, 0x05]

    def test_set_half_bar_pwm(self, lightbar: Lightbar):
        lightbar.set_half_bar_pwm(0x69, 0, 1365)
        assert lightbar.i2c.data[0x69][3][0] == [0, 0x06, 0, 0, 0x55, 0x05]
        lightbar.set_half_bar_pwm(0x71, 0, 1365)
        assert lightbar.i2c.data[0x71][3][0] == [0, 0x06, 0, 0, 0x55, 0x05]

    def test_ensure_safety_limits(self, lightbar: Lightbar):
        action = np.ones((2, 6))
        action = lightbar.ensure_safety_limits(action)
        assert action.shape == (2, 6)
        np.testing.assert_array_equal(action, np.ones((2, 6)) / 3)

        action = np.zeros((2, 6))
        action[0, 0] = 100
        action[1, :0] = -100
        action = lightbar.ensure_safety_limits(action)
        assert action.shape == (2, 6)
        np.testing.assert_array_equal(
            action, np.array([[2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
        )

        action = np.zeros((2, 6))
        action = lightbar.ensure_safety_limits(action)
        assert action.shape == (2, 6)
        np.testing.assert_array_equal(
            action, np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
        )

    def test_convert_to_duty_cycle_zeros(self, lightbar: Lightbar):
        action = np.zeros((2, 6))
        duty_cycle = lightbar.convert_to_duty_cycle(action)
        assert duty_cycle.shape == (2, 6)
        np.testing.assert_array_equal(duty_cycle, np.zeros((2, 6)))

    def test_convert_to_duty_cycle_ones(self, lightbar: Lightbar):
        action = np.ones((2, 6))
        duty_cycle = lightbar.convert_to_duty_cycle(action)
        assert duty_cycle.shape == (2, 6)
        np.testing.assert_array_equal(duty_cycle, np.ones((2, 6)) * 4095)

    def test_convert_to_duty_cycle_half(self, lightbar: Lightbar):
        action = np.ones((2, 6)) * 0.5
        duty_cycle = lightbar.convert_to_duty_cycle(action)
        assert duty_cycle.shape == (2, 6)
        np.testing.assert_array_equal(duty_cycle, np.ones((2, 6)) * 2047)


def test_get_lightbar_singleton():
    lightbar1 = get_lightbar()
    lightbar2 = get_lightbar()
    assert lightbar1 is lightbar2
