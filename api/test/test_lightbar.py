import numpy as np

from ..lightbar import convert_to_duty_cycle


def test_convert_to_duty_cycle_zeros():
    action = np.zeros((2, 6))
    duty_cycle = convert_to_duty_cycle(action)
    assert duty_cycle.shape == (2, 6)
    np.testing.assert_array_equal(
        duty_cycle, np.array([[340, 300, 360, 350, 400, 400], [340, 300, 360, 350, 400, 400]])
    )


def test_convert_to_duty_cycle_ones():
    action = np.ones((2, 6))
    duty_cycle = convert_to_duty_cycle(action)
    assert duty_cycle.shape == (2, 6)
    np.testing.assert_array_equal(
        duty_cycle, np.array([[3000, 3000, 3000, 3200, 2600, 3600], [3000, 3000, 3000, 3200, 2600, 3600]])
    )


def test_convert_to_duty_cycle_half():
    action = np.ones((2, 6)) * 0.5
    duty_cycle = convert_to_duty_cycle(action)
    assert duty_cycle.shape == (2, 6)
    np.testing.assert_array_equal(
        duty_cycle, np.array([[1670, 1650, 1680, 1775, 1500, 2000], [1670, 1650, 1680, 1775, 1500, 2000]])
    )
