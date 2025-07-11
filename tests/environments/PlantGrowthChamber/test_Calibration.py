import json
from pathlib import Path

import numpy as np
import pytest

from environments.PlantGrowthChamber.Calibration import Calibration
from utils.constants import (
    BALANCED_ACTION_100,
    BALANCED_ACTION_105,
    BALANCED_ACTION_120,
    BLUE_ACTION,
    RED_ACTION,
)


def create_calibration_object(zone: str) -> Calibration:
    """
    Helper function to create a Calibration object.
    """
    # Create a dummy calibration file
    config_dir = (
        Path(__file__).parent.parent.parent.parent
        / "src"
        / "environments"
        / "PlantGrowthChamber"
        / "configs"
    )

    # Load real calibration data
    calibration_file = config_dir / f"{zone}.json"
    with open(calibration_file, "r") as f:
        calibration_data = json.load(f)["zone"]["calibration"]

    # Create a Calibration object
    calibration = Calibration(
        action=calibration_data["action"],
        blue=calibration_data["blue"],
        cool_white=calibration_data["cool_white"],
        warm_white=calibration_data["warm_white"],
        orange_red=calibration_data["orange_red"],
        red=calibration_data["red"],
        far_red=calibration_data["far_red"],
    )
    return calibration


@pytest.fixture
def calibration_z3() -> Calibration:
    """
    A pytest fixture to create a Calibration object for testing.
    """
    return create_calibration_object("alliance-zone03")


@pytest.fixture
def calibration_z10() -> Calibration:
    """
    A pytest fixture to create a Calibration object for testing.
    """
    return create_calibration_object("alliance-zone10")


def test_post_init(calibration_z3: Calibration):
    """
    Test that the __post_init__ method correctly loads the maximum values.
    """
    assert calibration_z3.maximum_values is not None
    assert isinstance(calibration_z3.maximum_values, np.ndarray)
    assert calibration_z3.maximum_values.shape == (6,)
    np.testing.assert_allclose(
        calibration_z3.maximum_values,
        np.array([126, 118, 95, 101, 75, 32.8]),
        atol=1e-1,
    )
    assert calibration_z3.safe_maximum_values is not None
    assert isinstance(calibration_z3.safe_maximum_values, np.ndarray)
    assert calibration_z3.safe_maximum_values.shape == (6,)
    np.testing.assert_allclose(
        calibration_z3.safe_maximum_values,
        np.array([114, 101, 80, 89, 29, 21.5]),
        atol=1e-1,
    )


def test_get_calibrated_value(calibration_z3: Calibration):
    """
    Test that the _get_calibrated_value method correctly returns a calibrated value.
    """
    # Test with valid inputs
    action_cal = [0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1]
    color_cal = [
        0.0,
        1.0,
        6.0,
        14.0,
        23.0,
        41.0,
        58.0,
        74.0,
        90.0,
        97.0,
        104.0,
        111.0,
        117.0,
    ]
    desired = 100.0
    calibrated_value = calibration_z3._get_calibrated_value(
        action_cal, color_cal, desired
    )
    np.testing.assert_allclose(calibrated_value, 0.871, atol=1e-3)

    # Test with desired value <= 0
    desired = 0.0
    calibrated_value = calibration_z3._get_calibrated_value(
        action_cal, color_cal, desired
    )
    assert calibrated_value == 0.0

    # Test with action_cal is None
    calibrated_value = calibration_z3._get_calibrated_value(None, color_cal, desired)
    assert calibrated_value == 0.0

    # Test with color_cal is None
    calibrated_value = calibration_z3._get_calibrated_value(action_cal, None, desired)
    assert calibrated_value == 0.0


def test_adjusted_action(calibration_z3: Calibration):
    """
    Test that the adjusted action is correctly calculated.
    """
    curtis_calibrated_action = np.array([0.398, 0.762, 0.324, 0.000, 0.332, 0.606])
    curtis_action = calibration_z3.decalibrated_action(curtis_calibrated_action)
    adjusted_action = curtis_action.copy()
    adjusted_action[:5] = adjusted_action[:5] / np.sum(adjusted_action[:5]) * 120
    adjusted_action[5] = curtis_action[5] / curtis_action[4] * adjusted_action[4]

    np.testing.assert_allclose(
        adjusted_action,
        BALANCED_ACTION_120,
        atol=1e-1,
    )


def test_get_ppfd(calibration_z3: Calibration):
    """
    Test that the get_ppfd method correctly returns the PPFD value.
    """
    ppfd = calibration_z3.get_ppfd(BALANCED_ACTION_120)
    assert isinstance(ppfd, float)
    assert ppfd == 120
    ppfd = calibration_z3.get_ppfd(BALANCED_ACTION_105)
    assert isinstance(ppfd, float)
    assert ppfd == 105
    ppfd = calibration_z3.get_ppfd(BALANCED_ACTION_100)
    assert isinstance(ppfd, float)
    assert ppfd == 100


def test_get_calibrated_action(calibration_z3: Calibration):
    """
    Test that the get_calibrated_action method correctly returns a calibrated action.
    """
    calibrated_action = calibration_z3.get_calibrated_action(BALANCED_ACTION_120)
    assert isinstance(calibrated_action, np.ndarray)
    assert calibrated_action.shape == (6,)
    np.testing.assert_allclose(
        calibrated_action,
        np.array([0.397, 0.76, 0.324, 0.000, 0.332, 0.605]),
        atol=1e-3,
    )
    calibrated_action = calibration_z3.get_calibrated_action(BALANCED_ACTION_105)
    assert isinstance(calibrated_action, np.ndarray)
    assert calibrated_action.shape == (6,)
    np.testing.assert_allclose(
        calibrated_action,
        np.array([0.382, 0.693, 0.315, 0.000, 0.323, 0.605]),
        atol=1e-3,
    )


def test_decalibrated_action(calibration_z3: Calibration):
    """
    Test that the decalibrated_action method correctly decalibrates an action.
    """
    calibrated_action = np.array(
        [
            0.39722222222222225,
            0.76,
            0.32357142857142857,
            0.0,
            0.33199999999999996,
            0.6053354652215281,
        ]
    )
    decalibrated_action = calibration_z3.decalibrated_action(calibrated_action)
    assert isinstance(decalibrated_action, np.ndarray)
    assert decalibrated_action.shape == (6,)
    assert np.allclose(
        decalibrated_action,
        BALANCED_ACTION_120,
        atol=1e-1,
    )
    calibrated_action = np.array(
        [0.38159722, 0.69296875, 0.31526786, 0.0, 0.323, 0.605335]
    )
    decalibrated_action = calibration_z3.decalibrated_action(calibrated_action)
    assert isinstance(decalibrated_action, np.ndarray)
    assert decalibrated_action.shape == (6,)
    assert np.allclose(
        decalibrated_action,
        BALANCED_ACTION_105,
        atol=1e-1,
    )


def test_blue_action(calibration_z3: Calibration):
    """
    Test that the red action is correctly calibrated.
    """
    np.testing.assert_allclose(
        BLUE_ACTION,
        np.array([69.7, 29.3, 3.4, 0.0, 2.6, 5.8]),
        atol=1e-1,
    )

    # make sure ppfd is 105
    ppfd = calibration_z3.get_ppfd(BLUE_ACTION)
    assert np.isclose(ppfd, 105, atol=1)

    # check that it doesn't exceed the maximum safe values
    safe_maximum_values = calibration_z3.safe_maximum_values.copy()
    safe_maximum_values[4] = 62.0  # Ignoring chamber 7 outlier
    np.testing.assert_array_compare(
        np.less_equal,
        BLUE_ACTION,
        safe_maximum_values,
    )

    calibrated_action = calibration_z3.get_calibrated_action(BLUE_ACTION)

    assert isinstance(calibrated_action, np.ndarray)
    assert calibrated_action.shape == (6,)

    # check if between 0 and 1
    assert np.all(calibrated_action >= 0)
    assert np.all(calibrated_action <= 1)

    # check sum < 4
    assert np.sum(calibrated_action) < 4


def test_red_action(calibration_z3: Calibration):
    """
    Test that the red action is correctly calibrated.
    """
    np.testing.assert_allclose(
        RED_ACTION,
        np.array([9.7, 34.9, 4.0, 0.0, 56.3, 7.0]),
        atol=1e-1,
    )

    # make sure ppfd is 105
    ppfd = calibration_z3.get_ppfd(RED_ACTION)
    assert np.isclose(ppfd, 105, atol=1)

    # check that it doesn't exceed the maximum safe values
    safe_maximum_values = calibration_z3.safe_maximum_values.copy()
    safe_maximum_values[4] = 62.0  # Ignoring chamber 7 outlier
    np.testing.assert_array_compare(
        np.less_equal,
        RED_ACTION,
        safe_maximum_values,
    )

    calibrated_action = calibration_z3.get_calibrated_action(RED_ACTION)

    assert isinstance(calibrated_action, np.ndarray)
    assert calibrated_action.shape == (6,)

    # check if between 0 and 1
    assert np.all(calibrated_action >= 0)
    assert np.all(calibrated_action <= 1)

    # check sum < 4
    assert np.sum(calibrated_action) < 4
