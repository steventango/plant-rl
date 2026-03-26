import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Calibration:
    """
    A class to handle the calibration of actions for the plant growth chamber.
    """

    action: list[float]
    blue: list[float] | None
    cool_white: list[float] | None
    warm_white: list[float] | None
    orange_red: list[float] | None
    red: list[float] | None
    far_red: list[float] | None
    maximum_values: np.ndarray = field(init=False)

    def __post_init__(self):
        """
        Load the maximum values from the calibration file after the object is created.
        """
        config_dir = Path(__file__).parent / "configs"
        maximum_file = config_dir / "calibration.json"
        with open(maximum_file) as f:
            data = json.load(f)
        keys = [
            "blue",
            "cool_white",
            "warm_white",
            "orange_red",
            "red",
            "far_red",
        ]
        self.maximum_values = np.array([data["maximum"][key] for key in keys])
        self.safe_maximum_values = np.array([data["safe_maximum"][key] for key in keys])
        self.safe_minimum_values = np.array([data["safe_minimum"][key] for key in keys])

    def _get_calibrated_value(
        self,
        action_cal: list[float] | None,
        color_cal: list[float] | None,
        desired: float,
        safe_min_ppfd: float = 0.0,
    ) -> float:
        """
        Get the calibrated value for a single color channel.
        """
        if action_cal is None or color_cal is None or desired <= 0:
            return desired if desired > 0 else 0

        # If desired PPFD is below the safe minimum, at least one lightbar cannot
        # emit anything at this level — zero all bars for this channel.
        if safe_min_ppfd > 0 and desired < safe_min_ppfd:
            return 0.0

        color_array = np.array(color_cal)
        action_array = np.array(action_cal)
        action_array[color_array == 0] = 0

        # Interpolate the desired value
        return float(np.clip(np.interp(desired, color_array, action_array), 0, action_array.max()))

    def get_calibrated_action(self, action: np.ndarray) -> np.ndarray:
        """
        Get the calibrated action based on the maximum values and the action.
        """
        if len(action) != 6:
            raise ValueError("Action must be a 6-dimensional vector.")

        calibration_data = [
            self.blue,
            self.cool_white,
            self.warm_white,
            self.orange_red,
            self.red,
            self.far_red,
        ]

        def _calibrate(a: np.ndarray) -> np.ndarray:
            return np.array(
                [
                    self._get_calibrated_value(self.action, color, desired, float(safe_min))
                    for desired, color, safe_min in zip(
                        a, calibration_data, self.safe_minimum_values, strict=True
                    )
                ]
            )

        target_ppfd = self.get_ppfd(action)

        active = action >= self.safe_minimum_values
        active_ppfd = float(np.sum(np.where(active[:5], action[:5], 0.0)))
        if target_ppfd > 0 and active_ppfd > 0 and abs(active_ppfd - target_ppfd) > 0.5:
            scale = target_ppfd / active_ppfd
            scaled_action = np.where(
                active,
                np.minimum(action * scale, self.safe_maximum_values),
                action,
            )
        else:
            scaled_action = action
        return _calibrate(scaled_action)

    def decalibrated_action(self, calibrated_action: np.ndarray) -> np.ndarray:
        """
        Get the uncalibrated action based on the maximum values and the action.
        """
        if len(calibrated_action) != 6:
            raise ValueError("Action must be a 6-dimensional vector.")

        calibration_data = [
            self.blue,
            self.cool_white,
            self.warm_white,
            self.orange_red,
            self.red,
            self.far_red,
        ]

        uncalibrated_action = np.array(
            [
                self._get_calibrated_value(color, self.action, desired)
                for desired, color in zip(
                    calibrated_action, calibration_data, strict=True
                )
            ]
        )

        return uncalibrated_action

    def get_ppfd(self, uncalibrated_action: np.ndarray) -> float:
        """
        Get the PPFD (Photosynthetic Photon Flux Density) based on the uncalibrated action.
        """
        if len(uncalibrated_action) != 6:
            raise ValueError("Action must be a 6-dimensional vector.")

        # Calculate the PPFD as the sum of the scaled action values
        ppfd = np.sum(uncalibrated_action[:5])
        return ppfd

    def to_dict(self) -> dict:
        """
        Convert the Calibration object to a dictionary, excluding maximum_values.
        """
        return {
            "action": self.action,
            "blue": self.blue,
            "cool_white": self.cool_white,
            "warm_white": self.warm_white,
            "orange_red": self.orange_red,
            "red": self.red,
            "far_red": self.far_red,
        }
