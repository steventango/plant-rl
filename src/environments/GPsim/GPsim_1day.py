import logging
import pickle
import numpy as np
from RlGlue.environment import BaseEnvironment
from utils.metrics import UnbiasedExponentialMovingAverage as uema
import os
from utils.constants import BALANCED_ACTION_105, BLUE_ACTION, RED_ACTION

logger = logging.getLogger("rlglue")
logger.setLevel(logging.DEBUG)


class GPsim_1day(BaseEnvironment):
    def __init__(self, stochastic_pred=True, optimism=0, episode_length=13, **kwargs):
        self.state_dim = (1,)
        self.current_state = np.empty(1)
        self.action_dim = 3
        self.num_steps = 0
        self.gamma = 0.99

        self.stochastic_pred = stochastic_pred
        self.optimism = optimism  # 0 returns mean prediction, 1 predicts mean + 1stdev
        self.episode_length = episode_length  # days

        data_path = (
            os.path.dirname(os.path.abspath(__file__))
            + "/models/E13_every_size_1day_trace5_delta.pickle"
        )
        with open(data_path, "rb") as f:
            self.GP_model = pickle.load(f)

    def get_observation(self, action):
        trace5 = [self.action_trace5[i].compute().item() for i in range(3)]
        input = np.vstack([np.hstack([[self.current_area], action, trace5])])

        if self.stochastic_pred == False:
            predictive_mean, predictive_std = self.GP_model.predict_mean_std(input)
            next_area = (
                self.current_area + predictive_mean + self.optimism * predictive_std
            )
        else:
            sampled_predictions = self.GP_model.sample_output(input, N=100)
            percentile = 50 + self.optimism * 34.1
            next_area = self.current_area + np.percentile(
                sampled_predictions, percentile
            )

        return next_area

    def start(self):
        self.num_steps = 0
        self.action_trace5 = [uema(alpha=0.5), uema(alpha=0.5), uema(alpha=0.5)]
        self.current_area = np.random.uniform(10, 60)
        self.current_state = np.array([self.current_area])

        return self.current_state, self.get_info()

    def step(self, action):
        self.num_steps += 1

        # Convert 6D action space to 3D
        action = self.compute_action_coefficients(action)

        # Update action traces
        for i in range(3):
            self.action_trace5[i].update(action[i])

        next_area = self.get_observation(action)

        reward = next_area / self.current_area - 1

        self.current_area = next_area
        self.current_state = np.array([self.current_area])

        if self.num_steps == self.episode_length:
            return reward, self.current_state, True, self.get_info()
        else:
            return reward, self.current_state, False, self.get_info()

    def get_info(self):
        return {"gamma": self.gamma}

    def compute_action_coefficients(self, action: np.ndarray) -> np.ndarray:
        """
        Derive action coefficients by projecting action onto the basis spanned by RED, WHITE, BLUE.

        Solves: action ≈ coef[0] * RED + coef[1] * WHITE + coef[2] * BLUE

        Args:
            action: Array of shape (6,) representing the action vector

        Returns:
            coefficients: Array of shape (3,) with [red_coef, white_coef, blue_coef]
        """
        # Create basis matrix where each column is a basis vector
        basis = np.column_stack(
            [RED_ACTION, BALANCED_ACTION_105, BLUE_ACTION]
        )  # Shape: (6, 3)

        # Solve least squares: basis @ coefficients = action
        # This finds coefficients that minimize ||action - basis @ coefficients||^2
        coefficients, residuals, rank, s = np.linalg.lstsq(basis, action, rcond=None)

        return coefficients
