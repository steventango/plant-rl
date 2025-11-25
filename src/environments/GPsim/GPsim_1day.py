import logging
import pickle
import numpy as np
from RlGlue.environment import BaseEnvironment
from utils.metrics import UnbiasedExponentialMovingAverage as uema
import os
from utils.constants import BALANCED_ACTION_105, BLUE_ACTION, RED_ACTION
from jax import random

logger = logging.getLogger("rlglue")
logger.setLevel(logging.DEBUG)


class GPsim_1day(BaseEnvironment):
    def __init__(self, stochastic_pred=True, optimism=0, episode_length=13, seed=0, **kwargs):
        self.state_dim = (5,)
        self.current_state = np.empty(5)
        self.action_dim = 6
        self.num_steps = 0
        self.trace_beta = 0.5
        self.key = random.PRNGKey(seed)
        self.rng = np.random.default_rng(seed)

        self.stochastic_pred = stochastic_pred
        self.optimism = optimism  # 0 returns mean prediction, 1 predicts mean + 1stdev
        self.episode_length = episode_length  # days

        data_path = (
            os.path.dirname(os.path.abspath(__file__))
            + f"/models/E13only_every_day+size_1day_trace{int(self.trace_beta*10)}_ratio.pickle"
        )
        with open(data_path, "rb") as f:
            self.GP_model = pickle.load(f)

    def get_observation(self, action):
        trace = [self.action_trace[i].compute().item() for i in range(3)]   
        
        input = np.vstack(
            [np.hstack([[self.num_steps, self.current_area], action, trace])]
        )
        
        predictive_mean, predictive_std = self.GP_model.predict_mean_std(input)
        if not self.stochastic_pred:            
            next_area = self.current_area * (
                predictive_mean[0] + self.optimism * predictive_std[0]
            )
        else:
            sampled_predictions = self.sample_output(predictive_mean, predictive_std, N=100)
            percentile = 50 + self.optimism * 34.1
            next_area = self.current_area * np.percentile(
                sampled_predictions, percentile
            )

        return next_area

    def start(self):
        self.num_steps = 0
        alpha = 1 - self.trace_beta
        self.action_trace = [uema(alpha=alpha), uema(alpha=alpha), uema(alpha=alpha)]
        self.current_area = self.rng.uniform(30, 120)
        self.initial_area = self.current_area
        self.current_state = np.array(
            [self.num_steps / 14, self.current_area / self.initial_area / 10, 0, 0, 0]
        )
        
        return self.current_state, self.get_info()

    def step(self, action_rwb: np.ndarray):  # type:ignore
        self.num_steps += 1
        
        # Get observation from GP model            
        next_area = self.get_observation(action_rwb)

        # Update action traces
        for i in range(3):
            self.action_trace[i].update(action_rwb[i])
        trace = [self.action_trace[i].compute().item() for i in range(3)]  
        
        #reward = next_area / self.current_area - 1  # daily reward
        #reward = (reward - 0.2) * 1000  # rescaled
        
        reward = (next_area - self.current_area) / self.initial_area
        
        self.current_area = next_area
        self.current_state = np.array(
            [self.num_steps / 14, self.current_area / self.initial_area / 10] + trace
        )

        if self.num_steps == self.episode_length:
            return reward, self.current_state, True, self.get_info()
        else:
            return reward, self.current_state, False, self.get_info()

    def get_info(self):
        return {}

    def sample_output(self, mean, std, N=100):
        self.key, subkey = random.split(self.key)
        samples = (random.normal(subkey, (mean.shape[0], N)) * std[:, None]) + mean[
            :, None
        ]
        return samples