from typing import Any, Dict, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces

from RlGlue.environment import BaseEnvironment


class KArmedBandit(BaseEnvironment):
    """
    K-Armed Bandit environment as described in Sutton and Barto's Reinforcement Learning book (Chapter 2.3).

    This environment simulates the k-armed bandit problem where:
    - Each of the k actions has a true action value (q*(a)) sampled from a normal distribution with mean 0 and variance 1
    - When an action is selected, the reward is sampled from a normal distribution with mean q*(a) and variance 1
    - The goal is to maximize total reward by learning which actions yield the highest rewards

    By default, it simulates the 10-armed testbed described in the book.

    This implementation follows the RLGlue BaseEnvironment interface.
    """

    def __init__(self, k=10, mean=0.0, variance=1.0, reward_variance=1.0, max_steps=1000, seed=None):
        """
        Initialize the k-armed bandit environment.

        Parameters:
        -----------
        k : int
            Number of arms (actions) in the bandit
        mean : float
            Mean of the normal distribution from which true action values are drawn
        variance : float
            Variance of the normal distribution from which true action values are drawn
        reward_variance : float
            Variance of the reward noise distribution
        max_steps : int
            Maximum number of steps before the episode terminates
        seed : int, optional
            Random seed for reproducibility
        """
        self.k = k
        self.mean = mean
        self.variance = variance
        self.reward_variance = reward_variance
        self.max_steps = max_steps
        self.seed = seed

        # Set random number generator
        self.rng = np.random.RandomState(seed)

        # Initialize attributes that will be set in start()
        self.true_action_values = None
        self.steps = 0
        self.action_counts = None
        self.total_reward = 0
        self.action_history = None
        self.reward_history = None
        self.optimal_action = None
        self.optimal_action_count = 0

    def start(self) -> Any:
        """
        Reset the environment to its initial state and return the initial observation.

        Returns:
        --------
        observation : int
            The initial observation (always 0 for bandits)
        """
        # Generate true action values q*(a) from N(mean, variance)
        self.true_action_values = self.rng.normal(
            self.mean, np.sqrt(self.variance), size=self.k
        )

        # Reset step counter
        self.steps = 0

        # Reset statistics
        self.action_counts = np.zeros(self.k, dtype=int)
        self.total_reward = 0
        self.action_history = []
        self.reward_history = []
        self.optimal_action = np.argmax(self.true_action_values)
        self.optimal_action_count = 0

        # Initial observation is always 0 in bandit problems
        observation = 0

        return observation

    def step(self, action: int) -> Tuple[float, Any, bool, Dict[str, Any]]:
        """
        Take an action in the environment.

        Parameters:
        -----------
        action : int
            The arm to pull (0 to k-1)

        Returns:
        --------
        reward : float
            The reward received for the action
        observation : int
            The next observation (always 0 for bandits)
        terminal : bool
            Whether the episode has terminated
        info : dict
            Additional information
        """
        assert 0 <= action < self.k, f"Invalid action: {action}"

        # Generate reward based on selected action
        # Reward is sampled from N(q*(action), reward_variance)
        true_value = self.true_action_values[action]
        reward = self.rng.normal(true_value, np.sqrt(self.reward_variance))

        # Update statistics
        self.steps += 1
        self.action_counts[action] += 1
        self.total_reward += reward
        self.action_history.append(action)
        self.reward_history.append(reward)

        if action == self.optimal_action:
            self.optimal_action_count += 1

        # Check if episode is done (RLGlue only uses terminal, not truncated)
        terminal = self.steps >= self.max_steps

        # Next observation is always 0 in bandit problems
        observation = 0

        # Information dictionary
        info = {
            "action_counts": self.action_counts.copy(),
            "true_action_value": true_value,
            "optimal_action": self.optimal_action,
            "is_optimal": action == self.optimal_action,
            "optimal_action_percentage": self.optimal_action_count / self.steps,
            "average_reward": self.total_reward / self.steps,
        }

        return reward, observation, terminal, info

    def get_num_actions(self) -> int:
        """
        Get the number of possible actions in the environment.

        Returns:
        --------
        int : Number of actions
        """
        return self.k

    def render(self, mode='human'):
        """
        Renders the current state of the bandit problem.

        Parameters:
        -----------
        mode : str
            The rendering mode ('human' or 'rgb_array')

        Returns:
        --------
        If mode is 'rgb_array', return an RGB array representation of the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot true action values and estimated values
        estimated_values = np.zeros(self.k)
        for a in range(self.k):
            if self.action_counts[a] > 0:
                # Simple averaging for demonstration
                action_indices = [i for i, act in enumerate(self.action_history) if act == a]
                rewards = [self.reward_history[i] for i in action_indices]
                estimated_values[a] = np.mean(rewards) if rewards else 0

        actions = np.arange(self.k)
        width = 0.35

        ax1.bar(actions - width/2, self.true_action_values, width, label='True Values')
        ax1.bar(actions + width/2, estimated_values, width, label='Estimated Values')
        ax1.set_xlabel('Action')
        ax1.set_ylabel('Value')
        ax1.set_title('True vs Estimated Action Values')
        ax1.set_xticks(actions)
        ax1.legend()

        # Plot action counts
        ax2.bar(actions, self.action_counts)
        ax2.axvline(x=self.optimal_action, color='r', linestyle='--', label='Optimal Action')
        ax2.set_xlabel('Action')
        ax2.set_ylabel('Count')
        ax2.set_title('Action Counts')
        ax2.set_xticks(actions)
        ax2.legend()

        plt.tight_layout()

        if mode == 'human':
            plt.show()
        elif mode == 'rgb_array':
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return image

    def close(self):
        """Close the environment."""
        plt.close()

    @classmethod
    def create_testbed(cls, num_bandits=2000, k=10, runs=1000, mean=0.0, variance=1.0,
                      reward_variance=1.0, seed=None):
        """
        Create a testbed of multiple k-armed bandit problems for evaluation.

        Parameters:
        -----------
        num_bandits : int
            Number of different bandit problems to create
        k : int
            Number of arms in each bandit
        runs : int
            Number of steps to run for each bandit
        mean, variance, reward_variance : float
            Parameters for the bandit problems
        seed : int, optional
            Base random seed

        Returns:
        --------
        list of KArmedBandit instances
        """
        testbed = []
        for i in range(num_bandits):
            # If seed is provided, use different seeds for each bandit
            bandit_seed = None if seed is None else seed + i
            bandit = cls(k=k, mean=mean, variance=variance,
                         reward_variance=reward_variance, max_steps=runs,
                         seed=bandit_seed)
            testbed.append(bandit)
        return testbed


# Keep the Gymnasium version as a separate class for backward compatibility
class KArmedBanditGym(gym.Env):
    """
    K-Armed Bandit environment using Gymnasium interface.
    This class provides compatibility with Gymnasium-based code.
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(self, k=10, mean=0.0, variance=1.0, reward_variance=1.0, max_steps=1000):
        super().__init__()

        self.bandit = KArmedBandit(
            k=k,
            mean=mean,
            variance=variance,
            reward_variance=reward_variance,
            max_steps=max_steps
        )

        # Define action and observation spaces
        self.action_space = spaces.Discrete(k)
        self.observation_space = spaces.Discrete(1)  # Only one state in bandit problems

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.bandit.seed = seed
            self.bandit.rng = np.random.RandomState(seed)

        observation = self.bandit.start()
        info = {
            "optimal_action": self.bandit.optimal_action,
            "true_action_values": self.bandit.true_action_values.copy()
        }

        return observation, info

    def step(self, action):
        reward, observation, terminal, info = self.bandit.step(action)

        # In Gymnasium, 'terminated' and 'truncated' are separate
        # For bandits, we're using 'truncated' for reaching max steps
        terminated = False
        truncated = terminal

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        return self.bandit.render(mode=mode)

    def close(self):
        self.bandit.close()


if __name__ == "__main__":
    # Example usage with the RLGlue interface
    print("Testing RLGlue interface...")
    bandit = KArmedBandit(k=10, seed=42)
    observation = bandit.start()

    print("True action values:", bandit.true_action_values)
    print("Optimal action:", bandit.optimal_action)

    # Perform 1000 steps with a simple epsilon-greedy strategy
    epsilon = 0.1
    action_values = np.zeros(10)
    action_counts = np.zeros(10)

    for step in range(1000):
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = np.random.randint(bandit.k)  # Explore
        else:
            action = np.argmax(action_values)  # Exploit

        # Take action and observe reward
        reward, observation, terminal, info = bandit.step(action)

        # Update action value estimates
        action_counts[action] += 1
        action_values[action] += (reward - action_values[action]) / action_counts[action]

        if step % 100 == 0:
            print(f"Step {step}, Average Reward: {info['average_reward']:.4f}, "
                  f"Optimal Action %: {info['optimal_action_percentage']:.2f}")

        if terminal:
            break

    # Render the final state
    bandit.render()

    print("\nTesting Gymnasium interface...")
    env = KArmedBanditGym(k=10)
    obs, info = env.reset(seed=42)

    print("True action values:", info["true_action_values"])
    print("Optimal action:", info["optimal_action"])

    # Epsilon-greedy strategy as before
    epsilon = 0.1
    action_values = np.zeros(10)
    action_counts = np.zeros(10)

    for step in range(1000):
        if np.random.random() < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(action_values)  # Exploit

        obs, reward, terminated, truncated, info = env.step(action)

        action_counts[action] += 1
        action_values[action] += (reward - action_values[action]) / action_counts[action]

        if step % 100 == 0:
            print(f"Step {step}, Average Reward: {info['average_reward']:.4f}, "
                  f"Optimal Action %: {info['optimal_action_percentage']:.2f}")

        if terminated or truncated:
            break

    env.render()
