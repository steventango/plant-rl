from unittest.mock import MagicMock

import numpy as np

from algorithms.DirichletAgent import DirichletAgent
from utils.constants import BALANCED_ACTION_105, BLUE_ACTION, RED_ACTION


class TestDirichletAgent:
    def test_start_and_step_action_generation(self):
        """
        Test that start and step methods generate valid actions.
        """
        agent = DirichletAgent(
            observations=(1,),
            actions=6,
            params={},
            collector=MagicMock(),
            seed=99,
        )
        obs = np.array([0])
        extra = {}
        # Start should generate a valid action
        action, info = agent.start(obs, extra)
        assert isinstance(action, np.ndarray)
        assert action.shape == (6,)
        assert np.all(action >= 0)  # Actions should be non-negative

        # Step should also generate a valid action
        action, _ = agent.step(1.0, obs, extra)
        assert isinstance(action, np.ndarray)
        assert action.shape == (6,)
        assert np.all(action >= 0)

    def test_action_convex_combination(self):
        """
        Test that the action is a convex combination of RED, WHITE, BLUE actions.
        """
        agent = DirichletAgent(
            observations=(1,),
            actions=6,
            params={},
            collector=MagicMock(),
            seed=123,
        )
        n_samples = 100
        for _ in range(n_samples):
            action = agent.sample_action()
            # Action should be between the min and max of the three base actions
            min_action = np.minimum(
                np.minimum(RED_ACTION, BALANCED_ACTION_105), BLUE_ACTION
            )
            max_action = np.maximum(
                np.maximum(RED_ACTION, BALANCED_ACTION_105), BLUE_ACTION
            )
            assert np.all(action >= min_action - 1e-6)  # Allow small numerical errors
            assert np.all(action <= max_action + 1e-6)

    def test_action_ppfd_sum(self):
        """
        Test that the action PPFD sum is reasonable (around 105-117).
        """
        agent = DirichletAgent(
            observations=(1,),
            actions=6,
            params={},
            collector=MagicMock(),
            seed=456,
        )
        n_samples = 100
        sums = []
        for _ in range(n_samples):
            action = agent.sample_action()
            total_ppfd = np.sum(action[:5])
            sums.append(total_ppfd)
            np.testing.assert_approx_equal(total_ppfd, 105)
        # Mean should be around the average of the three
        mean_sum = np.mean(sums)
        expected_mean = (
            np.sum(RED_ACTION[:5])
            + np.sum(BALANCED_ACTION_105[:5])
            + np.sum(BLUE_ACTION[:5])
        ) / 3
        np.testing.assert_approx_equal(mean_sum, expected_mean)

    def test_deterministic_with_seed(self):
        """Test that the agent is deterministic with the same seed"""
        agent1 = DirichletAgent(
            observations=(1,), actions=6, params={}, collector=MagicMock(), seed=42
        )
        agent2 = DirichletAgent(
            observations=(1,), actions=6, params={}, collector=MagicMock(), seed=42
        )
        action1 = agent1.sample_action()
        action2 = agent2.sample_action()
        np.testing.assert_array_equal(action1, action2)
