from unittest.mock import MagicMock

import numpy as np

from algorithms.BernoulliAgent import BernoulliAgent


class TestBernoulliAgent:
    def test_sample_action_distribution(self):
        """
        Test that sample_action returns actions with approximately the expected probability.
        We run multiple trials and check that the empirical probability is close to the expected.
        """
        # Test with different probability values
        test_cases = [
            {"p": 0.2, "tolerance": 0.05, "n_samples": 1000},
            {"p": 0.5, "tolerance": 0.05, "n_samples": 1000},
            {"p": 0.8, "tolerance": 0.05, "n_samples": 1000},
        ]

        for case in test_cases:
            p = case["p"]
            tolerance = case["tolerance"]
            n_samples = case["n_samples"]

            # Create agent with specific probability
            agent = BernoulliAgent(
                observations=(1,),
                actions=2,
                params={"p": p},
                collector=MagicMock(),
                seed=42,
            )

            # Sample actions multiple times
            actions = [agent.sample_action() for _ in range(n_samples)]

            # Calculate empirical probability of action 1
            empirical_p = sum(actions) / n_samples

            # Assert that empirical probability is close to the expected probability
            np.testing.assert_allclose(
                empirical_p,
                p,
                atol=tolerance,
                err_msg=f"Expected p={p}, got empirical p={empirical_p} after {n_samples} samples",
            )

    def test_default_probability(self):
        """Test that the default probability is 0.5 if not specified"""
        agent = BernoulliAgent(
            observations=(1,), actions=2, params={}, collector=MagicMock(), seed=42
        )

        assert agent.p == 0.5, "Default probability should be 0.5"

    def test_actions_constraint(self):
        """Test that the agent raises an assertion error if actions != 2"""
        # Should work with 2 actions
        BernoulliAgent(
            observations=(1,), actions=2, params={}, collector=MagicMock(), seed=42
        )

        # Should raise AssertionError with actions != 2
        try:
            BernoulliAgent(
                observations=(1,), actions=3, params={}, collector=MagicMock(), seed=42
            )
            raise Exception("Should have raised an AssertionError")
        except AssertionError:
            pass
