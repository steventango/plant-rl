from unittest.mock import MagicMock

import numpy as np

from algorithms.PoissonAgent import PoissonAgent


class TestPoissonAgent:
    def test_start_and_step_action_generation(self):
        """
        Test that start and step methods generate valid actions and repeat logic works.
        """
        n_actions = 3
        lam = 3.0
        max_repeat = 5
        agent = PoissonAgent(
            observations=(1,),
            actions=n_actions,
            params={"lam": lam, "max_repeat": max_repeat},
            collector=MagicMock(),
            seed=99,
        )
        obs = np.array([0])
        extra = {}
        # Start should generate a valid action
        action, info = agent.start(obs, extra)
        assert 0 <= action < n_actions
        assert agent.current_action == action
        assert agent.current_repeat is not None
        assert 0 <= agent.current_repeat <= max_repeat

        # Step should decrement repeat and only sample new action when repeat hits 0
        for _ in range(agent.current_repeat):
            prev_action = agent.current_action
            prev_repeat = agent.current_repeat
            action, _ = agent.step(1.0, obs, extra)
            assert isinstance(action, np.integer)
            # Should keep same action until repeat is 0
            if prev_repeat > 1:
                assert agent.current_repeat == prev_repeat - 1
                assert action == prev_action
            else:
                # After repeat reaches 0, a new action is sampled
                assert 0 <= action < n_actions
                assert agent.current_action == action
                assert agent.current_repeat is not None
                assert 0 <= agent.current_repeat <= max_repeat

    def test_sample_action_uniformity(self):
        """
        Test that sample_action returns actions uniformly at random.
        """
        n_actions = 3
        agent = PoissonAgent(
            observations=(1,),
            actions=n_actions,
            params={"lam": 3.0, "max_repeat": 5},
            collector=MagicMock(),
            seed=42,
        )
        n_samples = 5000
        actions = [agent.sample_action() for _ in range(n_samples)]
        counts = np.bincount(actions, minlength=n_actions)
        expected = 1 / n_actions
        # Allow 5% tolerance
        np.testing.assert_allclose(counts / n_samples, expected, atol=0.05)

    def test_repeat_distribution(self):
        """
        Test that the repeat count is Poisson distributed and capped by max_repeat.
        """
        lam = 3.0
        max_repeat = 5
        agent = PoissonAgent(
            observations=(1,),
            actions=3,
            params={"lam": lam, "max_repeat": max_repeat},
            collector=MagicMock(),
            seed=123,
        )
        n_samples = 5000
        repeats = []
        for _ in range(n_samples):
            agent.sample_action()
            repeats.append(agent.current_repeat)
        # All repeats should be <= max_repeat
        assert all(0 <= r <= max_repeat for r in repeats)
        # The mean should be close to min(lam, max_repeat)
        empirical_mean = np.mean(repeats)
        # Theoretical mean for truncated Poisson is less than lam, so just check it's close
        np.testing.assert_allclose(
            empirical_mean,
            min(lam, max_repeat),
            atol=0.3,
            err_msg="Mean repeat count mismatch",
        )

    def test_default_lambda(self):
        """Test that the default lambda is 3.0 if not specified"""
        agent = PoissonAgent(
            observations=(1,), actions=2, params={}, collector=MagicMock(), seed=42
        )
        assert agent.lam == 3.0, "Default lambda should be 3.0"

    def test_lambda_positive(self):
        """Test that the agent raises an assertion error if lambda <= 0"""
        try:
            PoissonAgent(
                observations=(1,),
                actions=2,
                params={"lam": 0},
                collector=MagicMock(),
                seed=42,
            )
            raise Exception("Should have raised an AssertionError")
        except AssertionError:
            pass
