from unittest.mock import MagicMock

import numpy as np

from algorithms.DiscreteRandomAgent import DiscreteRandomAgent


class TestDiscreteRandomAgent:
    def test_sample_action_uniformity(self):
        """
        Test that sample_action returns actions uniformly at random.
        """
        n_actions = 3
        agent = DiscreteRandomAgent(
            observations=(1,),
            actions=n_actions,
            params={},
            collector=MagicMock(),
            seed=42,
        )
        n_samples = 5000
        actions = [agent.sample_action() for _ in range(n_samples)]
        counts = np.bincount(actions, minlength=n_actions)
        expected = 1 / n_actions
        # Allow 5% tolerance
        np.testing.assert_allclose(counts / n_samples, expected, atol=0.05)

    def test_checkpointing(self, tmpdir, setup_checkpoint_test):
        """Test that the agent state can be saved and loaded via checkpointing."""

        # Create agent with specific parameters
        params = {"seed": 123}

        # Define initialization function
        def init_agent(agent):
            # For example, sample an action to set internal state
            agent.sample_action()
            return agent

        # Use the common checkpoint test utility
        original_agent, loaded_agent = setup_checkpoint_test(
            tmpdir, params, DiscreteRandomAgent, actions=3, init_func=init_agent
        )

        # Verify agent state was properly restored
        # Add assertions specific to this agent type

        # Verify agent behavior is consistent
        obs = np.array([0])
        extra = {}

        original_action, _ = original_agent.step(1.0, obs, extra)

        loaded_action, _ = loaded_agent.step(1.0, obs, extra)

        # Actions should be the same if the state was properly restored
        assert original_action == loaded_action, (
            "Restored agent produces different actions"
        )
