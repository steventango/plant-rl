from unittest.mock import MagicMock

import numpy as np

from algorithms.tc.batch.ESARSA import ESARSA


class TestBatchESARSA:
    def test_initialization(self):
        """
        Test that the agent initializes correctly with default parameters.
        """
        obs_shape = (10,)
        n_actions = 3
        params = {
            "epsilon": 0.1,
            "alpha": 0.1,
            "gamma": 0.99,
            "batch": "buffer",
            "batch_size": 32,
            "buffer_size": 1000,
            "buffer_type": "uniform",
            "representation": {
                "which_tc": "RichTileCoder",
                "strategy": -1,
                "tiles": 1,
                "tilings": 1,
            },
        }
        agent = ESARSA(
            observations=obs_shape,
            actions=n_actions,
            params=params,
            collector=MagicMock(),
            seed=42,
        )

        # Check basic initialization
        assert agent.epsilon == params["epsilon"]
        assert agent.alpha == params["alpha"]
        assert agent.w.shape == (n_actions, agent.n_features)
        # All weights should be initialized to zero
        assert np.allclose(agent.w, 0.0)

    def test_policy(self):
        """
        Test that the policy method returns valid action probabilities.
        """
        obs_shape = (5,)
        n_actions = 2
        params = {
            "epsilon": 0.1,
            "alpha": 0.1,
            "gamma": 0.99,
            "batch": "buffer",
            "batch_size": 32,
            "buffer_size": 1000,
            "buffer_type": "uniform",
            "representation": {
                "which_tc": "RichTileCoder",
                "strategy": -1,
                "tiles": 1,
                "tilings": 1,
            },
        }
        agent = ESARSA(
            observations=obs_shape,
            actions=n_actions,
            params=params,
            collector=MagicMock(),
            seed=42,
        )

        # Create a mock observation - should be a single observation
        obs = np.ones(agent.n_features)

        # Get policy probabilities
        pi = agent.policy(obs)

        # Verify probabilities sum to 1 and are in valid range
        assert np.isclose(np.sum(pi), 1.0)
        assert np.all(pi >= 0.0) and np.all(pi <= 1.0)

        # With zero weights and epsilon=0.1, should be mostly uniform
        expected_probs = np.array([0.5, 0.5])  # Uniform when all Q-values are equal
        assert np.allclose(pi, expected_probs)

        # Set some weights to make the Q-values different
        agent.w[0, 0] = 1.0
        pi = agent.policy(obs)

        # Now action 0 should have higher probability
        expected_probs = np.array([0.95, 0.05])
        assert np.allclose(pi, expected_probs)

    def test_checkpointing(self, tmpdir, setup_checkpoint_test):
        """Test that the agent state can be saved and loaded via checkpointing."""

        # Create agent with specific parameters
        obs_shape = (2,)
        n_actions = 3
        params = {
            "epsilon": 0.2,
            "alpha": 0.1,
            "gamma": 0.95,
            "batch": "buffer",
            "batch_size": 2,
            "buffer_size": 1000,
            "buffer_type": "uniform",
            "seed": 123,
            "representation": {
                "which_tc": "RichTileCoder",
                "strategy": -1,
                "tiles": 32,
                "tilings": 4,
            },
        }

        # Verify agent behavior is consistent
        obs = np.ones(obs_shape)

        # Define initialization function to set the agent state
        def init_agent(agent):
            # Set some non-zero weights to test checkpointing
            agent.w[0, 0] = 0.5
            agent.w[1, 1] = -0.3
            agent.w[2, 2] = 0.7

            # Store in agent.info too since it's part of the state
            agent.info["w"] = agent.w

            obs = np.zeros(obs_shape)
            agent.start(obs, {})
            num_steps = 10
            for i in range(num_steps):
                obs = np.ones(obs_shape) * (i + 1) / num_steps
                agent.step(1.0, obs, {})

            return agent

        # Use the common checkpoint test utility
        original_agent, loaded_agent = setup_checkpoint_test(
            tmpdir,
            params,
            ESARSA,
            observations=obs_shape,
            actions=n_actions,
            init_func=init_agent,
        )

        # Verify agent parameters were properly restored
        assert loaded_agent.epsilon == original_agent.epsilon
        assert loaded_agent.alpha == original_agent.alpha
        assert np.array_equal(loaded_agent.w, original_agent.w)

        # Verify buffer state is restored
        assert loaded_agent.buffer.size() == original_agent.buffer.size()

        # Verify tile coder state is restored
        assert loaded_agent.tile_coder._c == original_agent.tile_coder._c
        assert (
            loaded_agent.tile_coder.iht.dictionary
            == original_agent.tile_coder.iht.dictionary
        )

        # Get policies
        original_x = original_agent.get_rep(obs)
        original_pi = original_agent.policy(original_x)
        loaded_x = loaded_agent.get_rep(obs)
        loaded_pi = loaded_agent.policy(loaded_x)

        # Policies should be identical
        assert np.array_equal(original_pi, loaded_pi)

        # Get values
        original_values = original_agent.values(original_x)
        loaded_values = loaded_agent.values(loaded_x)

        # Values should be identical
        assert np.array_equal(original_values, loaded_values)

        # Weights should be the same before step
        assert np.array_equal(original_agent.w, loaded_agent.w), (
            "Restored agent weights do not match original"
        )

        # Test that step method works
        original_action, _ = original_agent.step(1.0, obs, {})
        loaded_action, _ = loaded_agent.step(1.0, obs, {})

        # Actions should be the same if the state was properly restored
        assert original_action == loaded_action, (
            "Restored agent produces different actions"
        )

        # Weights should be the same after step
        assert np.array_equal(original_agent.w, loaded_agent.w), (
            "Restored agent weights do not match original"
        )
