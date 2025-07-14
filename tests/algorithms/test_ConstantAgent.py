from unittest.mock import MagicMock

import numpy as np

from algorithms.ConstantAgent import ConstantAgent


class TestConstantAgent:
    def test_constant_action(self):
        """
        Test that the agent always returns the specified constant action.
        """
        # Test with different constant action values
        test_cases = [0, 1, 2, 5]

        for constant_action in test_cases:
            # Create agent with specific constant action
            agent = ConstantAgent(
                observations=(1,),
                actions=10,  # Number of actions should be greater than constant_action
                params={"constant_action": constant_action},
                collector=MagicMock(),
                seed=42,
            )

            # Test start method
            obs = np.array([0])
            extra = {}
            action, _ = agent.start(obs, extra)
            assert action == constant_action, (
                f"Expected action {constant_action}, got {action}"
            )

            # Test step method
            action, _ = agent.step(1.0, obs, extra)
            assert action == constant_action, (
                f"Expected action {constant_action}, got {action}"
            )

    def test_default_action(self):
        """Test that the default action is 1 if not specified"""
        agent = ConstantAgent(
            observations=(1,), actions=2, params={}, collector=MagicMock(), seed=42
        )

        assert agent.action == 1, "Default action should be 1"

    def test_checkpointing(self, tmpdir, setup_checkpoint_test):
        """Test that the agent state can be saved and loaded via checkpointing."""

        # Create agent with specific parameters
        constant_action = 3
        params = {"constant_action": constant_action, "seed": 123}

        # Use the common checkpoint test utility
        original_agent, loaded_agent = setup_checkpoint_test(
            tmpdir,
            params,
            ConstantAgent,
            actions=5,  # Must be greater than constant_action
        )

        # Verify agent state was properly restored
        assert loaded_agent.action == original_agent.action, (
            "Constant action parameter not restored correctly"
        )
        assert loaded_agent.action == constant_action, (
            f"Expected constant action {constant_action}, got {loaded_agent.action}"
        )

        # Verify agent behavior is consistent
        obs = np.array([0])
        extra = {}

        original_action, _ = original_agent.step(1.0, obs, extra)
        loaded_action, _ = loaded_agent.step(1.0, obs, extra)

        # Actions should be the same if the state was properly restored
        assert original_action == loaded_action, (
            "Restored agent produces different actions"
        )
        assert original_action == constant_action, (
            f"Original agent should return constant action {constant_action}"
        )
        assert loaded_action == constant_action, (
            f"Loaded agent should return constant action {constant_action}"
        )
