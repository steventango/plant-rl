import os
from unittest.mock import MagicMock

import numpy as np
from PyExpUtils.models.ExperimentDescription import ExperimentDescription

from algorithms.SpreadsheetAgent import SpreadsheetAgent
from utils.checkpoint import Checkpoint


class TestSpreadsheetAgent:
    def test_get_action_single_day_cycle(self):
        agent = SpreadsheetAgent(
            observations=(1,),
            actions=6,
            params={
                "filepath": "tests/test_data/z3-0min-100ppfd-Balanced_optima12_12.xlsx"
            },
            collector=None,
            seed=0,
        )

        cycles = 3

        for cycle in range(cycles):
            # 12:00:00 AM
            action = agent.get_action(cycle * 86400 + 0 - agent.offset)
            np.testing.assert_almost_equal(action, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            # 8:59:59 AM
            action = agent.get_action(
                cycle * 86400 + 8 * 3600 + 59 * 60 + 59 - agent.offset
            )
            np.testing.assert_almost_equal(action, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            # 9:00:00 AM
            action = agent.get_action(cycle * 86400 + 9 * 3600 - agent.offset)
            np.testing.assert_almost_equal(
                action, [0.199, 0.381, 0.162, 0.0, 0.166, 0.303]
            )

            # 9:00:01 AM
            action = agent.get_action(cycle * 86400 + 9 * 3600 + 1 - agent.offset)
            np.testing.assert_almost_equal(
                action, [0.199, 0.381, 0.162, 0.0, 0.166, 0.303]
            )

            # 8:59:59 PM
            action = agent.get_action(
                cycle * 86400 + 20 * 3600 + 59 * 60 + 59 - agent.offset
            )
            np.testing.assert_almost_equal(
                action, [0.199, 0.381, 0.162, 0.0, 0.166, 0.303]
            )

            # 9:00:00 PM
            action = agent.get_action(cycle * 86400 + 21 * 3600 - agent.offset)
            np.testing.assert_almost_equal(
                action, [0.199, 0.381, 0.162, 0.0, 0.166, 0.303]
            )

            # 9:00:01 PM
            action = agent.get_action(cycle * 86400 + 21 * 3600 + 1 - agent.offset)
            np.testing.assert_almost_equal(action, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_get_action_multi_day_cycle(self):
        agent = SpreadsheetAgent(
            observations=(1,),
            actions=6,
            params={"filepath": "tests/test_data/multi_day_cycle.xlsx"},
            collector=None,
            seed=0,
        )

        cycles = 3
        for cycle in range(cycles):
            # Day 0: 12:00:00 AM
            action = agent.get_action(2 * cycle * 86400 + 0 - agent.offset)
            np.testing.assert_almost_equal(action, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            # Day 0: 8:59:59 AM
            action = agent.get_action(
                2 * cycle * 86400 + 8 * 3600 + 59 * 60 + 59 - agent.offset
            )
            np.testing.assert_almost_equal(action, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            # Day 0: 9:00:00 AM
            action = agent.get_action(2 * cycle * 86400 + 9 * 3600 - agent.offset)
            np.testing.assert_almost_equal(
                action, [0.199, 0.381, 0.162, 0.0, 0.166, 0.303]
            )

            # Day 0: 9:00:01 AM
            action = agent.get_action(2 * cycle * 86400 + 9 * 3600 + 1 - agent.offset)
            np.testing.assert_almost_equal(
                action, [0.199, 0.381, 0.162, 0.0, 0.166, 0.303]
            )

            # Day 0: 8:59:59 PM
            action = agent.get_action(
                2 * cycle * 86400 + 20 * 3600 + 59 * 60 + 59 - agent.offset
            )
            np.testing.assert_almost_equal(
                action, [0.199, 0.381, 0.162, 0.0, 0.166, 0.303]
            )

            # Day 0: 9:00:00 PM
            action = agent.get_action(2 * cycle * 86400 + 21 * 3600 - agent.offset)
            np.testing.assert_almost_equal(
                action, [0.199, 0.381, 0.162, 0.0, 0.166, 0.303]
            )

            # Day 0: 9:00:01 PM
            action = agent.get_action(2 * cycle * 86400 + 21 * 3600 + 1 - agent.offset)
            np.testing.assert_almost_equal(action, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            # Day 1: 12:00:00 AM
            action = agent.get_action((2 * cycle + 1) * 86400 - agent.offset)
            np.testing.assert_almost_equal(action, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            # Day 1: 8:59:59 AM
            action = agent.get_action(
                (2 * cycle + 1) * 86400 + 8 * 3600 + 59 * 60 + 59 - agent.offset
            )
            np.testing.assert_almost_equal(action, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            # Day 1: 9:00:00 AM
            action = agent.get_action((2 * cycle + 1) * 86400 + 9 * 3600 - agent.offset)
            np.testing.assert_almost_equal(action, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

            # Day 1: 9:00:01 AM
            action = agent.get_action(
                (2 * cycle + 1) * 86400 + 9 * 3600 + 1 - agent.offset
            )
            np.testing.assert_almost_equal(action, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

            # Day 1: 8:59:59 PM
            action = agent.get_action(
                (2 * cycle + 1) * 86400 + 20 * 3600 + 59 * 60 + 59 - agent.offset
            )
            np.testing.assert_almost_equal(action, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

            # Day 1: 9:00:00 PM
            action = agent.get_action(
                (2 * cycle + 1) * 86400 + 21 * 3600 - agent.offset
            )
            np.testing.assert_almost_equal(action, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

            # Day 1: 9:00:01 PM
            action = agent.get_action(
                (2 * cycle + 1) * 86400 + 21 * 3600 + 1 - agent.offset
            )
            np.testing.assert_almost_equal(action, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_get_action_interpolation(self):
        agent = SpreadsheetAgent(
            observations=(1,),
            actions=6,
            params={"filepath": "tests/test_data/interpolation.xlsx"},
            collector=None,
            seed=0,
        )

        cycles = 3
        for cycle in range(cycles):
            # 12:00:00 AM
            action = agent.get_action(cycle * 86400 - agent.offset)
            np.testing.assert_almost_equal(action, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            # 8:59:59 AM
            action = agent.get_action(
                cycle * 86400 + 8 * 3600 + 59 * 60 + 59 - agent.offset
            )
            np.testing.assert_almost_equal(action, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            # 9:00:00 AM
            action = agent.get_action(cycle * 86400 + 9 * 3600 - agent.offset)
            np.testing.assert_almost_equal(action, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

            # 12:00:00 PM
            action = agent.get_action(cycle * 86400 + 12 * 3600 - agent.offset)
            np.testing.assert_almost_equal(
                action, [0.225, 0.275, 0.325, 0.375, 0.425, 0.475]
            )

            # 3:00:00 PM
            action = agent.get_action(cycle * 86400 + 15 * 3600 - agent.offset)
            np.testing.assert_almost_equal(action, [0.35, 0.35, 0.35, 0.35, 0.35, 0.35])

            # 6:00:00 PM
            action = agent.get_action(cycle * 86400 + 18 * 3600 - agent.offset)
            np.testing.assert_almost_equal(
                action, [0.475, 0.425, 0.375, 0.325, 0.275, 0.225]
            )

            # 9:00:00 PM
            action = agent.get_action(cycle * 86400 + 21 * 3600 - agent.offset)
            np.testing.assert_almost_equal(action, [0.6, 0.5, 0.4, 0.3, 0.2, 0.1])

            # 9:00:01 PM
            action = agent.get_action(cycle * 86400 + 21 * 3600 + 1 - agent.offset)
            np.testing.assert_almost_equal(action, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_get_action_wrapping(self):
        agent = SpreadsheetAgent(
            observations=(1,),
            actions=6,
            params={"filepath": "tests/test_data/wrapping.xlsx"},
            collector=None,
            seed=0,
        )

        cycles = 3
        for cycle in range(cycles):
            # 12:00:00 AM
            action = agent.get_action(cycle * 86400 - agent.offset)
            np.testing.assert_almost_equal(
                action, [0.475, 0.425, 0.375, 0.325, 0.275, 0.225]
            )

            # 3:00:00 AM
            action = agent.get_action(cycle * 86400 + 3 * 3600 - agent.offset)
            np.testing.assert_almost_equal(action, [0.35, 0.35, 0.35, 0.35, 0.35, 0.35])

            # 6:00:00 AM
            action = agent.get_action(cycle * 86400 + 6 * 3600 - agent.offset)
            np.testing.assert_almost_equal(
                action, [0.225, 0.275, 0.325, 0.375, 0.425, 0.475]
            )

            # 9:00:00 AM
            action = agent.get_action(cycle * 86400 + 9 * 3600 - agent.offset)
            np.testing.assert_almost_equal(action, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

            # 9:00:01 AM
            action = agent.get_action(cycle * 86400 + 9 * 3600 + 1 - agent.offset)
            np.testing.assert_almost_equal(action, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            # 8:59:59 PM
            action = agent.get_action(
                cycle * 86400 + 20 * 3600 + 59 * 60 + 59 - agent.offset
            )
            np.testing.assert_almost_equal(action, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            # 9:00:00 PM
            action = agent.get_action(cycle * 86400 + 21 * 3600 - agent.offset)
            np.testing.assert_almost_equal(action, [0.6, 0.5, 0.4, 0.3, 0.2, 0.1])

    def test_compatibility_mode(self):
        agent = SpreadsheetAgent(
            observations=(1,),
            actions=6,
            params={
                "filepath": "tests/test_data/z3-0min-100ppfd-Balanced_optima12_12.xlsx",
                "compatibility_mode": True,
            },
            collector=None,
            seed=0,
        )

        cycles = 3

        for cycle in range(cycles):
            # 12:00:00 AM
            action = agent.get_action(cycle * 86400 + 0 - agent.offset)
            np.testing.assert_almost_equal(action, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            # 8:59:59 AM
            action = agent.get_action(
                cycle * 86400 + 8 * 3600 + 59 * 60 + 59 - agent.offset
            )
            np.testing.assert_almost_equal(action, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            # 9:00:00 AM
            action = agent.get_action(cycle * 86400 + 9 * 3600 - agent.offset)
            np.testing.assert_almost_equal(
                action, [0.398, 0.762, 0.324, 0.0, 0.332, 0.606]
            )

            # 9:00:01 AM
            action = agent.get_action(cycle * 86400 + 9 * 3600 + 1 - agent.offset)
            np.testing.assert_almost_equal(
                action, [0.398, 0.762, 0.324, 0.0, 0.332, 0.606]
            )

            # 8:59:59 PM
            action = agent.get_action(
                cycle * 86400 + 20 * 3600 + 59 * 60 + 59 - agent.offset
            )
            np.testing.assert_almost_equal(
                action, [0.398, 0.762, 0.324, 0.0, 0.332, 0.606]
            )

            # 9:00:00 PM
            action = agent.get_action(cycle * 86400 + 21 * 3600 - agent.offset)
            np.testing.assert_almost_equal(
                action, [0.398, 0.762, 0.324, 0.0, 0.332, 0.606]
            )

            # 9:00:01 PM
            action = agent.get_action(cycle * 86400 + 21 * 3600 + 1 - agent.offset)
            np.testing.assert_almost_equal(action, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_checkpointing(self, tmpdir):
        """Test that the agent state can be saved and loaded via checkpointing."""
        tmp_dir = str(tmpdir)

        # Create agent with specific parameters
        n_actions = 6
        filepath = "tests/test_data/z3-0min-100ppfd-Balanced_optima12_12.xlsx"
        compatibility_mode = False
        seed = 123

        # Create directory structure
        checkpoint_dir = os.path.join(tmp_dir, "0")
        os.makedirs(checkpoint_dir, exist_ok=True)
        params_file = os.path.join(checkpoint_dir, "params.json")

        # Set up context mock
        mock_ctx = MagicMock()
        mock_ctx.resolve.side_effect = lambda path: os.path.join(tmp_dir, path)
        mock_ctx.exists.return_value = True
        mock_ctx.ensureExists.side_effect = lambda path, is_file: os.path.join(
            tmp_dir, path
        )

        # Create a mock experiment description
        mock_exp = MagicMock(spec=ExperimentDescription)
        mock_exp.getPermutation.return_value = {
            "filepath": filepath,
            "compatibility_mode": compatibility_mode,
            "seed": seed,
        }
        mock_exp.buildSaveContext.return_value = mock_ctx

        # Initialize a checkpoint with the mock experiment
        chk = Checkpoint(mock_exp, 0, base_path=tmp_dir)

        # Manually write params file that checkpoint expects
        os.makedirs(os.path.dirname(params_file), exist_ok=True)
        with open(params_file, "w") as f:
            import json

            json.dump(
                {
                    "filepath": filepath,
                    "compatibility_mode": compatibility_mode,
                    "seed": seed,
                },
                f,
            )

        # Create the original agent
        original_agent = SpreadsheetAgent(
            observations=(1,),
            actions=n_actions,
            params={"filepath": filepath, "compatibility_mode": compatibility_mode},
            collector=None,
            seed=seed,
        )

        # Store agent state for comparison
        original_offset = original_agent.offset
        original_df_shape = original_agent.df.shape

        # Store the agent in the checkpoint
        chk["a"] = original_agent

        # Save the checkpoint
        chk.save()

        # Create a new checkpoint and load
        new_chk = Checkpoint(mock_exp, 0, base_path=tmp_dir)
        loaded = new_chk.load()

        # Verify checkpoint loaded successfully
        assert loaded, "Failed to load checkpoint"

        # Get the loaded agent
        loaded_agent = new_chk["a"]

        # Verify agent state was properly restored
        assert loaded_agent.params["filepath"] == original_agent.params["filepath"], (
            "Filepath parameter not restored correctly"
        )
        assert (
            loaded_agent.params["compatibility_mode"]
            == original_agent.params["compatibility_mode"]
        ), "Compatibility mode parameter not restored correctly"
        assert loaded_agent.offset == original_offset, "Offset not restored correctly"
        assert loaded_agent.df.shape == original_df_shape, (
            "DataFrame shape not restored correctly"
        )
        # Verify agent behavior is consistent
        for test_time in [0, 3600, 43200, 86400]:
            original_action = original_agent.get_action(test_time)
            loaded_action = loaded_agent.get_action(test_time)
            np.testing.assert_array_equal(
                original_action, loaded_action, f"Actions differ at time {test_time}"
            )

            # Test step method
            reward = 0.0
            obs = np.array([test_time])
            original_action, _ = original_agent.step(reward, obs, {})
            loaded_action, _ = loaded_agent.step(reward, obs, {})

            np.testing.assert_array_equal(
                original_action,
                loaded_action,
                f"Step actions differ at time {test_time}",
            )
