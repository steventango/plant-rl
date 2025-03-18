import numpy as np

from algorithms.SpreadsheetAgent import SpreadsheetAgent


class TestSpreadsheetAgent:
    def test_get_action_single_day_cycle(self):
        agent = SpreadsheetAgent(
            observations=(1,),
            actions=6,
            params={"filepath": "tests/test_data/z3-0min-100ppfd-Balanced_optima12_12.xlsx"},
            collector=None,
            seed=0,
        )

        cycles = 3

        for cycle in range(cycles):
            # 12:00:00 AM
            action = agent.get_action(cycle * 86400 + 0 - agent.offset)
            np.testing.assert_almost_equal(action, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            # 8:59:59 AM
            action = agent.get_action(cycle * 86400 + 8 * 3600 + 59 * 60 + 59 - agent.offset)
            np.testing.assert_almost_equal(action, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            # 9:00:00 AM
            action = agent.get_action(cycle * 86400 + 9 * 3600 - agent.offset)
            np.testing.assert_almost_equal(action, [0.199, 0.381, 0.162, 0.0, 0.166, 0.303])

            # 9:00:01 AM
            action = agent.get_action(cycle * 86400 + 9 * 3600 + 1 - agent.offset)
            np.testing.assert_almost_equal(action, [0.199, 0.381, 0.162, 0.0, 0.166, 0.303])

            # 8:59:59 PM
            action = agent.get_action(cycle * 86400 + 20 * 3600 + 59 * 60 + 59 - agent.offset)
            np.testing.assert_almost_equal(action, [0.199, 0.381, 0.162, 0.0, 0.166, 0.303])

            # 9:00:00 PM
            action = agent.get_action(cycle * 86400 + 21 * 3600 - agent.offset)
            np.testing.assert_almost_equal(action, [0.199, 0.381, 0.162, 0.0, 0.166, 0.303])

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
            action = agent.get_action(2 * cycle * 86400 + 8 * 3600 + 59 * 60 + 59 - agent.offset)
            np.testing.assert_almost_equal(action, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            # Day 0: 9:00:00 AM
            action = agent.get_action(2 * cycle * 86400 + 9 * 3600 - agent.offset)
            np.testing.assert_almost_equal(action, [0.199, 0.381, 0.162, 0.0, 0.166, 0.303])

            # Day 0: 9:00:01 AM
            action = agent.get_action(2 * cycle * 86400 + 9 * 3600 + 1 - agent.offset)
            np.testing.assert_almost_equal(action, [0.199, 0.381, 0.162, 0.0, 0.166, 0.303])

            # Day 0: 8:59:59 PM
            action = agent.get_action(2 * cycle * 86400 + 20 * 3600 + 59 * 60 + 59 - agent.offset)
            np.testing.assert_almost_equal(action, [0.199, 0.381, 0.162, 0.0, 0.166, 0.303])

            # Day 0: 9:00:00 PM
            action = agent.get_action(2 * cycle * 86400 + 21 * 3600 - agent.offset)
            np.testing.assert_almost_equal(action, [0.199, 0.381, 0.162, 0.0, 0.166, 0.303])

            # Day 0: 9:00:01 PM
            action = agent.get_action(2 * cycle * 86400 + 21 * 3600 + 1 - agent.offset)
            np.testing.assert_almost_equal(action, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            # Day 1: 12:00:00 AM
            action = agent.get_action((2 * cycle + 1) * 86400 - agent.offset)
            np.testing.assert_almost_equal(action, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            # Day 1: 8:59:59 AM
            action = agent.get_action((2 * cycle + 1) * 86400 + 8 * 3600 + 59 * 60 + 59 - agent.offset)
            np.testing.assert_almost_equal(action, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            # Day 1: 9:00:00 AM
            action = agent.get_action((2 * cycle + 1) * 86400 + 9 * 3600 - agent.offset)
            np.testing.assert_almost_equal(action, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

            # Day 1: 9:00:01 AM
            action = agent.get_action((2 * cycle + 1) * 86400 + 9 * 3600 + 1 - agent.offset)
            np.testing.assert_almost_equal(action, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

            # Day 1: 8:59:59 PM
            action = agent.get_action((2 * cycle + 1) * 86400 + 20 * 3600 + 59 * 60 + 59 - agent.offset)
            np.testing.assert_almost_equal(action, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

            # Day 1: 9:00:00 PM
            action = agent.get_action((2 * cycle + 1) * 86400 + 21 * 3600 - agent.offset)
            np.testing.assert_almost_equal(action, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

            # Day 1: 9:00:01 PM
            action = agent.get_action((2 * cycle + 1) * 86400 + 21 * 3600 + 1 - agent.offset)
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
            action = agent.get_action(cycle * 86400 + 8 * 3600 + 59 * 60 + 59 - agent.offset)
            np.testing.assert_almost_equal(action, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            # 9:00:00 AM
            action = agent.get_action(cycle * 86400 + 9 * 3600 - agent.offset)
            np.testing.assert_almost_equal(action, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

            # 12:00:00 PM
            action = agent.get_action(cycle * 86400 + 12 * 3600 - agent.offset)
            np.testing.assert_almost_equal(action, [0.225, 0.275, 0.325, 0.375, 0.425, 0.475])

            # 3:00:00 PM
            action = agent.get_action(cycle * 86400 + 15 * 3600 - agent.offset)
            np.testing.assert_almost_equal(action, [0.35, 0.35, 0.35, 0.35, 0.35, 0.35])

            # 6:00:00 PM
            action = agent.get_action(cycle * 86400 + 18 * 3600 - agent.offset)
            np.testing.assert_almost_equal(action, [0.475, 0.425, 0.375, 0.325, 0.275, 0.225])

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
            np.testing.assert_almost_equal(action, [0.475, 0.425, 0.375, 0.325, 0.275, 0.225])

            # 3:00:00 AM
            action = agent.get_action(cycle * 86400 + 3 * 3600 - agent.offset)
            np.testing.assert_almost_equal(action, [0.35, 0.35, 0.35, 0.35, 0.35, 0.35])

            # 6:00:00 AM
            action = agent.get_action(cycle * 86400 + 6 * 3600 - agent.offset)
            np.testing.assert_almost_equal(action, [0.225, 0.275, 0.325, 0.375, 0.425, 0.475])

            # 9:00:00 AM
            action = agent.get_action(cycle * 86400 + 9 * 3600 - agent.offset)
            np.testing.assert_almost_equal(action, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

            # 9:00:01 AM
            action = agent.get_action(cycle * 86400 + 9 * 3600 + 1 - agent.offset)
            np.testing.assert_almost_equal(action, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            # 8:59:59 PM
            action = agent.get_action(cycle * 86400 + 20 * 3600 + 59 * 60 + 59 - agent.offset)
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
            action = agent.get_action(cycle * 86400 + 8 * 3600 + 59 * 60 + 59 - agent.offset)
            np.testing.assert_almost_equal(action, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            # 9:00:00 AM
            action = agent.get_action(cycle * 86400 + 9 * 3600 - agent.offset)
            np.testing.assert_almost_equal(action, [0.398, 0.762, 0.324, 0.0, 0.332, 0.606])

            # 9:00:01 AM
            action = agent.get_action(cycle * 86400 + 9 * 3600 + 1 - agent.offset)
            np.testing.assert_almost_equal(action, [0.398, 0.762, 0.324, 0.0, 0.332, 0.606])

            # 8:59:59 PM
            action = agent.get_action(cycle * 86400 + 20 * 3600 + 59 * 60 + 59 - agent.offset)
            np.testing.assert_almost_equal(action, [0.398, 0.762, 0.324, 0.0, 0.332, 0.606])

            # 9:00:00 PM
            action = agent.get_action(cycle * 86400 + 21 * 3600 - agent.offset)
            np.testing.assert_almost_equal(action, [0.398, 0.762, 0.324, 0.0, 0.332, 0.606])

            # 9:00:01 PM
            action = agent.get_action(cycle * 86400 + 21 * 3600 + 1 - agent.offset)
            np.testing.assert_almost_equal(action, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
