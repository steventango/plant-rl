import numpy as np

from algorithms.SpreadsheetAgent import SpreadsheetAgent


class TestSpreadsheetAgent:
    def test_get_action_single_day_cycle(self):
        agent = SpreadsheetAgent(
            observations=(1,),
            actions=2,
            params={"filepath": "tests/test_data/z3-0min-100ppfd-Balanced_optima12_12.xlsx"},
            collector=None,
            seed=0,
        )

        # Day 0: 12:00:00 AM
        action = agent.get_action(0)
        np.testing.assert_almost_equal(action, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Day 0: 8:59:59 AM
        action = agent.get_action(8 * 3600 + 59 * 60 + 59)
        np.testing.assert_almost_equal(action, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Day 0: 9:00:00 AM
        action = agent.get_action(9 * 3600)
        np.testing.assert_almost_equal(action, [0.199, 0.381, 0.162, 0.   , 0.166, 0.303])

        # Day 0: 9:00:01 AM
        action = agent.get_action(9 * 3600 + 1)
        np.testing.assert_almost_equal(action, [0.199, 0.381, 0.162, 0.   , 0.166, 0.303])

        # Day 0: 8:59:59 PM
        action = agent.get_action(20 * 3600 + 59 * 60 + 59)
        np.testing.assert_almost_equal(action, [0.199, 0.381, 0.162, 0.   , 0.166, 0.303])

        # Day 0: 9:00:00 PM
        action = agent.get_action(21 * 3600)
        np.testing.assert_almost_equal(action, [0.199, 0.381, 0.162, 0.   , 0.166, 0.303])

        # Day 0: 9:00:01 PM
        action = agent.get_action(21 * 3600 + 1)
        np.testing.assert_almost_equal(action, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Day 1: 12:00:00 AM
        action = agent.get_action(86400)
        np.testing.assert_almost_equal(action, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Day 1: 8:59:59 AM
        action = agent.get_action(86400 + 8 * 3600 + 59 * 60 + 59)
        np.testing.assert_almost_equal(action, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Day 1: 9:00:00 AM
        action = agent.get_action(86400 + 9 * 3600)
        np.testing.assert_almost_equal(action, [0.199, 0.381, 0.162, 0.   , 0.166, 0.303])

        # Day 1: 9:00:01 AM
        action = agent.get_action(86400 + 9 * 3600 + 1)
        np.testing.assert_almost_equal(action, [0.199, 0.381, 0.162, 0.   , 0.166, 0.303])

        # Day 1: 8:59:59 PM
        action = agent.get_action(86400 + 20 * 3600 + 59 * 60 + 59)
        np.testing.assert_almost_equal(action, [0.199, 0.381, 0.162, 0.   , 0.166, 0.303])

        # Day 1: 9:00:00 PM
        action = agent.get_action(86400 + 21 * 3600)
        np.testing.assert_almost_equal(action, [0.199, 0.381, 0.162, 0.   , 0.166, 0.303])

        # Day 1: 9:00:01 PM
        action = agent.get_action(86400 + 21 * 3600 + 1)
        np.testing.assert_almost_equal(action, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
