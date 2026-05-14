from datetime import datetime

import numpy as np

from agents.Incubator import Incubator
from utils.constants import BALANCED_ACTION_100


def make_agent(incubation_ppfd=100.0):
    return Incubator(
        observations=(1,),
        actions=6,
        params={"agent": {"incubation_ppfd": incubation_ppfd}},
    )


def dt(hour, minute=0):
    return datetime(2025, 1, 1, hour, minute)


class TestIncubator:
    async def test_night_action_is_zero(self):
        agent = make_agent()
        for hour in [0, 1, 8, 21, 22, 23]:
            action, _ = await agent.start(dt(hour))
            assert np.all(action == 0), f"Expected zeros at hour {hour}"

            action, _ = await agent.step(0.0, dt(hour), {})
            assert np.all(action == 0), f"Expected zeros at hour {hour}"

    async def test_morning_action_is_half(self):
        agent = make_agent()
        expected = 0.5 * BALANCED_ACTION_100
        action, _ = await agent.start(dt(9, 0))
        np.testing.assert_array_almost_equal(action, expected)

    async def test_morning_only_at_exact_minute(self):
        agent = make_agent()
        morning_action = 0.5 * BALANCED_ACTION_100
        for minute in [1, 30, 59]:
            action, _ = await agent.start(dt(9, minute))
            assert not np.allclose(action, morning_action), (
                f"Expected daytime action at 9:{minute:02d}, not morning action"
            )

    async def test_daytime_action(self):
        agent = make_agent(incubation_ppfd=80.0)
        expected = 0.80 * BALANCED_ACTION_100
        for hour in [9, 10, 15, 20]:
            obs = dt(9, 30) if hour == 9 else dt(hour)
            action, _ = await agent.start(obs)
            np.testing.assert_array_almost_equal(
                action, expected, err_msg=f"hour={hour}"
            )

    async def test_default_incubation_ppfd(self):
        agent = Incubator(
            observations=(1,),
            actions=6,
            params={},
        )
        assert agent.incubation_ppfd == 100.0

    async def test_start_and_step_return_same_action(self):
        agent = make_agent()
        obs = dt(12)
        start_action, _ = await agent.start(obs)
        step_action, _ = await agent.step(1.0, obs, {})
        np.testing.assert_array_equal(start_action, step_action)

    async def test_end_returns_empty_dict(self):
        agent = make_agent()
        result = await agent.end(1.0, {})
        assert result == {}
