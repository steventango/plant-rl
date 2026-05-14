from datetime import datetime, timedelta

import numpy as np
import pytest

from agents.ScheduleAgent import ScheduleAgent
from utils.constants import BALANCED_ACTION_100


def make_agent(action_days=None, action_inputs=None):
    params = {}
    if action_days is not None:
        params["action_days"] = action_days
    if action_inputs is not None:
        params["action_inputs"] = action_inputs
    return ScheduleAgent(observations=(1,), actions=6, params=params, seed=0)


def dt(date, hour, minute=0):
    return datetime(date.year, date.month, date.day, hour, minute)


START = datetime(2025, 3, 1).date()


class TestScheduleAgent:
    async def test_night_action_is_zero(self):
        agent = make_agent(action_days=[1], action_inputs=[100.0])
        await agent.start(dt(START, 10))
        for hour in [0, 1, 8, 21, 22, 23]:
            action, _ = await agent.step(0.0, dt(START, hour), {})
            assert np.all(action == 0), f"Expected zeros at hour {hour}"

    async def test_morning_action_is_half_balanced(self):
        agent = make_agent(action_days=[1], action_inputs=[100.0])
        await agent.start(dt(START, 10))
        action, _ = await agent.step(0.0, dt(START, 9, 0), {})
        np.testing.assert_array_almost_equal(action, 0.5 * BALANCED_ACTION_100)

    async def test_morning_only_at_exact_minute(self):
        agent = make_agent(action_days=[1], action_inputs=[100.0])
        await agent.start(dt(START, 10))
        morning_action = 0.5 * BALANCED_ACTION_100
        for minute in [1, 30, 59]:
            action, _ = await agent.step(0.0, dt(START, 9, minute), {})
            assert not np.allclose(action, morning_action), (
                f"Expected daytime action at 9:{minute:02d}, not morning action"
            )

    async def test_single_step_constant_ppfd(self):
        agent = make_agent(action_days=[1], action_inputs=[150.0])
        await agent.start(dt(START, 10))
        for hour in [10, 12, 15, 20]:
            action, _ = await agent.step(0.0, dt(START, hour), {})
            assert action == pytest.approx(150.0), f"hour={hour}"

    async def test_multi_step_schedule(self):
        # Days 1-4: 100, days 5-9: 150, days 10+: 200
        agent = make_agent(action_days=[1, 5, 10], action_inputs=[100.0, 150.0, 200.0])
        await agent.start(dt(START, 10))

        cases = [
            (0, 100.0),   # day 1
            (3, 100.0),   # day 4
            (4, 150.0),   # day 5
            (8, 150.0),   # day 9
            (9, 200.0),   # day 10
            (14, 200.0),  # day 15
        ]
        for day_offset, expected_ppfd in cases:
            obs = dt(START + timedelta(days=day_offset), 12)
            action, _ = await agent.step(0.0, obs, {})
            assert action == pytest.approx(expected_ppfd), (
                f"day {day_offset}: expected {expected_ppfd}, got {action}"
            )

    async def test_schedule_transition_on_exact_day(self):
        agent = make_agent(action_days=[1, 3], action_inputs=[80.0, 160.0])
        await agent.start(dt(START, 10))

        action_before, _ = await agent.step(0.0, dt(START + timedelta(days=1), 12), {})
        assert action_before == pytest.approx(80.0)

        action_on, _ = await agent.step(0.0, dt(START + timedelta(days=2), 12), {})
        assert action_on == pytest.approx(160.0)

    async def test_start_sets_start_date(self):
        agent = make_agent(action_days=[0], action_inputs=[100.0])
        assert agent.start_date is None
        await agent.start(dt(START, 10))
        assert agent.start_date == START

    async def test_default_params(self):
        agent = ScheduleAgent(observations=(1,), actions=6, params={}, seed=0)
        assert agent.action_days == [1]
        assert agent.action_inputs == [0.0]

    async def test_end_returns_empty_dict(self):
        agent = make_agent(action_days=[1], action_inputs=[100.0])
        result = await agent.end(1.0, {})
        assert result == {}

    async def test_start_and_step_return_same_action(self):
        agent = make_agent(action_days=[1], action_inputs=[120.0])
        obs = dt(START, 12)
        start_action, _ = await agent.start(obs)
        step_action, _ = await agent.step(1.0, obs, {})
        np.testing.assert_array_equal(start_action, step_action)
