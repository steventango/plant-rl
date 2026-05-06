import json
from unittest.mock import MagicMock

import numpy as np

from algorithms.SequenceAgent import SequenceAgent

RED = [1, 0, 0]
BLUE = [0, 0, 1]

RED_BLUE_ACTIONS = json.dumps([RED] * 7 + [BLUE] * 7)
BLUE_RED_ACTIONS = json.dumps([BLUE] * 7 + [RED] * 7)


def make_agent(actions_json: str) -> SequenceAgent:
    return SequenceAgent(
        observations=(1,),
        actions=3,
        params={"actions": actions_json},
        collector=MagicMock(),
        seed=42,
    )


class TestSequenceAgentE17P1:
    """Validate that SequenceAgent produces the correct per-day actions for E17/P1."""

    def _collect_actions(self, agent: SequenceAgent, n: int = 14) -> list[np.ndarray]:
        """Simulate n daily 9:30 AM polls: start() + (n - 1) × step()."""
        obs = np.zeros(1)
        extra: dict = {}
        actions = []
        action, _ = agent.start(obs, extra)
        actions.append(action)
        for _ in range(n - 1):
            action, _ = agent.step(0.0, obs, extra)
            actions.append(action)
        return actions

    def test_red_blue_week1_is_red(self):
        agent = make_agent(RED_BLUE_ACTIONS)
        actions = self._collect_actions(agent)
        for i, a in enumerate(actions[:7]):
            assert np.array_equal(a, RED), f"Day {i + 1}: expected Red, got {a}"

    def test_red_blue_week2_is_blue(self):
        agent = make_agent(RED_BLUE_ACTIONS)
        actions = self._collect_actions(agent)
        for i, a in enumerate(actions[7:]):
            assert np.array_equal(a, BLUE), f"Day {i + 8}: expected Blue, got {a}"

    def test_blue_red_week1_is_blue(self):
        agent = make_agent(BLUE_RED_ACTIONS)
        actions = self._collect_actions(agent)
        for i, a in enumerate(actions[:7]):
            assert np.array_equal(a, BLUE), f"Day {i + 1}: expected Blue, got {a}"

    def test_blue_red_week2_is_red(self):
        agent = make_agent(BLUE_RED_ACTIONS)
        actions = self._collect_actions(agent)
        for i, a in enumerate(actions[7:]):
            assert np.array_equal(a, RED), f"Day {i + 8}: expected Red, got {a}"

    def test_sequence_length(self):
        agent = make_agent(RED_BLUE_ACTIONS)
        actions = self._collect_actions(agent)
        assert len(actions) == 14

    def test_no_index_error_at_end(self):
        """Agent must not raise IndexError after exhausting the 14-action sequence."""
        agent = make_agent(RED_BLUE_ACTIONS)
        actions = self._collect_actions(agent, n=15)
        # step beyond the sequence should repeat the last action (Blue)
        last_action = actions[-1]
        assert np.array_equal(actions[-1], BLUE), (
            f"Expected last action Blue, got {last_action}"
        )


class TestSequenceAgentCheckpointing:
    def test_steps_survives_checkpoint(self, tmpdir, setup_checkpoint_test):
        """steps counter must be restored so the agent continues mid-sequence."""
        params = {"actions": RED_BLUE_ACTIONS, "seed": 42}

        def advance_7_days(agent):
            obs = np.zeros(1)
            agent.start(obs, {})
            for _ in range(6):
                agent.step(0.0, obs, {})

        original, loaded = setup_checkpoint_test(
            tmpdir,
            params,
            SequenceAgent,
            actions=3,
            init_func=advance_7_days,
        )

        assert loaded.steps == original.steps, (
            f"steps not restored: original={original.steps}, loaded={loaded.steps}"
        )
        # Next poll should be day 8 — first day of week 2 (Blue)
        action, _ = loaded.step(0.0, np.zeros(1), {})
        assert np.array_equal(action, BLUE), (
            f"Expected Blue on day 8 after restore, got {action}"
        )
