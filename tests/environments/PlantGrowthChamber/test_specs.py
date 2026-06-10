from unittest.mock import MagicMock

import numpy as np

from environments.PlantGrowthChamber.factory import ComposedPlantGrowthChamber
from environments.PlantGrowthChamber.specs import (
    ACTION_SPECS,
    ColorTriangleAction,
    DayAreaTraceObservation,
    DiscreteAction,
    IntensityAction,
    OneHotTimeObservation,
    create_observation_spec,
)
from utils.constants import BALANCED_ACTION_105, DIM_ACTION


def test_intensity_action_scales_scalar():
    spec = IntensityAction()
    result = spec.decode(1.0, backend=None)
    np.testing.assert_array_equal(result, BALANCED_ACTION_105)


def test_discrete_action_maps_indices():
    spec = DiscreteAction()
    np.testing.assert_array_equal(spec.decode(0, None), DIM_ACTION)
    np.testing.assert_array_equal(spec.decode(1, None), BALANCED_ACTION_105)


def test_color_triangle_decodes_simplex():
    spec = ColorTriangleAction()
    action = np.array([0.0, 1.0, 0.0])
    result = spec.decode(action, None)
    np.testing.assert_array_equal(result, BALANCED_ACTION_105)


def test_color_triangle_trace_dim():
    assert ACTION_SPECS["color_triangle"].trace_dim == 3
    assert create_observation_spec(
        "day_area_trace", ACTION_SPECS["color_triangle"], {}
    ).shape == (5,)


def test_intensity_trace_action_decodes_scalar():
    spec = IntensityAction()
    result = spec.trace_action(0.8, backend=None)
    np.testing.assert_allclose(result, BALANCED_ACTION_105 * 0.8)


def test_color_triangle_trace_action_keeps_coefficients():
    spec = ColorTriangleAction()
    action = np.array([0.2, 0.5, 0.3])
    np.testing.assert_array_equal(spec.trace_action(action, None), action)


def test_color_triangle_trace_action_projects_six_channel():
    spec = ColorTriangleAction()
    result = spec.trace_action(np.zeros(6), None)
    assert result.shape == (3,)


def test_update_action_trace_decodes_before_uema_update():
    backend = MagicMock()
    backend.get_local_time.return_value.date.return_value = __import__(
        "datetime"
    ).date(2025, 1, 1)
    obs_spec = DayAreaTraceObservation(trace_dim=6)
    action_spec = IntensityAction()
    env = ComposedPlantGrowthChamber(backend, obs_spec, action_spec, {})

    env.update_action_trace(0.8)
    assert obs_spec.action_uema is not None
    trace = np.asarray(obs_spec.action_uema.compute()).reshape(-1)
    np.testing.assert_allclose(trace, BALANCED_ACTION_105 * 0.8)


def test_update_action_trace_handles_six_channel_night_action_for_color_triangle():
    backend = MagicMock()
    backend.get_local_time.return_value.date.return_value = __import__(
        "datetime"
    ).date(2025, 1, 1)
    obs_spec = DayAreaTraceObservation(trace_dim=3)
    action_spec = ColorTriangleAction()
    env = ComposedPlantGrowthChamber(backend, obs_spec, action_spec, {})

    env.update_action_trace(np.zeros(6))
    assert obs_spec.action_uema is not None
    assert np.asarray(obs_spec.action_uema.compute()).reshape(-1).shape == (3,)


def test_one_hot_time_observation_shape():
    import asyncio
    from datetime import datetime
    from zoneinfo import ZoneInfo

    class FakeBackend:
        tz = ZoneInfo("Etc/UTC")

    async def run():
        spec = OneHotTimeObservation()
        raw = (datetime(2025, 6, 10, 12, 0, tzinfo=ZoneInfo("Etc/UTC")), None, None)
        return await spec.encode(raw, FakeBackend())

    obs = asyncio.run(run())
    assert obs.shape == (13,)
    assert obs.sum() == 1.0
