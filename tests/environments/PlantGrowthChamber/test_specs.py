import numpy as np

from environments.PlantGrowthChamber.specs import (
    ACTION_SPECS,
    ColorTriangleAction,
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
