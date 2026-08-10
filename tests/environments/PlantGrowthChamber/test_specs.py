from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pickle

from environments.PlantGrowthChamber.factory import (
    ComposedPlantGrowthChamber,
    create_plant_growth_chamber,
)
from environments.PlantGrowthChamber.specs import (
    ACTION_SPECS,
    ColorTriangleAction,
    DayAreaTraceObservation,
    DiscreteAction,
    IntensityAction,
    OneHotTimeObservation,
    RawObservation,
    create_observation_spec,
)
from utils.constants import BALANCED_ACTION_100, BALANCED_ACTION_105, DIM_ACTION


def test_intensity_action_scales_scalar():
    spec = IntensityAction()
    result = spec.decode(1.0)
    np.testing.assert_array_equal(result, BALANCED_ACTION_100)


def test_discrete_action_maps_indices():
    spec = DiscreteAction()
    np.testing.assert_array_equal(spec.decode(0), DIM_ACTION)
    np.testing.assert_array_equal(spec.decode(1), BALANCED_ACTION_105)


def test_color_triangle_decodes_simplex():
    spec = ColorTriangleAction()
    action = np.array([0.0, 1.0, 0.0])
    result = spec.decode(action)
    np.testing.assert_array_equal(result, BALANCED_ACTION_105)


def test_color_triangle_trace_dim():
    assert ACTION_SPECS["color_triangle"].trace_dim == 3
    assert create_observation_spec(
        "day_area_trace", ACTION_SPECS["color_triangle"], {}
    ).shape == (5,)


def test_intensity_trace_action_decodes_scalar():
    spec = IntensityAction()
    result = spec.trace_action(0.8)
    np.testing.assert_allclose(result, BALANCED_ACTION_100 * 0.8)


def test_color_triangle_trace_action_keeps_coefficients():
    spec = ColorTriangleAction()
    action = np.array([0.2, 0.5, 0.3])
    np.testing.assert_array_equal(spec.trace_action(action), action)


def test_color_triangle_trace_action_projects_six_channel():
    spec = ColorTriangleAction()
    result = spec.trace_action(np.zeros(6))
    assert result.shape == (3,)


def test_update_action_trace_decodes_before_uema_update():
    backend = MagicMock()
    backend.get_local_time.return_value.date.return_value = __import__("datetime").date(
        2025, 1, 1
    )
    obs_spec = DayAreaTraceObservation(trace_dim=6)
    action_spec = IntensityAction()
    env = ComposedPlantGrowthChamber(backend, obs_spec, action_spec, {})

    env.update_action_trace(0.8)
    assert obs_spec.action_uema is not None
    trace = np.asarray(obs_spec.action_uema.compute()).reshape(-1)
    np.testing.assert_allclose(trace, BALANCED_ACTION_100 * 0.8)


def test_update_action_trace_handles_six_channel_night_action_for_color_triangle():
    backend = MagicMock()
    backend.get_local_time.return_value.date.return_value = __import__("datetime").date(
        2025, 1, 1
    )
    obs_spec = DayAreaTraceObservation(trace_dim=3)
    action_spec = ColorTriangleAction()
    env = ComposedPlantGrowthChamber(backend, obs_spec, action_spec, {})

    env.update_action_trace(np.zeros(6))
    assert obs_spec.action_uema is not None
    assert np.asarray(obs_spec.action_uema.compute()).reshape(-1).shape == (3,)


def test_composed_plant_growth_chamber_survives_checkpoint_roundtrip():
    env = create_plant_growth_chamber(
        backend="mock",
        observation="scalar",
        action="ppfd6",
        zone="alliance-zone08",
    )
    loaded = pickle.loads(pickle.dumps(env))
    assert isinstance(loaded, ComposedPlantGrowthChamber)
    assert type(loaded._backend) is type(env._backend)


def test_one_hot_time_observation_shape():
    import asyncio
    from datetime import datetime
    from zoneinfo import ZoneInfo

    async def run():
        spec = OneHotTimeObservation()
        raw = RawObservation(
            local_time=datetime(2025, 6, 10, 12, 0, tzinfo=ZoneInfo("Etc/UTC")),
            df=pd.DataFrame(),
            dli=0.0,
        )
        return await spec.encode(raw)

    obs = asyncio.run(run())
    assert obs.shape == (13,)
    assert obs.sum() == 1.0


def test_iqm_log_clean_area_is_robust_to_dead_nan_and_cv_failures():
    import pytest

    from environments.PlantGrowthChamber.specs.observations import iqm_log_clean_area

    # Ten healthy plants, plus a dead plant (0), a CV failure (NaN), and a
    # spurious huge detection (1e6) that a plain mean would be dragged by.
    areas = [
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0,
        17.0,
        18.0,
        19.0,
        0.0,
        np.nan,
        1e6,
    ]
    df = pd.DataFrame({"pot_id": range(len(areas)), "clean_area": areas})

    value = iqm_log_clean_area(df)

    # Reproduce the spec: drop non-finite/non-positive, IQM over log-areas.
    cleaned = np.array([a for a in areas if np.isfinite(a) and a > 0.0])
    log_a = np.log(cleaned)
    lo, hi = np.quantile(log_a, [0.25, 0.75])
    expected = log_a[(log_a >= lo) & (log_a <= hi)].mean()
    assert value == pytest.approx(expected, abs=1e-4)
    # Robust: the 1e6 outlier is trimmed, so the estimate stays in the healthy band.
    assert value < np.log(50)


def test_iqm_log_clean_area_empty_uses_in_distribution_fallback():
    import pytest

    from environments.PlantGrowthChamber.specs.observations import (
        FALLBACK_LOG_AREA,
        iqm_log_clean_area,
    )

    # Never the old log(1e-6) = -13.8 floor: that is far outside the range the
    # deployed policies were trained on, so they would extrapolate.
    assert iqm_log_clean_area(pd.DataFrame()) == pytest.approx(FALLBACK_LOG_AREA)
    all_dead = pd.DataFrame({"pot_id": [0, 1], "clean_area": [0.0, np.nan]})
    assert iqm_log_clean_area(all_dead) == pytest.approx(FALLBACK_LOG_AREA)
    assert iqm_log_clean_area(pd.DataFrame()) > float(np.log(1e-6))


def test_log_area_observation_uses_iqm():
    import asyncio

    from environments.PlantGrowthChamber.specs.observations import (
        LogAreaObservation,
        iqm_log_clean_area,
    )

    areas = [10.0, 12.0, 14.0, 16.0, 18.0, 0.0, np.nan, 1e6]
    df = pd.DataFrame({"pot_id": range(len(areas)), "clean_area": areas})
    raw = RawObservation(local_time=None, df=df, dli=0.0)  # type: ignore[arg-type]
    obs = asyncio.run(LogAreaObservation().encode(raw))
    assert obs.shape == (1,)
    assert float(obs[0]) == iqm_log_clean_area(df)


def test_iqm_log_clean_area_fallback_is_in_distribution():
    """No-plant fallback must be a plausible day-0 area, not a -13.8 floor."""
    import pytest

    from environments.PlantGrowthChamber.specs.observations import (
        FALLBACK_LOG_AREA,
        iqm_log_clean_area,
    )

    # plant-data/visu-v28 day-0 interquantile mean log clean-area.
    assert FALLBACK_LOG_AREA == pytest.approx(-1.0744097232818604)
    # Well inside the log-area range the deployed policies were trained on.
    assert -3.10 < FALLBACK_LOG_AREA < 2.41

    assert iqm_log_clean_area(pd.DataFrame()) == pytest.approx(FALLBACK_LOG_AREA)
    no_plants = pd.DataFrame({"pot_id": [0, 1], "clean_area": [0.0, np.nan]})
    assert iqm_log_clean_area(no_plants) == pytest.approx(FALLBACK_LOG_AREA)
    # A caller may still override it explicitly.
    assert iqm_log_clean_area(pd.DataFrame(), fallback=-2.0) == pytest.approx(-2.0)
