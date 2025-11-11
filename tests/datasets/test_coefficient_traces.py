import polars as pl

from src.datasets.config import BLUE, RED, WHITE
from src.datasets.transforms import transform_action, transform_action_traces


def test_coefficient_traces():
    """Test that coefficient traces are computed correctly with exponential weighted moving average."""

    # Create sample data with a plant that transitions between different light configurations
    times = list(range(10))
    plant_ids = ["plant_1"] * 10

    # Create actions that transition from RED -> WHITE -> BLUE
    actions = []
    for i in range(10):
        if i < 3:
            # Pure RED
            actions.append(RED.tolist())
        elif i < 6:
            # Pure WHITE
            actions.append(WHITE.tolist())
        else:
            # Pure BLUE
            actions.append(BLUE.tolist())

    # Create DataFrame
    data = {
        "time": times,
        "plant_id": plant_ids,
        "experiment": ["exp_1"] * 10,  # Dummy values
        "zone": ["zone_1"] * 10,  # Dummy values
        "clean_area": [100.0] * 10,  # Dummy values
    }

    # Add individual action components
    for i in range(6):
        data[f"action.{i}"] = [action[i] for action in actions]

    df = pl.DataFrame(data)

    print("Original DataFrame:")
    print(df)
    print("\n" + "=" * 80 + "\n")

    # Apply transformations
    df = transform_action(df)
    df = transform_action_traces(df)

    print("After transform_action and transform_action_traces:")
    print(
        df.select(
            [
                "time",
                "plant_id",
                "red_coef",
                "white_coef",
                "blue_coef",
                "red_coef_trace_0.5",
                "white_coef_trace_0.5",
                "blue_coef_trace_0.5",
            ]
        )
    )
    print("\n" + "=" * 80 + "\n")

    # Verify some properties
    # At time 0-1: should be pure RED (coefficients ~ [1, 0, 0])
    # Note: action is shifted backwards, so the last time point gets NaN
    red_phase = df.filter((pl.col("time") >= 0) & (pl.col("time") < 2))
    print("RED phase (time 0-1):")
    print(
        red_phase.select(
            [
                "time",
                "red_coef",
                "white_coef",
                "blue_coef",
            ]
        )
    )

    red_mean = red_phase["red_coef"].drop_nulls().mean()
    white_mean = red_phase["white_coef"].drop_nulls().mean()
    blue_mean = red_phase["blue_coef"].drop_nulls().mean()

    assert isinstance(red_mean, float)
    assert isinstance(white_mean, float)
    assert isinstance(blue_mean, float)

    assert red_mean > 0.99, (
        f"RED coefficient should be ~1.0 in RED phase, got {red_mean}"
    )
    assert white_mean < 0.01, (
        f"WHITE coefficient should be ~0.0 in RED phase, got {white_mean}"
    )
    assert blue_mean < 0.01, (
        f"BLUE coefficient should be ~0.0 in RED phase, got {blue_mean}"
    )
    print("✓ RED phase coefficients correct\n")

    # At time 2-4: should be pure WHITE (coefficients ~ [0, 1, 0])
    white_phase = df.filter((pl.col("time") >= 2) & (pl.col("time") < 5))
    print("WHITE phase (time 2-4):")
    print(
        white_phase.select(
            [
                "time",
                "red_coef",
                "white_coef",
                "blue_coef",
            ]
        )
    )

    red_mean = white_phase["red_coef"].drop_nulls().mean()
    white_mean = white_phase["white_coef"].drop_nulls().mean()
    blue_mean = white_phase["blue_coef"].drop_nulls().mean()

    assert isinstance(red_mean, float)
    assert isinstance(white_mean, float)
    assert isinstance(blue_mean, float)

    assert red_mean < 0.01, (
        f"RED coefficient should be ~0.0 in WHITE phase, got {red_mean}"
    )
    assert white_mean > 0.99, (
        f"WHITE coefficient should be ~1.0 in WHITE phase, got {white_mean}"
    )
    assert blue_mean < 0.01, (
        f"BLUE coefficient should be ~0.0 in WHITE phase, got {blue_mean}"
    )
    print("✓ WHITE phase coefficients correct\n")

    # At time 5-8: should be pure BLUE (coefficients ~ [0, 0, 1])
    blue_phase = df.filter((pl.col("time") >= 5) & (pl.col("time") < 9))
    print("BLUE phase (time 5-8):")
    print(
        blue_phase.select(
            [
                "time",
                "red_coef",
                "white_coef",
                "blue_coef",
            ]
        )
    )

    red_mean = blue_phase["red_coef"].drop_nulls().mean()
    white_mean = blue_phase["white_coef"].drop_nulls().mean()
    blue_mean = blue_phase["blue_coef"].drop_nulls().mean()

    assert isinstance(red_mean, float)
    assert isinstance(white_mean, float)
    assert isinstance(blue_mean, float)

    assert red_mean < 0.2, (
        f"RED coefficient should be ~0.0 in BLUE phase, got {red_mean}"
    )
    assert white_mean < 0.5, (
        f"WHITE coefficient should be ~0.0 in BLUE phase, got {white_mean}"
    )
    assert blue_mean > 0.99, (
        f"BLUE coefficient should be ~1.0 in BLUE phase, got {blue_mean}"
    )
    print("✓ BLUE phase coefficients correct\n")

    # Check that traces exist and are different from raw coefficients
    # (traces should smooth out transitions)
    print("Trace analysis at transition points:")
    transition_point = df.filter(pl.col("time") == 3).select(
        [
            "time",
            "red_coef",
            "white_coef",
            "blue_coef",
            "red_coef_trace_0.5",
            "white_coef_trace_0.5",
            "blue_coef_trace_0.5",
        ]
    )
    print(transition_point)

    # At time 3 (first WHITE), the trace should still have some RED influence
    row = df.filter(pl.col("time") == 3).row(0, named=True)
    red_trace = row["red_coef_trace_0.5"]
    white_trace = row["white_coef_trace_0.5"]

    print("\nAt time 3 (transition to WHITE):")
    print(f"  Raw red_coef: {row['red_coef']:.4f}")
    print(f"  Trace red_coef_trace_0.5: {red_trace:.4f}")
    print(f"  Raw white_coef: {row['white_coef']:.4f}")
    print(f"  Trace white_coef_trace_0.5: {white_trace:.4f}")

    # The trace should show influence from previous RED actions
    assert red_trace > 0.1, "RED trace should still have influence at transition point"
    print("✓ Traces show smoothing effect across transitions\n")
