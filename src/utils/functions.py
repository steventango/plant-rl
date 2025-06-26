from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax._src.typing import Array
from jax.nn import relu


class PiecewiseLinear:
    def __init__(self, x_values, y_values):
        """
        Initialize with arrays of x and y coordinates
        Interpolate between consecutive points and return a callable PiecewiseLinear object.
        """
        if len(x_values) != len(y_values):
            raise ValueError("x_values and y_values must have same length")

        self.segments = []  # Will store (x_min, x_max, slope, intercept)

        # Create segments from consecutive points
        for i in range(len(x_values) - 1):
            x_min, x_max = x_values[i], x_values[i + 1]
            y_min, y_max = y_values[i], y_values[i + 1]

            # Calculate slope and intercept for this segment
            slope = (y_max - y_min) / (x_max - x_min)
            intercept = y_min - slope * x_min
            self.segments.append((x_min, x_max, slope, intercept))

    def __call__(self, x):
        # Find the appropriate segment and evaluate
        for x_min, x_max, slope, intercept in self.segments:
            if x_min <= x <= x_max:
                return slope * x + intercept
        raise ValueError(f"x={x} is not within any defined segment")

    def copy(self):
        """
        Create a deep copy of the piecewise linear function
        Returns:
            A new PiecewiseLinear object with the same segments
        """
        new_obj = PiecewiseLinear([0], [0])  # Create dummy object
        new_obj.segments = list(self.segments)  # Make a copy of segments list
        return new_obj

    def insert_plateau(self, t0, t1):
        """
        Modifies the piecewise function by:
        1. Keeping the left part (before t0) unchanged
        2. Creating a plateau from t0 to t1
        3. Shifting the right part (after t0) by (t1-t0)
        """
        if t1 < t0:
            raise ValueError("t1 must be greater than or equal to t0")

        if t1 == t0:
            return

        shift = t1 - t0
        new_segments = []

        shift = t1 - t0
        new_segments = []
        plateau_added = False

        # Process each segment
        for x_min, x_max, slope, intercept in self.segments:
            if x_max < t0:
                # Segment completely before t0 - keep unchanged
                new_segments.append((x_min, x_max, slope, intercept))
            elif x_min > t0:
                # Segment completely after t0 - shift right
                new_intercept = intercept - slope * shift
                new_segments.append(
                    (x_min + shift, x_max + shift, slope, new_intercept)
                )
            else:
                # Segment contains or touches t0
                # Find value at t0
                val_at_t0 = slope * t0 + intercept

                # Add segment up to t0 if needed
                if x_min < t0:
                    new_segments.append((x_min, t0, slope, intercept))

                # Add plateau if we haven't yet
                if not plateau_added:
                    new_segments.append((t0, t1, 0, val_at_t0))
                    plateau_added = True

                # Add shifted remainder if there is any
                if x_max > t0:
                    new_segments.append(
                        (t1, x_max + shift, slope, val_at_t0 - slope * t1)
                    )

        self.segments = new_segments

    def rescale_x(self, n):
        """
        Rescales the x-axis of the piecewise linear function by a factor of n.
        Args:
            n (float): The scaling factor for the x-axis. Must be non-zero.
        Returns:
            PiecewiseLinear: A new PiecewiseLinear object with the rescaled x-axis.
        Raises:
            ValueError: If n is zero.
        """
        if n == 0:
            raise ValueError("Scaling factor n cannot be zero")

        new_obj = self.copy()
        new_segments = []
        for x_min, x_max, slope, intercept in self.segments:
            new_x_min = x_min * n
            new_x_max = x_max * n
            new_slope = slope / n
            new_intercept = intercept
            new_segments.append((new_x_min, new_x_max, new_slope, new_intercept))
        new_obj.segments = new_segments

        return new_obj


@partial(jax.jit, static_argnums=(2))
def fta(
    x: Array,
    eta: float = 2,
    tiles: int = 20,
    lower_bound: float = -20,
    upper_bound: float = 20,
) -> Array:
    r"""Fuzzy Tiling Activation

    Computes the element-wise function:

    .. math::
        I_{\eta(,+}(x) = I_+(x - \eta) x + I_+(x - \eta)
        \mathrm{fta}(x) = 1 - I_{\eta(,+}\max(c - x, 0) + \max(x - c - \delta, 0))

    For more information, see
    `Fuzzy Tiling Activations: A Simple Approach to Learning Sparse Representations Online
    <https://arxiv.org/abs/1911.08068>`_.

    Args:
        x : input array
        eta : sparsity control parameter
        tiles : number of tiles
        lower_bound : lower bound for the input
        upper_bound : upper bound for the input

    Returns:
        An array.
    """
    delta = (upper_bound - lower_bound) / tiles
    c = lower_bound + jnp.arange(tiles) * delta
    c = c[None, :]
    x = x[..., None]
    z = 1 - fuzzy_indicator_function(relu(c - x) + relu(x - delta - c), eta)
    z = z.reshape(x.shape[0], -1)
    return z


@jax.jit
def fuzzy_indicator_function(x: Array, eta: float):
    return jnp.greater(eta, x).astype(x.dtype) * x + jnp.greater_equal(x, eta).astype(
        x.dtype
    )


def normalize(x, lower, upper):
    return (x - lower) / (upper - lower)
