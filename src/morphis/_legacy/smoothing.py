"""
Temporal Smoothing Functions

Weight functions for distributing a transformation parameter across time.
Unlike easing functions (which give cumulative progress 0→1), smoothing
functions give instantaneous weights that sum to 1 over the interval.

Mathematical Model:
    delta_lambda = f(t, T) * dt * lambda_total
    where sum(f(t_i, T) * dt) ≈ 1

Relationship to Easing:
    If g(t) is an easing function (cumulative), then f(t) = g'(t) is smoothing.
    - Easing: g(t) ∈ [0, 1], g(0) = 0, g(1) = 1
    - Smoothing: f(t) ≥ 0, integral(f, 0, 1) = 1

Usage:
    for i in range(N):
        t = i * dt
        delta_angle = smooth_in_out_cubic(t, T) * dt * total_angle
        M = rotor(B, delta_angle)
        blade.data[...] = (M * blade * ~M)[blade.grade].data
"""

from math import pi, sin
from typing import Callable


def smooth_linear(t: float, T: float) -> float:
    """
    Constant weight (linear motion).

    The simplest smoothing: constant velocity throughout.
    Integral over [0, T] equals 1.

    Args:
        t: Current time
        T: Total duration

    Returns:
        Weight at time t (always 1/T)
    """
    if t < 0 or t > T:
        return 0.0
    return 1.0 / T


def smooth_in_out_cubic(t: float, T: float) -> float:
    """
    Cubic ease-in-out smoothing (derivative of cubic ease).

    Starts slow, accelerates through middle, decelerates at end.
    This is the derivative of:
        g(s) = 4s³             for s < 0.5
        g(s) = 1 - 4(1-s)³     for s >= 0.5

    The derivative g'(s) is:
        g'(s) = 12s²           for s < 0.5
        g'(s) = 12(1-s)²       for s >= 0.5

    Both branches equal 3 at s=0.5, ensuring continuity.

    Args:
        t: Current time
        T: Total duration

    Returns:
        Instantaneous weight at time t
    """
    if t < 0 or t > T:
        return 0.0

    s = t / T  # Normalized time [0, 1]

    if s < 0.5:
        # g'(s) = 12s²
        return 12 * s * s / T
    else:
        # g'(s) = 12(1-s)²
        u = 1 - s
        return 12 * u * u / T


def smooth_in_out_sine(t: float, T: float) -> float:
    """
    Sinusoidal ease-in-out smoothing (derivative of sine ease).

    Smooth sinusoidal acceleration/deceleration.
    This is the derivative of:
        g(s) = (1 - cos(pi * s)) / 2

    g'(s) = pi * sin(pi * s) / 2

    Args:
        t: Current time
        T: Total duration

    Returns:
        Instantaneous weight at time t
    """
    if t < 0 or t > T:
        return 0.0

    s = t / T  # Normalized time [0, 1]
    return (pi / 2) * sin(pi * s) / T


def smooth_in_quad(t: float, T: float) -> float:
    """
    Quadratic ease-in smoothing (accelerating).

    Starts slow, accelerates throughout.
    Derivative of g(s) = s².

    Args:
        t: Current time
        T: Total duration

    Returns:
        Instantaneous weight at time t
    """
    if t < 0 or t > T:
        return 0.0

    s = t / T
    return 2 * s / T


def smooth_out_quad(t: float, T: float) -> float:
    """
    Quadratic ease-out smoothing (decelerating).

    Starts fast, decelerates throughout.
    Derivative of g(s) = 1 - (1 - s)².

    Args:
        t: Current time
        T: Total duration

    Returns:
        Instantaneous weight at time t
    """
    if t < 0 or t > T:
        return 0.0

    s = t / T
    return 2 * (1 - s) / T


def smooth_in_out_quad(t: float, T: float) -> float:
    """
    Quadratic ease-in-out smoothing.

    Accelerates to midpoint, decelerates after.

    Args:
        t: Current time
        T: Total duration

    Returns:
        Instantaneous weight at time t
    """
    if t < 0 or t > T:
        return 0.0

    s = t / T

    if s < 0.5:
        return 4 * s / T
    else:
        return 4 * (1 - s) / T


# Type alias for smoothing functions
Smoother = Callable[[float, float], float]


# Named smoothing functions for string-based lookup
SMOOTHERS: dict[str, Smoother] = {
    "linear": smooth_linear,
    "in_out_cubic": smooth_in_out_cubic,
    "in_out_sine": smooth_in_out_sine,
    "in_quad": smooth_in_quad,
    "out_quad": smooth_out_quad,
    "in_out_quad": smooth_in_out_quad,
}


def get_smoother(name: str) -> Smoother:
    """
    Retrieve a smoothing function by name.

    Args:
        name: Smoother name (e.g., "linear", "in_out_cubic")

    Returns:
        The smoothing function

    Raises:
        KeyError: If name not found
    """
    if name not in SMOOTHERS:
        available = ", ".join(SMOOTHERS.keys())
        raise KeyError(f"Unknown smoother '{name}'. Available: {available}")
    return SMOOTHERS[name]
