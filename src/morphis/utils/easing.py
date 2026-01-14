"""
Easing Functions

Standard easing functions for smooth animations. Each function takes a parameter
t in [0, 1] and returns a value in [0, 1] with different acceleration profiles.
"""

from numpy import cos, pi, sin


def linear(t: float) -> float:
    """Linear interpolation (no easing)."""
    return t


def ease_in_quad(t: float) -> float:
    """Quadratic ease-in: slow start, accelerating."""
    return t * t


def ease_out_quad(t: float) -> float:
    """Quadratic ease-out: fast start, decelerating."""
    return 1 - (1 - t) * (1 - t)


def ease_in_out_quad(t: float) -> float:
    """Quadratic ease-in-out: slow start and end."""
    if t < 0.5:
        return 2 * t * t
    else:
        return 1 - pow(-2 * t + 2, 2) / 2


def ease_in_cubic(t: float) -> float:
    """Cubic ease-in: slow start, accelerating."""
    return t * t * t


def ease_out_cubic(t: float) -> float:
    """Cubic ease-out: fast start, decelerating."""
    return 1 - pow(1 - t, 3)


def ease_in_out_cubic(t: float) -> float:
    """Cubic ease-in-out: slow start and end, fast middle."""
    if t < 0.5:
        return 4 * t * t * t
    else:
        return 1 - pow(-2 * t + 2, 3) / 2


def ease_in_quart(t: float) -> float:
    """Quartic ease-in: very slow start."""
    return t * t * t * t


def ease_out_quart(t: float) -> float:
    """Quartic ease-out: very slow end."""
    return 1 - pow(1 - t, 4)


def ease_in_out_quart(t: float) -> float:
    """Quartic ease-in-out: very slow start and end."""
    if t < 0.5:
        return 8 * t * t * t * t
    else:
        return 1 - pow(-2 * t + 2, 4) / 2


def ease_in_sine(t: float) -> float:
    """Sinusoidal ease-in: gentle acceleration."""
    return 1 - cos((t * pi) / 2)


def ease_out_sine(t: float) -> float:
    """Sinusoidal ease-out: gentle deceleration."""
    return sin((t * pi) / 2)


def ease_in_out_sine(t: float) -> float:
    """Sinusoidal ease-in-out: gentle start and end."""
    return -(cos(pi * t) - 1) / 2


def ease_in_expo(t: float) -> float:
    """Exponential ease-in: very slow start, then explosive."""
    if t == 0:
        return 0
    return pow(2, 10 * t - 10)


def ease_out_expo(t: float) -> float:
    """Exponential ease-out: explosive start, very slow end."""
    if t == 1:
        return 1
    return 1 - pow(2, -10 * t)


def ease_in_out_expo(t: float) -> float:
    """Exponential ease-in-out: extreme acceleration/deceleration."""
    if t == 0:
        return 0
    if t == 1:
        return 1
    if t < 0.5:
        return pow(2, 20 * t - 10) / 2
    else:
        return (2 - pow(2, -20 * t + 10)) / 2
