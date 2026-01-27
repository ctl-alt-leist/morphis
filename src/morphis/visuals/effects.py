"""
Scheduled Visual Effects

Effects are visual modifications (opacity, scale, etc.) that are scheduled
to occur over a time range. The Animation system evaluates effects at each
capture time to determine the current visual state.

Effects are declarative: you schedule them ahead of time, and the animation
system applies them automatically when capturing frames.
"""

from abc import abstractmethod

from pydantic import BaseModel, ConfigDict


class Effect(BaseModel):
    """
    Base class for scheduled visual effects.

    An effect applies to a specific object over a time range [t_start, t_end].
    The evaluate() method returns the effect's value at any time t.
    """

    model_config = ConfigDict(frozen=True)

    object_id: int
    t_start: float
    t_end: float

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start

    def is_active(self, t: float) -> bool:
        """Check if this effect is active at time t."""
        return self.t_start <= t <= self.t_end

    def progress(self, t: float) -> float:
        """
        Return normalized progress [0, 1] at time t.

        Returns 0 before t_start, 1 after t_end, and linear interpolation
        between.
        """
        if t <= self.t_start:
            return 0.0
        if t >= self.t_end:
            return 1.0
        return (t - self.t_start) / self.duration

    @abstractmethod
    def evaluate(self, t: float) -> float:
        """Evaluate the effect at time t. Returns effect-specific value."""
        pass


class FadeIn(Effect):
    """
    Fade an object from invisible to visible.

    Opacity goes from 0.0 at t_start to 1.0 at t_end.
    """

    def evaluate(self, t: float) -> float:
        """Return opacity at time t."""
        return self.progress(t)


class FadeOut(Effect):
    """
    Fade an object from visible to invisible.

    Opacity goes from 1.0 at t_start to 0.0 at t_end.
    """

    def evaluate(self, t: float) -> float:
        """Return opacity at time t."""
        return 1.0 - self.progress(t)


class Hold(Effect):
    """
    Hold an object at constant opacity.

    This is useful for making an object visible for a duration after
    fading in, before fading out.
    """

    opacity: float = 1.0

    def evaluate(self, t: float) -> float:
        """Return constant opacity."""
        return self.opacity


def compute_opacity(effects: list[Effect], object_id: int, t: float) -> float | None:
    """
    Compute the effective opacity for an object at time t.

    Finds all effects that apply to this object and are active at time t,
    then combines them. If no effects apply, returns None (object not
    scheduled to be visible).

    Args:
        effects: List of all scheduled effects
        object_id: The object to compute opacity for
        t: Current time

    Returns:
        Opacity value [0, 1] or None if object has no active effects
    """
    # Find all effects for this object
    relevant = [e for e in effects if e.object_id == object_id]

    if not relevant:
        return None

    # Find the effect that's active at time t
    # If multiple are active, use the most recently started one
    active = [e for e in relevant if e.is_active(t)]

    if not active:
        # Check if we're past all effects (object should stay at final state)
        past = [e for e in relevant if t > e.t_end]
        if past:
            # Use the latest effect's final value
            latest = max(past, key=lambda e: e.t_end)
            return latest.evaluate(latest.t_end)
        # Before any effects - object not yet visible
        return 0.0

    # Use the most recently started active effect
    current = max(active, key=lambda e: e.t_start)
    return current.evaluate(t)
