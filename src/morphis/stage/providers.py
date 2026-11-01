"""
Stage - Dial Providers

Which dials an object gets. A provider is a function from an object to a
group of dials, registered against a class under a name; an entry's dials are
the groups of every provider registered for a class the object is an
instance of, from the most general class to the most specific. A new kind of
object gets dials by registering a provider for its class, with no change to
the stage or any front end.

    @provide(Vector, "motion")
    def motion(element: Vector) -> Motion:
        return Motion(element.metric)
"""

from collections.abc import Callable
from typing import Any

from morphis.elements import Frame, Surface, Vector
from morphis.manifold import Mesh
from morphis.manifold.domain import Interval
from morphis.stage.dial import Dial, Dials
from morphis.stage.motion import Motion


Provider = Callable[[Any], Dials]

PROVIDERS: dict[type, dict[str, Provider]] = {}


def provide(kind: type, name: str) -> Callable[[Provider], Provider]:
    """
    Register a provider of a group of dials, under a name, for a class of object.
    """

    def register(provider: Provider) -> Provider:
        PROVIDERS.setdefault(kind, {})[name] = provider

        return provider

    return register


def provided(obj: Any) -> dict[str, Dials]:
    """
    Returns the groups of dials provided for an object, by name, from its most
    general class to its most specific; a nearer class's provider replaces a
    farther one's of the same name.
    """
    result = {}
    for kind in reversed(type(obj).__mro__):
        for name, provider in PROVIDERS.get(kind, {}).items():
            result[name] = provider(obj)

    return result


# =============================================================================
# The Common Providers
# =============================================================================


class Look(Dials):
    """How a drawn object looks: how opaque it is, and whether it is shown."""

    opacity: float = Dial(1.0, Interval(0.0, 1.0))
    visible: bool = True


DRAWN = (Vector, Frame, Surface, Mesh)


def _motion(element: Vector | Frame | Surface | Mesh) -> Motion:
    return Motion(element.metric)


def _look(element: Vector | Frame | Surface | Mesh) -> Look:
    return Look()


for drawn in DRAWN:
    provide(drawn, "motion")(_motion)
    provide(drawn, "look")(_look)
