"""
Stage - Events

Every change to a stage is an event, appended to its journal with the clock
time it happened at and applied by the stage's one reducer. The state every
front end draws from is the journal's events applied in order, so a session
can be replayed, and between two events every rate is constant, so the poses
at each event's time do not depend on how finely the time between was
stepped.

Entries are named by their identity, a number given when they are added and
kept through renaming and moving; the space at the root of the tree is 0.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Event(BaseModel):
    """
    Attributes:
        time: The stage's clock time when the event happened [s]
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    time: float = 0.0


class Added(Event):
    """An entry put on the stage: an actor, or a group to hold others."""

    type: Literal["added"] = "added"
    id: int
    name: str
    kind: Literal["actor", "group"]
    parent: int = 0
    position: int | None = Field(default=None, description="Place among the parent's children; last if none")
    source: str | None = Field(default=None, description="The catalog entry it was made from; none if handed in")


class Removed(Event):
    """An entry taken off the stage, with everything beneath it."""

    type: Literal["removed"] = "removed"
    id: int


class Selected(Event):
    """An entry selected, or none."""

    type: Literal["selected"] = "selected"
    id: int | None


class DialSet(Event):
    """A dial of an entry set to a value."""

    type: Literal["dial set"] = "dial set"
    id: int
    path: str
    value: Any


class Renamed(Event):
    type: Literal["renamed"] = "renamed"
    id: int
    name: str


class Moved(Event):
    """An entry moved in the tree, to a new parent or a new place among its siblings."""

    type: Literal["moved"] = "moved"
    id: int
    parent: int
    position: int | None = None


class Played(Event):
    type: Literal["played"] = "played"


class Paused(Event):
    type: Literal["paused"] = "paused"


class Stopped(Event):
    """Every rate of an entry's motion set to zero."""

    type: Literal["stopped"] = "stopped"
    id: int


class Reset(Event):
    """An entry's pose returned to rest."""

    type: Literal["reset"] = "reset"
    id: int


AnyEvent = Annotated[
    Added | Removed | Selected | DialSet | Renamed | Moved | Played | Paused | Stopped | Reset,
    Field(discriminator="type"),
]
