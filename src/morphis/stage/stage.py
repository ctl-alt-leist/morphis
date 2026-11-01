"""
Stage - The Stage

The backend of the playground: the space, the tree of entries in it, which
one is selected, the clock, and the notices the user reads. Every front end
is a view over a stage, and a script drives one the same way a panel or a key
does. The stage knows nothing of any window, panel, or key.

The tree's root is the space itself. Beneath it are actors, each holding one
object handed in, and groups, holding other entries:

    stage = Stage(metric(space=4))
    stage.add("cube", Frame(*basis_vectors(g)), filled=True)
    stage.add("torus", clifford_torus, Torus())        # a shape built from dials
    stage.group("pair", ["cube", "torus"])
    stage.set("cube", "motion.rate.1,3", 0.5)
    stage.play()
    stage.advance(1 / 30)

Every change is an action, and every action is recorded as an event in the
journal and applied by `apply`, the one reducer; nothing else changes the
stage. An action checks what it is given before recording anything, so a
value outside its domain or a name that is not on the stage is refused where
it is given, and the journal holds only what happened.

An object's pose carries it rigidly. Poses compose down the tree, a member's
drawn pose being its group's pose applied after its own, each taken about the
entry's pivot. The stage carries the rest shape of every actor through its
drawn pose into the object it hands the Scene, so the Scene redraws it at the
next capture without knowing the stage exists.
"""

from collections.abc import Callable, Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

from numpy import zeros
from pydantic import BaseModel, ConfigDict

from morphis.elements import Frame, Metric, Vector
from morphis.stage import journal
from morphis.stage.carriers import carrier
from morphis.stage.catalog import CATALOG
from morphis.stage.dial import Dials
from morphis.stage.events import (
    Added,
    Baked,
    DialSet,
    Event,
    Moved,
    Paused,
    Played,
    Removed,
    Renamed,
    Reset,
    Saved,
    Selected,
    Stopped,
)
from morphis.stage.motion import Motion
from morphis.stage.operation import act, operation_dials, operation_map
from morphis.stage.pose import Pose
from morphis.stage.providers import provided
from morphis.stage.space import space_dials


SPACE = 0  # the identity of the space, at the root of the tree


class Notice(BaseModel):
    """A message for the user, worded once here for every front end."""

    model_config = ConfigDict(frozen=True)

    text: str
    level: Literal["info", "warning"] = "info"


# =============================================================================
# Entries
# =============================================================================


class Entry:
    """
    One entry of the outline: its identity, its name, its parent, and its
    dials, in named groups.

    Attributes:
        id: The identity, kept through renaming and moving
        name: The name the outline shows and a script looks it up by
        parent: The identity of the entry holding this one, or none for the space
        dials: The groups of dials, by name, in the order shown
    """

    def __init__(self, id: int, name: str, parent: int | None, dials: dict[str, Dials]) -> None:
        self.id = id
        self.name = name
        self.parent = parent
        self.dials = dials

    @property
    def kind(self) -> str:
        """What the entry is, for the outline."""
        return type(self).__name__

    @property
    def motion(self) -> Motion | None:
        """The entry's motion, if it moves or moves others."""
        return self.dials.get("motion")

    def get(self, path: str) -> Any:
        """
        Returns the value of a dial, by its group and its path within it.
        """
        group, _, rest = path.partition(".")

        return self._group(group).get(rest)

    def set(self, path: str, value: Any) -> None:
        group, _, rest = path.partition(".")
        self._group(group).set(rest, value)

    def _group(self, name: str) -> Dials:
        if name not in self.dials:
            raise KeyError(f"{self.name} has no dials named {name!r}")

        return self.dials[name]

    def __repr__(self) -> str:
        return f"{self.kind}({self.id}, {self.name!r})"


class Container(Entry):
    """An entry holding others, in order."""

    def __init__(self, id: int, name: str, parent: int | None, dials: dict[str, Dials]) -> None:
        super().__init__(id, name, parent, dials)
        self.children: list[int] = []


class Space(Container):
    """The root of the tree: the space itself, with the dials of the whole picture."""


class Placed(Entry):
    """
    An entry with a pose, taken about its pivot.

    Attributes:
        pose: Where the entry is, relative to its parent
        pivot: The point it turns about, in its parent's frame
    """

    def __init__(self, id: int, name: str, parent: int, dials: dict[str, Dials], metric: Metric) -> None:
        super().__init__(id, name, parent, dials)
        self.pose = Pose.identity(metric)
        self.pivot = Vector(zeros(metric.dim), grade=1, metric=metric)


class Group(Placed, Container):
    """Entries held together, moved together by the group's own pose."""

    def __init__(self, id: int, name: str, parent: int, metric: Metric) -> None:
        Placed.__init__(self, id, name, parent, {"motion": Motion(metric)}, metric)
        self.children = []


class Operation(Entry):
    """
    An entry that moves the entries beside it in its group, and so everything
    beneath them: a rotation, a translation, a boost.

    Attributes:
        label: What kind of operation it is, as the catalog names it
    """

    def __init__(self, id: int, name: str, parent: int, metric: Metric, label: str) -> None:
        super().__init__(id, name, parent, operation_dials(metric))
        self.label = label

    @property
    def kind(self) -> str:
        return self.label

    @property
    def on(self) -> bool:
        return self.dials["operation"].on


class Actor(Placed):
    """
    One object on the stage: the object as handed in, which is what is drawn,
    and a copy of it at rest, which the stage carries through the drawn pose
    into the object.

    Attributes:
        object: The object, as drawn
        rest: The object at rest, in its own frame
        origin: Where the object is drawn from, as drawn, if it has one
        build: The function the object is built from, given its shape dials
        step: The object's own law, given its dials, its rest, and a time
        options: How the Scene draws it
    """

    def __init__(
        self,
        id: int,
        name: str,
        parent: int,
        obj: Any,
        metric: Metric,
        dials: Dials | None = None,
        step: Callable[[Dials, Any, float], None] | None = None,
        origin: Vector | None = None,
        options: dict | None = None,
    ) -> None:
        self.build = obj if callable(obj) and dials is not None and step is None else None
        drawn = self.build(dials) if self.build is not None else obj
        groups = provided(drawn)
        if dials is not None:
            groups = {**groups, "shape" if self.build is not None else "step": dials}
        super().__init__(id, name, parent, groups, metric)
        self.object = drawn
        self.rest = _copy(drawn)
        # A vector or frame is drawn from a point, so that a velocity has somewhere to move it
        if origin is None and isinstance(drawn, (Vector, Frame)):
            origin = Vector(zeros(metric.dim), grade=1, metric=metric)
        self.origin = None if origin is None else origin.copy()
        self.rest_origin = origin
        self.step = step
        self.options = dict(options or {})
        if "opacity" in self.options and "look" in self.dials:
            self.dials["look"].opacity = self.options.pop("opacity")
        self.scene_id: str | None = None

    @property
    def kind(self) -> str:
        return type(self.object).__name__

    def rebuild(self) -> None:
        """
        Build the object again from its shape dials, as a new object at rest.
        """
        self.object = self.build(self.dials["shape"])
        self.rest = _copy(self.object)


# An entry, its name, or its identity
Reference = Entry | str | int


def _copy(obj: Any) -> Any:
    """A copy of an object to keep at rest, however its class copies."""
    own = getattr(type(obj), "copy", None)
    if own is not None and own is not BaseModel.copy:
        result = obj.copy()
    elif isinstance(obj, BaseModel):
        result = obj.model_copy(deep=True)
    else:
        result = deepcopy(obj)

    return result


# =============================================================================
# The Stage
# =============================================================================


class Stage:
    """
    The space, the tree of entries in it, the selection, the clock, and the
    journal every change is recorded in.

    Attributes:
        metric: The space
        scene: The Scene the actors are drawn in, if any
        entries: Every entry, by identity
        selection: The identity of the selected entry, or none
        time: The clock's time [s]
        playing: Whether the clock runs
        journal: Every event, in order
        notices: The messages for the user, in order
    """

    def __init__(self, metric: Metric, scene: Any = None) -> None:
        self.metric = metric
        self.scene = scene
        self.space = Space(SPACE, "space", None, space_dials(metric))
        self.entries: dict[int, Entry] = {SPACE: self.space}
        self.selection: int | None = None
        self.time = 0.0
        self.playing = False
        self.journal: list[Event] = []
        self.notices: list[Notice] = []
        self._arriving: dict[int, Entry] = {}
        self._subscribers: list[Callable[[], None]] = []
        self._next_id = 1
        if scene is not None:
            self._picture("view")
            self._picture("camera")

    # =========================================================================
    # Looking Up
    # =========================================================================

    def __getitem__(self, reference: Reference) -> Entry:
        return self.entry(reference)

    def entry(self, reference: Reference) -> Entry:
        """
        Returns an entry, given the entry itself, its name, or its identity.
        """
        if isinstance(reference, Entry):
            found = self.entries.get(reference.id)
        elif isinstance(reference, int):
            found = self.entries.get(reference)
        else:
            found = next((entry for entry in self.entries.values() if entry.name == reference), None)
        if found is None:
            raise KeyError(f"nothing on the stage is {reference!r}")

        return found

    def walk(self, start: Reference = SPACE) -> Iterator[Entry]:
        """
        Yields an entry and everything beneath it, in the order of the outline.
        """
        entry = self.entry(start)
        yield entry
        for child in getattr(entry, "children", ()):
            yield from self.walk(child)

    @property
    def selected(self) -> Entry | None:
        return None if self.selection is None else self.entries[self.selection]

    # =========================================================================
    # Actions
    # =========================================================================

    def add(
        self,
        name: str,
        obj: Any,
        dials: Dials | None = None,
        *,
        step: Callable[[Dials, Any, float], None] | None = None,
        parent: Reference = SPACE,
        position: int | None = None,
        origin: Vector | None = None,
        **options: Any,
    ) -> Actor:
        """
        Put an object on the stage, and return its actor.

        The object may be any object; what it can do depends on the dials
        provided for its class. Given with dials and no step, a callable is a
        shape built from them, and its dials are the actor's shape dials.
        Given with dials and a step, the step is the object's own law, called
        each tick with the dials, the object at rest, and the time elapsed.

        Args:
            name: The name to show; made unique if taken
            obj: The object, or the function a shape is built from
            dials: The shape's dials, or the law's
            step: The object's own law of motion
            parent: The group to put it in; the space by default
            position: Its place among the parent's children; last by default
            origin: Where the object is drawn from, a point carried with it
            **options: How the Scene draws it (color, filled, opacity, ...)
        """
        result = self._add(name, obj, dials, step, parent, position, origin, options, None)

        return result

    def create(
        self, line: str, name: str | None = None, parent: Reference = SPACE, position: int | None = None
    ) -> Entry:
        """
        Add what a catalog entry makes, named after the entry unless named
        otherwise, and return it.
        """
        made = CATALOG[line]
        name = line.lower() if name is None else name
        if not made.applies(self.metric):
            raise ValueError(f"there is no {line.lower()} in this space")

        if made.kind == "group":
            result = self.group(name, parent=parent)
        elif made.kind == "operation":
            holder = self._container(parent)
            id = self._take_id()
            name = self._unique(name)
            operation = Operation(id, name, holder.id, self.metric, line)
            for path, value in made.make(self.metric).items():
                operation.set(path, value)
            self._arriving[id] = operation
            self._record(Added(id=id, name=name, kind="operation", parent=holder.id, position=position, source=line))
            result = operation
        else:
            obj, dials, step, options = made.make(self.metric)
            result = self._add(name, obj, dials, step, parent, position, None, options, line)

        return result

    def group(self, name: str, members: list[Reference] = (), parent: Reference = SPACE) -> Group:
        """
        Put a group on the stage and move the members into it, in order.
        """
        holder = self._container(parent)
        found = [self.entry(member) for member in members]
        id = self._take_id()
        name = self._unique(name)
        self._arriving[id] = Group(id, name, holder.id, self.metric)
        self._record(Added(id=id, name=name, kind="group", parent=holder.id))
        for member in found:
            self.move(member, id)

        return self.entries[id]

    def remove(self, reference: Reference) -> None:
        """
        Take an entry off the stage, with everything beneath it.
        """
        entry = self.entry(reference)
        if entry.id == SPACE:
            raise ValueError("the space cannot be removed")
        self._record(Removed(id=entry.id))

    def select(self, reference: Reference | None) -> None:
        """
        Select an entry, or none.
        """
        id = None if reference is None else self.entry(reference).id
        self._record(Selected(id=id))

    def set(self, reference: Reference, path: str, value: Any) -> None:
        """
        Set a dial of an entry, by its group and path ("motion.rate.1,3"). A
        value outside the dial's domain is refused before anything is recorded.
        """
        entry = self.entry(reference)
        group, _, rest = path.partition(".")
        trial = entry._group(group).model_copy(deep=True)
        trial.set(rest, value)
        self._record(DialSet(id=entry.id, path=path, value=trial.get(rest)))

    def rename(self, reference: Reference, name: str) -> None:
        entry = self.entry(reference)
        self._record(Renamed(id=entry.id, name=self._unique(name, entry.id)))

    def move(self, reference: Reference, parent: Reference, position: int | None = None) -> None:
        """
        Move an entry into a group, or to another place among its siblings.
        """
        entry = self.entry(reference)
        holder = self._container(parent)
        if entry.id == SPACE or any(beneath.id == holder.id for beneath in self.walk(entry)):
            raise ValueError(f"{entry.name} cannot be moved into itself")
        self._record(Moved(id=entry.id, parent=holder.id, position=position))

    def play(self) -> None:
        self._record(Played())

    def pause(self) -> None:
        self._record(Paused())

    def stop(self, reference: Reference | None = None) -> None:
        """
        Set every rate of an entry's motion to zero, the selection's by default.
        """
        self._record(Stopped(id=self._moving(reference).id))

    def reset(self, reference: Reference | None = None) -> None:
        """
        Return an entry to rest, the selection's by default.
        """
        entry = self._moving(reference)
        if not isinstance(entry, Placed):
            raise ValueError(f"{entry.name} has no pose to reset")
        self._record(Reset(id=entry.id))

    def toggle_on(self, reference: Reference | None = None) -> None:
        """
        Turn an operation on or off, the selection by default.
        """
        operation = self._operation(reference)
        self.set(operation, "operation.on", not operation.on)

    def bake(self, reference: Reference | None = None) -> None:
        """
        Apply an operation once to everything it reaches, the selection by
        default, its rates read as amounts.
        """
        self._record(Baked(id=self._operation(reference).id))

    def scope(self, reference: Reference) -> list[Placed]:
        """
        Returns the entries an operation acts on directly: those beside it in
        its group that have a pose. Everything beneath them follows.
        """
        operation = self._operation(reference)
        result = [self.entries[id] for id in self.entries[operation.parent].children]
        result = [entry for entry in result if isinstance(entry, Placed)]

        return result

    def toggle_frame(self, reference: Reference | None = None) -> None:
        """
        Switch an entry's motion between extrinsic and intrinsic, the selection's by default.
        """
        entry = self._moving(reference)
        self.set(entry, "motion.intrinsic", not entry.motion.intrinsic)

    # =========================================================================
    # The Reducer
    # =========================================================================

    def apply(self, event: Event) -> None:
        """
        Apply an event to the stage: the one place the stage is changed.
        """
        if isinstance(event, Added):
            entry = self._arriving.pop(event.id, None) or self._rebuild(event)
            self._next_id = max(self._next_id, event.id + 1)
            self.entries[event.id] = entry
            self._place(entry, event.parent, event.position)
            self._draw(entry)
            self.notice(f"added {entry.name}")
        elif isinstance(event, Removed):
            self._take_off(self.entries[event.id])
        elif isinstance(event, Selected):
            self.selection = event.id
        elif isinstance(event, DialSet):
            self._set(self.entries[event.id], event.path, event.value)
        elif isinstance(event, Renamed):
            self.entries[event.id].name = event.name
        elif isinstance(event, Moved):
            entry = self.entries[event.id]
            self.entries[entry.parent].children.remove(entry.id)
            self._place(entry, event.parent, event.position)
        elif isinstance(event, Played):
            self.playing = True
        elif isinstance(event, Paused):
            self.playing = False
        elif isinstance(event, Stopped):
            self.entries[event.id].motion.stop()
        elif isinstance(event, Baked):
            self._operate(self.entries[event.id], 1.0)
        elif isinstance(event, Reset):
            entry = self.entries[event.id]
            entry.pose = Pose.identity(self.metric)

    def _record(self, event: Event) -> None:
        stamped = event.model_copy(update={"time": self.time})
        self.journal.append(stamped)
        self.apply(stamped)
        self.carry()
        self._changed()

    # =========================================================================
    # Time
    # =========================================================================

    def advance(self, dt: float) -> None:
        """
        Advance the clock by dt of wall-clock time, scaled by the clock's
        speed, when it is playing: every entry's motion steps, and every
        actor's own law; then every actor is carried to where it now is.
        """
        if self.playing:
            self._run(dt * self.space.dials["clock"].speed)
        self.carry()
        self._changed()

    # =========================================================================
    # Saving and Replaying
    # =========================================================================

    def save(self, path: str | Path) -> None:
        """
        Write the stage's journal, which is its whole state and history,
        marked with the time it was saved at.
        """
        self._record(Saved())
        journal.write(path, self.metric, self.journal)

    @classmethod
    def load(
        cls,
        path: str | Path,
        setup: Callable[["Stage"], None] | None = None,
        scene: Any = None,
        frame_rate: float | None = None,
    ) -> "Stage":
        """
        Returns the stage a journal was written from, by replaying its events.

        Entries made from the catalog are rebuilt from it. Entries a script
        handed in are made again by running the script's setup on the new
        stage first; its events must be the journal's first, and the rest are
        replayed after them. Given a frame rate, the time between events is
        stepped and captured at that rate, so a session played by hand is
        recorded smoothly (inside `scene.record`).
        """
        space, events = journal.read(path)
        stage = cls(space, scene)
        if setup is not None:
            setup(stage)
        count = len(stage.journal)
        for made, written in zip(stage.journal, events[:count], strict=False):
            if made.model_dump(exclude={"time"}) != written.model_dump(exclude={"time"}):
                raise ValueError(f"the setup does not match the journal: {made} where it wrote {written}")
        stage.replay(events[count:], frame_rate)

        return stage

    def replay(self, events: list[Event], frame_rate: float | None = None, until: float | None = None) -> None:
        """
        Apply events in order, running the clock up to each one's time first:
        in one step, or given a frame rate, in steps of one frame, each
        captured. Steps of constant rates compose exactly, so either way the
        stage reaches every event in the same state, except where an entry is
        turned at once by its own extrinsic motion and by an operation.
        """
        for event in events:
            self._run_to(event.time, frame_rate)
            self.journal.append(event)
            self.apply(event)
            self.carry()
            self._changed()
        if until is not None:
            self._run_to(until, frame_rate)

    def _run_to(self, time: float, frame_rate: float | None) -> None:
        while self.playing and time - self.time > 1e-12:
            remaining = time - self.time
            self._run(remaining if frame_rate is None else min(1.0 / frame_rate, remaining))
            if frame_rate is not None and self.scene is not None:
                self.carry()
                self.capture()

    def _run(self, elapsed: float) -> None:
        """Advance the clock by elapsed stage time: own motions and laws, then operations, in order."""
        self.time += elapsed
        for entry in self.walk():
            self._step(entry, elapsed)
        for entry in list(self.walk()):
            if isinstance(entry, Operation) and entry.on:
                self._operate(entry, elapsed)

    def carry(self) -> None:
        """
        Carry every actor's rest shape through its drawn pose into the object
        the Scene draws.
        """
        self._carry(self.space, Pose.identity(self.metric))

    def capture(self) -> None:
        """
        Draw the stage as it is now, in the Scene.
        """
        self.scene.capture(self.time)

    # =========================================================================
    # Front Ends
    # =========================================================================

    def subscribe(self, callback: Callable[[], None]) -> None:
        """
        Be told whenever the stage changes, to redraw from what it holds.
        """
        self._subscribers.append(callback)

    def notice(self, text: str, level: Literal["info", "warning"] = "info") -> None:
        self.notices.append(Notice(text=text, level=level))

    # =========================================================================
    # Internals
    # =========================================================================

    def _changed(self) -> None:
        for callback in self._subscribers:
            callback()

    def _add(
        self,
        name: str,
        obj: Any,
        dials: Dials | None,
        step: Callable | None,
        parent: Reference,
        position: int | None,
        origin: Vector | None,
        options: dict,
        source: str | None,
    ) -> Actor:
        holder = self._container(parent)
        id = self._take_id()
        name = self._unique(name)
        self._arriving[id] = Actor(id, name, holder.id, obj, self.metric, dials, step, origin, options)
        self._record(Added(id=id, name=name, kind="actor", parent=holder.id, position=position, source=source))

        return self.entries[id]

    def _rebuild(self, event: Added) -> Entry:
        """An entry made again from the catalog line it came from, for replay."""
        if event.source is None and event.kind != "group":
            raise ValueError(f"{event.name} was handed in by a script; load the journal with that script's setup")

        if event.kind == "group":
            result = Group(event.id, event.name, event.parent, self.metric)
        elif event.kind == "operation":
            result = Operation(event.id, event.name, event.parent, self.metric, event.source)
            for path, value in CATALOG[event.source].make(self.metric).items():
                result.set(path, value)
        else:
            obj, dials, step, options = CATALOG[event.source].make(self.metric)
            result = Actor(event.id, event.name, event.parent, obj, self.metric, dials, step, None, options)

        return result

    def _take_id(self) -> int:
        id = self._next_id
        self._next_id += 1

        return id

    def _unique(self, name: str, keep: int | None = None) -> str:
        taken = {entry.name for entry in self.entries.values() if entry.id != keep}
        result = name
        n = 2
        while result in taken:
            result = f"{name} {n}"
            n += 1

        return result

    def _container(self, reference: Reference) -> Container:
        entry = self.entry(reference)
        if not isinstance(entry, Container):
            raise ValueError(f"{entry.name} is not a group")

        return entry

    def _moving(self, reference: Reference | None) -> Entry:
        entry = self.selected if reference is None else self.entry(reference)
        if entry is None or entry.motion is None:
            raise ValueError("select an entry that moves")

        return entry

    def _place(self, entry: Entry, parent: int, position: int | None) -> None:
        entry.parent = parent
        children = self.entries[parent].children
        children.insert(len(children) if position is None else position, entry.id)

    def _take_off(self, entry: Entry) -> None:
        for child in list(getattr(entry, "children", ())):
            self._take_off(self.entries[child])
        if isinstance(entry, Actor) and entry.scene_id is not None:
            self.scene.remove(entry.scene_id)
        self.entries[entry.parent].children.remove(entry.id)
        del self.entries[entry.id]
        if self.selection == entry.id:
            self.selection = None
        self.notice(f"removed {entry.name}")

    def _set(self, entry: Entry, path: str, value: Any) -> None:
        entry.set(path, value)
        group = path.partition(".")[0]
        if isinstance(entry, Actor) and entry.build is not None and group == "shape":
            if entry.scene_id is not None:
                self.scene.remove(entry.scene_id)
            entry.rebuild()
            self._draw(entry)
        elif isinstance(entry, Actor) and group == "look" and entry.scene_id is not None:
            self._style(entry)
        elif isinstance(entry, Space) and group in ("view", "camera") and self.scene is not None:
            self._picture(group)

    def _picture(self, group: str) -> None:
        """Hand the Scene the space's view or camera."""
        if group == "view":
            self.scene.set_projection(self.space.dials["view"].projective)
        else:
            self.scene.camera(**self.space.dials["camera"].placement)

    def _style(self, entry: Actor) -> None:
        look = entry.dials["look"]
        self.scene.style(entry.scene_id, opacity=look.opacity, visible=look.visible)

    def _draw(self, entry: Entry) -> None:
        if self.scene is not None and isinstance(entry, Actor):
            options = {**entry.options, **({} if entry.origin is None else {"origin": entry.origin})}
            entry.scene_id = self.scene.add(entry.object, **options)
            if "look" in entry.dials:
                self._style(entry)

    def _operation(self, reference: Reference | None) -> Operation:
        entry = self.selected if reference is None else self.entry(reference)
        if not isinstance(entry, Operation):
            raise ValueError("select an operation")

        return entry

    def _operate(self, operation: Operation, elapsed: float) -> None:
        G = operation_map(operation.dials, elapsed)
        for entry in self.scope(operation):
            entry.pose = act(G, entry.pose, entry.pivot, operation.motion.intrinsic)

    def _step(self, entry: Entry, elapsed: float) -> None:
        if isinstance(entry, Placed) and entry.motion is not None:
            entry.pose = entry.motion.advance(entry.pose, elapsed)
        if isinstance(entry, Actor) and entry.step is not None:
            entry.step(entry.dials["step"], entry.rest, elapsed)

    def _carry(self, entry: Entry, outer: Pose) -> None:
        drawn = outer.compose(entry.pose.about(entry.pivot)) if isinstance(entry, Placed) else outer
        if isinstance(entry, Actor):
            carry = carrier(entry.object)
            if carry is not None:
                carry(entry.rest, entry.object, drawn)
            if entry.origin is not None:
                entry.origin.data[...] = drawn.points(entry.rest_origin).data
        for child in getattr(entry, "children", ()):
            self._carry(self.entries[child], drawn)
