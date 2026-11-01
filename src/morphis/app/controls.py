"""
What each command does, for every front end.

The vocabulary says which key and label a command has; this says what
pressing it does, when it can be pressed, and how its button reads. A front
end hands every key it receives to `key`, by its neutral name, or presses a
button with `press`; it never decides for itself what a key means, so the
picture and the panel cannot drift apart in behaviour any more than in keys.

The controls also hold where the user is: which screen is shown, which list
has focus, the dial highlighted in the inspector, a plane half named by its
first digit, and a command waiting on its second press. The selection itself
is the stage's, so a script or another front end that changes it changes it
here too.

A few commands need something only a front end has: a line of typed text (a
dial's value, a filter, where to save) or a choice from a picker (the
catalog, the screens). The front end hands in how it asks, and gives what it
was told back to `answer` or `take`; what that means is decided here.

A command that cannot run now does nothing but say why, in a notice, and an
action the stage refuses (a value outside its domain, a name not on the
stage) is said the same way, so nothing typed is lost silently.
"""

from collections.abc import Callable
from math import pi
from typing import Any, Literal

from pydantic import ValidationError

from morphis.app.vocabulary import CONFIRMED, PLACEHOLDERS, Command, Screen, direction
from morphis.elements.metric import Label
from morphis.stage import Stage
from morphis.stage.catalog import sections
from morphis.stage.dial import DialSpec
from morphis.stage.stage import Actor, Entry, Operation, Placed


Focus = Literal["outline", "inspector"]

# A picker's line: the key that takes it ("" for none), what taking it means, and how it reads.
Choice = tuple[str, Any, str]

# How far one press of an arrow turns a dial
RATE_STEP = 0.1  # a rate, or any number without a domain
ANGLE_STEP = pi / 36  # an angle, five degrees
INTERVAL_STEPS = 50  # an interval, in this many presses end to end

# The commands that ask for a line of text, and what the line is, as a front end prompts for it.
ASKS = {Command.FIND: "find", Command.TAKE: "value", Command.SAVE: "save to", Command.RECORD: "record to"}

# The commands that open a picker instead, with its title.
PICKERS = {Command.ADD: "add", Command.GO: "go to"}

# Commands live on every screen; the rest belong to the Stage screen.
EVERYWHERE = (Command.GO, Command.DONE, Command.SAVE, Command.OPEN, Command.RECORD)


class Controls:
    """
    The commands over a stage.

    Attributes:
        stage: The stage the commands act on
        screen: The screen shown
        focus: The list the arrows walk, the outline or the inspector
        highlight: The path of the dial highlighted in the inspector, its group first
        sense: +1, or -1 when the highlighted plane was named in reversed order
        naming: The first label of a plane half named, or none
        waiting: The command waiting on its second press, or none
        filter: The text the outline is filtered by
        recording: Where the picture is being recorded to, or none
    """

    def __init__(
        self,
        stage: Stage,
        ask: Callable[[Command], None] = lambda command: None,
        record: Callable[[str | None], None] = lambda path: None,
    ) -> None:
        """
        Args:
            ask: Asks the user for a line of text for a command in ASKS, or
                opens the picker for a command in PICKERS
            record: Starts recording the picture to a path, or stops on none
        """
        self.stage = stage
        self._ask = ask
        self._record = record
        self.screen = Screen.STAGE
        self.focus: Focus = "outline"
        self.highlight: str | None = None
        self.sense = 1.0
        self.naming: Label | None = None
        self.waiting: Command | None = None
        self.filter = ""
        self.recording: str | None = None
        self.saved_to: str | None = None
        self.actions: dict[Command, Callable[[], None]] = {
            Command.ADD: lambda: ask(Command.ADD),
            Command.PLAY: lambda: self.stage.pause() if self.stage.playing else self.stage.play(),
            Command.STOP: lambda: self.stage.stop(),
            Command.RESET: lambda: self.stage.reset(),
            Command.INTRINSIC: lambda: self.stage.toggle_frame(),
            Command.HIDE: self._hide,
            Command.ON: lambda: self.stage.toggle_on(),
            Command.BAKE: lambda: self.stage.bake(),
            Command.REMOVE: lambda: self.stage.remove(self.stage.selection),
            Command.PREVIOUS: lambda: self._walk(-1),
            Command.NEXT: lambda: self._walk(+1),
            Command.LOWER: lambda: self._turn(-1),
            Command.RAISE: lambda: self._turn(+1),
            Command.FOCUS: self._toggle_focus,
            Command.TAKE: self._take,
            Command.FIND: lambda: ask(Command.FIND),
            Command.GO: lambda: ask(Command.GO),
            Command.DONE: self._done,
            Command.SAVE: self._save,
            Command.OPEN: lambda: self.stage.notice("opening a journal is not built yet", "warning"),
            Command.RECORD: self._toggle_recording,
        }

    # =========================================================================
    # Keys and Buttons
    # =========================================================================

    def key(self, name: str) -> None:
        """
        A key, by its neutral name: a command's, or a digit naming a direction.
        """
        commands = {command.key: command for command in Command}
        label = direction(name)
        if name in commands:
            self.press(commands[name])
        elif label is not None and self.screen is Screen.STAGE:
            self._name(label)

    def press(self, command: Command) -> None:
        """
        Press a command: run it if it can run now, or say why not. A command
        that cannot be undone waits for a second press of itself, and any
        other command calls the wait off.
        """
        if not self.enabled(command):
            self.stage.notice(self._why_not(command), "warning")
        elif command in CONFIRMED and self.waiting is not command:
            self.waiting = command
            self.stage.notice(f"press {command.glyph()} again to {command.label.lower()} {self.stage.selected.name}")
        else:
            if command is not Command.DONE:
                self.waiting = None
            self._run(command)

    def enabled(self, command: Command) -> bool:
        """
        Returns whether the command can be pressed now: on its screen, and with
        the selection it needs.
        """
        selected = self.stage.selected
        needs = {
            "always": True,
            "entry": selected is not None and selected is not self.stage.space,
            "motion": selected is not None and selected.motion is not None,
            "pose": isinstance(selected, Placed),
            "drawn": isinstance(selected, Actor) and "look" in selected.dials,
            "operation": isinstance(selected, Operation),
        }
        on_screen = self.screen is Screen.STAGE or command in EVERYWHERE
        result = on_screen and needs[command.needs]

        return result

    def status(self, command: Command) -> Literal["lit", "waiting", ""]:
        """
        Returns how a command's button reads: lit while what it toggles is on,
        waiting while it waits on its second press, or plain.
        """
        selected = self.stage.selected
        lit = {
            Command.PLAY: self.stage.playing,
            Command.INTRINSIC: selected is not None and selected.motion is not None and selected.motion.intrinsic,
            Command.HIDE: isinstance(selected, Actor) and "look" in selected.dials and not selected.get("look.visible"),
            Command.ON: isinstance(selected, Operation) and selected.on,
            Command.RECORD: self.recording is not None,
        }
        if self.waiting is command:
            result = "waiting"
        elif lit.get(command, False):
            result = "lit"
        else:
            result = ""

        return result

    def _run(self, command: Command) -> None:
        try:
            self.actions[command]()
        except (ValueError, KeyError) as error:
            self.stage.notice(_worded(error), "warning")

    def _why_not(self, command: Command) -> str:
        reasons = {
            "entry": f"select an entry to {command.label.lower()} it",
            "motion": f"select an entry that moves to {command.label.lower()} it",
            "pose": f"select an entry with a pose to {command.label.lower()} it",
            "drawn": f"select a drawn entry to {command.label.lower()} it",
            "operation": f"select an operation to {command.label.lower()} it",
        }
        if self.screen is not Screen.STAGE and command not in EVERYWHERE:
            result = f"{command.label.lower()} is on the Stage screen: esc goes back"
        else:
            result = reasons[command.needs]

        return result

    # =========================================================================
    # Pickers and Typed Lines
    # =========================================================================

    def choices(self, command: Command) -> list[Choice]:
        """
        Returns the lines of a command's picker: the catalog entries that
        apply in the space, or the screens, each under its own key.
        """
        if command is Command.ADD:
            lines = [line for listed in sections(self.stage.metric).values() for line in listed]
            letters = iter("abcdefghijklmnopqrstuvwxyz")
            result = [(next(letters), line, line) for line in lines]
        elif command is Command.GO:
            result = [(screen.key, screen, screen.label) for screen in Screen]
        else:
            result = []

        return result

    def take(self, command: Command, choice: Any) -> None:
        """
        Take a line of a command's picker.
        """
        if command is Command.ADD:
            self._add(choice)
        elif command is Command.GO:
            self.go(choice)

    def answer(self, command: Command, text: str) -> None:
        """
        Take the line of text a command asked for.
        """
        text = text.strip()
        try:
            if command is Command.FIND:
                self.filter = text
                self._select_first()
            elif command is Command.TAKE and self.highlight is not None:
                self.stage.set(self.stage.selected, self.highlight, _typed(text))
            elif command is Command.SAVE and text:
                self.stage.save(text)
                self.saved_to = text
                self.stage.notice(f"saved to {text}")
            elif command is Command.RECORD and text:
                self._record(text)
                self.recording = text
                self.stage.notice(f"recording to {text}")
        except (ValueError, KeyError, ValidationError) as error:
            self.stage.notice(_worded(error), "warning")

    # =========================================================================
    # Screens, the Outline, and the Inspector
    # =========================================================================

    def go(self, screen: Screen) -> None:
        self.screen = screen
        if screen in PLACEHOLDERS:
            self.stage.notice(f"the {screen.label} screen is not built yet")

    def outline(self) -> list[Entry]:
        """
        Returns the entries the outline lists, in order, those matching the
        filter by name or kind; the space always first.
        """
        found = self.filter.lower()
        result = [
            entry
            for entry in self.stage.walk()
            if entry is self.stage.space or found in entry.name.lower() or found in entry.kind.lower()
        ]

        return result

    def dials(self) -> list[DialSpec]:
        """
        Returns the dials of the selection, every path starting with its group.
        """
        selected = self.stage.selected
        groups = {} if selected is None else selected.dials
        result = [spec for name, group in groups.items() for spec in group.describe(f"{name}.")]

        return result

    def _walk(self, step: int) -> None:
        if self.focus == "outline":
            entries = self.outline()
            ids = [entry.id for entry in entries]
            current = self.stage.selection
            if current in ids:
                n = min(max(ids.index(current) + step, 0), len(ids) - 1)
            else:
                n = 0 if step > 0 else len(ids) - 1
            self._select(ids[n])
        else:
            paths = [spec.path for spec in self.dials()]
            if self.highlight in paths:
                n = min(max(paths.index(self.highlight) + step, 0), len(paths) - 1)
            else:
                n = 0 if step > 0 else len(paths) - 1
            if paths:
                self.highlight, self.sense = paths[n], 1.0

    def _select(self, id: int | None) -> None:
        if id != self.stage.selection:
            self.stage.select(id)
            self.highlight = None
            self.naming = None
            self.sense = 1.0

    def _select_first(self) -> None:
        listed = [entry for entry in self.outline() if entry is not self.stage.space]
        if listed:
            self._select(listed[0].id)

    def _toggle_focus(self) -> None:
        self.focus = "inspector" if self.focus == "outline" else "outline"

    def _take(self) -> None:
        if self.focus == "inspector" and self.highlight is not None:
            self._ask(Command.TAKE)
        elif self.focus == "outline" and self.stage.selection is not None:
            self.focus = "inspector"

    def _done(self) -> None:
        if self.waiting is not None:
            self.waiting = None
        elif self.screen is not Screen.STAGE:
            self.screen = Screen.STAGE
        elif self.naming is not None:
            self.naming = None
        elif self.focus == "inspector":
            self.focus = "outline"
        elif self.stage.selection is not None:
            self._select(None)

    # =========================================================================
    # Dials
    # =========================================================================

    def _name(self, label: Label) -> None:
        """A digit: the first names half a plane, the second completes it and highlights its rate."""
        selected = self.stage.selected
        if selected is None or selected.motion is None:
            self.stage.notice("select an entry that moves to name a plane", "warning")
        elif self.naming is None:
            self.naming = label
        else:
            first, self.naming = self.naming, None
            forward, backward = f"{first},{label}", f"{label},{first}"
            rates = selected.motion.rate
            if forward in rates:
                self.highlight, self.sense = f"motion.rate.{forward}", 1.0
                self.focus = "inspector"
            elif backward in rates:
                self.highlight, self.sense = f"motion.rate.{backward}", -1.0
                self.focus = "inspector"
            else:
                self.stage.notice(f"there is no plane e{first}{label} in this space", "warning")

    def _turn(self, step: int) -> None:
        """Turn the highlighted dial down or up by one step of its kind."""
        specs = {spec.path: spec for spec in self.dials()}
        spec = specs.get(self.highlight)
        if spec is None:
            self.stage.notice("highlight a dial to turn it: ⇥ to the inspector, or name a plane", "warning")
        else:
            entry = self.stage.selected
            value = entry.get(spec.path)
            self.stage.set(entry, spec.path, _turned(spec, value, self.sense * step))

    def _hide(self) -> None:
        entry = self.stage.selected
        self.stage.set(entry, "look.visible", not entry.get("look.visible"))

    def _add(self, line: str) -> None:
        """Add a catalog entry beside the selection, inside it if it is a group, and select it."""
        selected = self.stage.selected
        if selected is None or selected is self.stage.space:
            made = self.stage.create(line)
        elif hasattr(selected, "children"):
            made = self.stage.create(line, parent=selected)
        else:
            siblings = self.stage.entries[selected.parent].children
            made = self.stage.create(line, parent=selected.parent, position=siblings.index(selected.id) + 1)
        self._select(made.id)

    # =========================================================================
    # The Playground Itself
    # =========================================================================

    def _save(self) -> None:
        if self.saved_to is None:
            self._ask(Command.SAVE)
        else:
            self.answer(Command.SAVE, self.saved_to)

    def _toggle_recording(self) -> None:
        if self.recording is None:
            self._ask(Command.RECORD)
        else:
            self._record(None)
            self.stage.notice(f"recorded to {self.recording}")
            self.recording = None


# =============================================================================
# Helpers
# =============================================================================


def _turned(spec: DialSpec, value: Any, step: float) -> Any:
    """A dial's value one step down or up, the step suited to its kind."""
    if spec.kind == "toggle":
        result = not value
    elif spec.kind == "choice":
        n = spec.choices.index(value)
        result = spec.choices[(n + int(step)) % len(spec.choices)]
    elif spec.kind == "angle":
        result = value + step * ANGLE_STEP
    elif spec.domain is not None:
        result = min(max(value + step * spec.domain.length / INTERVAL_STEPS, spec.domain.start), spec.domain.stop)
    elif spec.kind == "number":
        result = value + step * RATE_STEP
    else:
        result = value

    return result


def _typed(text: str) -> Any:
    """A typed value: a number where it reads as one, else the text itself."""
    try:
        result = float(text)
    except ValueError:
        result = text

    return result


def _worded(error: Exception) -> str:
    """An error as the user reads it: its message, without the machinery around it."""
    if isinstance(error, ValidationError):
        result = error.errors()[0]["msg"].removeprefix("Value error, ")
    elif isinstance(error, KeyError):
        result = str(error.args[0]) if error.args else "not found"
    else:
        result = str(error)

    return result
