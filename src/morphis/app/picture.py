"""
The picture: the playground in the PyVista window alone, driven from the
keyboard.

The window draws the stage, and its upper left corner holds what a panel
would: the status line, the outline, the dials of the selection with the
highlighted one marked, the latest notice, and the action row, every button
with its key. Everything the panel will do can be done here, so the panel is
a convenience and never a requirement.

The picture decides nothing. It translates the window's keys into the
vocabulary's neutral names and hands them to the controls, draws what the
readout words, and does only what it alone can do: ask for a line of text and
show a picker, both in the corner, and record the window. Its own key
bindings are cleared, so every key means what the vocabulary says; the
primary modifier here is Control, as in a terminal.

    uv run python -m morphis.app.picture            # an empty ℝ⁴
    uv run python -m morphis.app.picture 3          # an empty ℝ³
"""

import sys
from contextlib import AbstractContextManager
from time import perf_counter
from typing import Any

from morphis.app.controls import ASKS, Choice, Controls
from morphis.app.readout import action_row, dial_rows, outline_rows, status_line
from morphis.app.vocabulary import MOD, Command
from morphis.elements import metric
from morphis.stage import Stage
from morphis.visuals import Scene


FRAME_RATE = 30

# The window's names for keys the vocabulary names otherwise
KEYSYMS = {
    "Up": "up",
    "Down": "down",
    "Left": "left",
    "Right": "right",
    "Return": "enter",
    "KP_Enter": "enter",
    "Escape": "escape",
    "Tab": "tab",
    "space": "space",
    "BackSpace": "backspace",
}

RULE = "─" * 24


def key_name(keysym: str, character: str, control: bool, shift: bool) -> str | None:
    """
    Returns a key's neutral name from what the window reports: the modifier
    and a letter, a named key, or the character typed; none for a key alone
    such as Shift.
    """
    if control and len(keysym) == 1 and keysym.isalpha():
        result = MOD + keysym.lower()
    elif keysym in KEYSYMS:
        result = KEYSYMS[keysym]
    elif character:
        result = character
    else:
        result = None

    return result


class Prompt:
    """
    What the picture is asking for: a line of text, or a choice from a picker.

    Attributes:
        command: The command that asked
        title: What is asked for
        choices: The picker's lines, or none for a line of text
        cursor: The picker line highlighted
        text: The line typed so far
    """

    def __init__(self, command: Command, title: str, choices: list[Choice] | None = None) -> None:
        self.command = command
        self.title = title
        self.choices = choices
        self.cursor = 0
        self.text = ""


class Picture:
    """
    The playground's window: a stage, its controls, and the corner they are read in.
    """

    def __init__(self, stage: Stage) -> None:
        self.stage = stage
        self.scene: Scene = stage.scene
        self.controls = Controls(stage, ask=self._ask, record=self._record)
        self.prompt: Prompt | None = None
        self._recording: AbstractContextManager | None = None
        self._shown: str | None = None
        self.scene.on_key(self.press)

    # =========================================================================
    # Keys
    # =========================================================================

    def press(self, keysym: str, character: str = "", control: bool = False, shift: bool = False) -> None:
        """
        A key pressed in the window: to the prompt while one is open, to the
        controls otherwise.
        """
        name = key_name(keysym, character, control, shift)
        if name is not None and self.prompt is not None and self.prompt.choices is not None:
            self._pick(name)
        elif name is not None and self.prompt is not None:
            self._type(name)
        elif name is not None:
            self.controls.key(name)

    def _ask(self, command: Command) -> None:
        if self.controls.picks(command):
            self.prompt = Prompt(command, self.controls.title(command), self.controls.choices(command))
        elif command in ASKS:
            self.prompt = Prompt(command, self.controls.title(command))
            self.prompt.text = self.controls.draft(command)

    def _pick(self, name: str) -> None:
        prompt = self.prompt
        keyed = {key: choice for key, choice, _ in prompt.choices if key}
        if name in ("up", "down"):
            step = 1 if name == "down" else -1
            prompt.cursor = min(max(prompt.cursor + step, 0), len(prompt.choices) - 1)
        elif name == "enter" and prompt.choices:
            self.prompt = None
            self.controls.take(prompt.command, prompt.choices[prompt.cursor][1])
        elif name in keyed:
            self.prompt = None
            self.controls.take(prompt.command, keyed[name])
        elif name == "escape":
            self.prompt = None

    def _type(self, name: str) -> None:
        prompt = self.prompt
        if name == "enter":
            self.prompt = None
            self.controls.answer(prompt.command, prompt.text)
        elif name == "escape":
            self.prompt = None
        elif name == "backspace":
            prompt.text = prompt.text[:-1]
        elif name == "space":
            prompt.text += " "
        elif len(name) == 1:
            prompt.text += name

    # =========================================================================
    # The Corner
    # =========================================================================

    def corner(self) -> str:
        """
        Returns what the corner shows: the status line, the outline, the
        selection's dials, the prompt or the latest notice, and the action row.
        """
        controls = self.controls
        outline = [
            f"{'▸' if selected else ' '} {'  ' * max(depth - 1, 0)}{name}   {kind}"
            for depth, _, name, kind, selected, _ in outline_rows(controls)
        ]
        dials = [f"{'▸' if lit else ' '} {label:<10} {shown}" for _, label, shown, lit in dial_rows(controls)]
        mark = {name: " ◂" if controls.focus == name else "" for name in ("outline", "dials")}
        lines = [status_line(controls), RULE, "outline" + mark["outline"], *outline]
        if dials:
            lines += [RULE, "dials" + mark["dials"], *dials]
        lines += [RULE, self._message(), self._row()]
        result = "\n".join(lines)

        return result

    def _message(self) -> str:
        prompt = self.prompt
        notices = self.stage.notices
        if prompt is not None and prompt.choices is not None:
            listed = [
                f"{'▸' if n == prompt.cursor else ' '} {key} {shown}"
                for n, (key, _, shown) in enumerate(prompt.choices)
            ]
            result = "\n".join([f"{prompt.title}:", *listed])
        elif prompt is not None:
            result = f"{prompt.title}: {prompt.text}▏"
        elif notices:
            last = notices[-1]
            result = ("⚠ " if last.level == "warning" else "") + last.text
        else:
            result = f"{Command.ADD.glyph()} adds to the stage"

        return result

    def _row(self) -> str:
        marks = {"lit": "●", "waiting": "◌", "": ""}
        result = "  ".join(f"{key} {label}{marks[reads]}" for key, label, reads in action_row(self.controls))

        return result

    # =========================================================================
    # Running
    # =========================================================================

    def run(self) -> None:
        """
        Run the playground until the window closes: each frame, advance the
        stage by the wall-clock time since the last, carry it, redraw the
        corner if what it reads has changed, and capture.
        """
        start = last = perf_counter()
        frame = 0
        self._draw_corner()
        self.scene.capture(0.0)
        while not self.scene.is_closed():
            now = perf_counter()
            self.stage.advance(now - last)
            last = now
            self._draw_corner()
            frame += 1
            self.scene.capture(max(frame / FRAME_RATE, now - start))
        self._record(None)

    def _draw_corner(self) -> None:
        shown = self.corner()
        if shown != self._shown:
            self.scene.status(shown)
            self._shown = shown

    def _record(self, path: str | None) -> None:
        """Start recording the window to a path, or stop on none."""
        if path is not None and self._recording is None:
            self._recording = self.scene.record(path)
            self._recording.__enter__()
        elif path is None and self._recording is not None:
            self._recording.__exit__(None, None, None)
            self._recording = None


def run(stage: Stage) -> Picture:
    """
    Run the playground on a stage in its window until the window closes.
    """
    picture = Picture(stage)
    picture.run()

    return picture


def main(arguments: list[str]) -> Any:
    dimension = int(arguments[0]) if arguments else 4
    stage = Stage(metric(space=dimension), Scene())

    return run(stage)


if __name__ == "__main__":
    main(sys.argv[1:])
