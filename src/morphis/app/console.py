"""
The playground console: a full-screen terminal layout driven by single keys.

A status bar runs along the top, and under it a line naming the screens, the
one shown lit; `g` goes to another. The Stage screen holds three columns: the
outline of the stage on the left, the dials of the selected entry in the
middle, and on the right the space's own dials over a reading of what the
selection is, as mathematics. Under them run the messages and the action row,
every command with its key.

The console is a view over a `Readings`, and every pane is drawn from what it
reads, so the console keeps no copies of its own. For now it reads a sketch,
and the commands that would change a stage only say so; moving about the
layout (the panes, the cursors, the screens, the pickers, a typed line) works
in full, since that is the console's own.

    uv run python -m morphis.app.console
"""

from typing import Literal

from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.layout import ConditionalContainer, Dimension, HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.widgets import Frame

from morphis.app.sketch import Readings, Section, Sketch
from morphis.app.vocabulary import ACTION_ROW, MOD, PLACEHOLDERS, Command, Screen


Pane = Literal["outline", "dials", "space"]
Fragments = list[tuple[str, str]]

PANES: tuple[Pane, ...] = ("outline", "dials", "space")

NOTICE_STYLES = {"info": "", "warning": "ansiyellow"}

# The vocabulary's names for keys that prompt_toolkit names otherwise
TERMINAL_KEYS = {"space": " "}

# How long a lone Escape is held to see whether a sequence follows [s]
ESCAPE_WAIT = 0.05

# A pane that takes whatever height its column has left
FILL = Dimension(weight=1)

# How many messages the messages pane shows
SHOWN_NOTICES = 3

# The commands the console carries out itself: moving about its own layout
OWN = (Command.PREVIOUS, Command.NEXT, Command.FOCUS, Command.TAKE, Command.FIND, Command.GO, Command.ADD, Command.DONE)


def terminal_key(key: str) -> str:
    """
    Returns a vocabulary key as prompt_toolkit names it, the modifier being
    Control in a terminal.
    """
    result = TERMINAL_KEYS.get(key, key.replace(MOD, "c-"))

    return result


class Console:
    """
    The full-screen console over a reading of a stage.

    Attributes:
        readings: What every pane is drawn from
        screen: The screen shown
        focus: The pane the arrows move in
        cursor: The highlighted row of each pane
        filter: The text the outline is filtered by
        picking: The command whose picker is open, or none
        typing: The command a line is being typed for, or none
    """

    def __init__(self, readings: Readings) -> None:
        self.readings = readings
        self.screen = Screen.STAGE
        self.focus: Pane = "outline"
        self.cursor: dict[Pane, int] = dict.fromkeys(PANES, 0)
        self.filter = ""
        self.picking: Command | None = None
        self.pick = 0
        self.typing: Command | None = None
        self.said: list[tuple[str, str]] = []
        self.line = Buffer(multiline=False, accept_handler=self._accept)
        self.app = self._build()

    # =========================================================================
    # Layout
    # =========================================================================

    def _build(self) -> Application:
        keys = KeyBindings()
        prompting = Condition(lambda: self.picking is not None or self.typing is not None)
        picking = Condition(lambda: self.picking is not None)
        typing = Condition(lambda: self.typing is not None)

        for command in Command:
            keys.add(terminal_key(command.key), filter=~prompting, eager=command is Command.DONE)(
                lambda event, command=command: self.press(command)
            )

        @keys.add("c-c")
        def _quit(event: KeyPressEvent) -> None:
            event.app.exit()

        @keys.add("up", filter=picking)
        @keys.add("down", filter=picking)
        def _move_pick(event: KeyPressEvent) -> None:
            step = 1 if event.key_sequence[0].key == "down" else -1
            self.pick = min(max(self.pick + step, 0), len(self._choices()) - 1)

        @keys.add("enter", filter=picking)
        def _take_pick(event: KeyPressEvent) -> None:
            self._take(self.pick)

        @keys.add("<any>", filter=picking)
        def _take_key(event: KeyPressEvent) -> None:
            letters = [key for key, _ in self._choices()]
            pressed = event.key_sequence[0].key
            if pressed in letters:
                self._take(letters.index(pressed))

        @keys.add("escape", filter=prompting, eager=True)
        def _cancel(event: KeyPressEvent) -> None:
            self.picking = self.typing = None
            self.line.reset()

        outline = Window(FormattedTextControl(self._outline), width=Dimension(min=24, preferred=30), height=FILL)
        dials = Window(FormattedTextControl(self._dials), height=FILL)
        space = Window(FormattedTextControl(self._space))
        about = Window(FormattedTextControl(self._about), wrap_lines=True, height=FILL)
        stage_screen = VSplit([
            Frame(outline, title=lambda: self._title("outline")),
            Frame(dials, title=lambda: self._title("dials")),
            HSplit(
                [Frame(space, title=lambda: self._title("space")), Frame(about, title="about")],
                width=Dimension(min=30, preferred=40),
            ),
        ])
        placeholder = Frame(Window(), title=lambda: self.screen.label)
        messages = Frame(Window(FormattedTextControl(self._messages), height=SHOWN_NOTICES), title="messages")
        picker = Frame(Window(FormattedTextControl(self._picker)), title=lambda: self._picker_title())
        input_line = Window(
            BufferControl(buffer=self.line),
            height=1,
            get_line_prefix=lambda line, wrap: [("bold", f" {self._typing_title()}: ")],
        )
        actions = Window(FormattedTextControl(self._action_row), height=1)

        root = HSplit([
            Window(FormattedTextControl(self._status_bar), height=1, style="reverse"),
            Window(FormattedTextControl(self._screens), height=1),
            ConditionalContainer(stage_screen, filter=self._on(Screen.STAGE)),
            ConditionalContainer(placeholder, filter=self._on(*PLACEHOLDERS)),
            ConditionalContainer(picker, filter=picking),
            messages,
            ConditionalContainer(input_line, filter=typing),
            ConditionalContainer(actions, filter=~typing),
        ])
        result = Application(layout=Layout(root, focused_element=input_line), key_bindings=keys, full_screen=True)
        # Escape alone acts at once, rather than waiting to see whether it starts a longer sequence
        result.ttimeoutlen = ESCAPE_WAIT

        return result

    def _on(self, *screens: Screen) -> Condition:
        return Condition(lambda: self.screen in screens)

    def _title(self, pane: Pane) -> str:
        names = {"outline": "outline", "dials": self.selected or "dials", "space": "space"}
        result = f"▸ {names[pane]}" if pane == self.focus else names[pane]

        return result

    # =========================================================================
    # Panes
    # =========================================================================

    def _status_bar(self) -> FormattedText:
        parts = self.readings.status(self.selected)
        result = FormattedText([("", " " + "   ".join(parts))])

        return result

    def _screens(self) -> FormattedText:
        fragments: Fragments = [("", " ")]
        for screen in Screen:
            style = "reverse bold" if screen is self.screen else "ansibrightblack"
            fragments += [(style, f" {screen.label} "), ("", " ")]
        fragments.append(("ansibrightblack", f"   {Command.GO.glyph()} go to"))

        return FormattedText(fragments)

    def _outline(self) -> FormattedText:
        lines = []
        for n, (depth, name, kind, holds) in enumerate(self._listed()):
            mark = "▾ " if holds else "  "
            text = f"{'  ' * depth}{mark}{name}"
            style = self._row_style("outline", n)
            lines.append([(style, f" {text:<18}"), ("ansibrightblack" if not style else style, f"{kind} ")])

        return _joined(lines)

    def _dials(self) -> FormattedText:
        sections = self.readings.dials(self.selected) if self.selected else []

        return self._sections("dials", sections)

    def _space(self) -> FormattedText:
        return self._sections("space", self.readings.space())

    def _sections(self, pane: Pane, sections: list[Section]) -> FormattedText:
        lines = []
        n = 0
        for title, rows in sections:
            lines.append([("ansibrightblack", f" {title}")])
            for name, value in rows:
                style = self._row_style(pane, n)
                lines.append([(style, f"   {name:<12}"), (style, f"{value} ")])
                n += 1

        return _joined(lines)

    def _about(self) -> FormattedText:
        lines = [[("", f" {line}")] for line in (self.readings.about(self.selected) if self.selected else [])]

        return _joined(lines)

    def _messages(self) -> FormattedText:
        notices = [*self.readings.notices(), *self.said][-SHOWN_NOTICES:]
        lines = [[(NOTICE_STYLES[level], f" {text}")] for text, level in notices]

        return _joined(lines)

    def _action_row(self) -> FormattedText:
        fragments: Fragments = [("", " ")]
        for command in ACTION_ROW:
            fragments += [("bold", f" {command.glyph()} "), ("", f"{command.label} ")]
        fragments.append(("ansibrightblack", f"    {Command.FOCUS.glyph()} pane  ↑↓ move  ↩ take  / find  q quit"))

        return FormattedText(fragments)

    def _picker(self) -> FormattedText:
        lines = []
        for n, (key, shown) in enumerate(self._choices()):
            style = "reverse" if n == self.pick else ""
            lines.append([("bold", f" {key} "), (style, f" {shown} ")])

        return _joined(lines)

    def _picker_title(self) -> str:
        return {Command.ADD: "add", Command.GO: "go to"}.get(self.picking, "")

    def _typing_title(self) -> str:
        return {Command.FIND: "find", Command.TAKE: "value"}.get(self.typing, "")

    def _row_style(self, pane: Pane, n: int) -> str:
        if n != self.cursor[pane]:
            result = ""
        elif pane == self.focus:
            result = "reverse"
        else:
            result = "bold"

        return result

    # =========================================================================
    # Moving About
    # =========================================================================

    @property
    def selected(self) -> str | None:
        """The name of the entry under the outline's cursor."""
        listed = self._listed()
        n = self.cursor["outline"]
        result = listed[n][1] if 0 <= n < len(listed) else None

        return result

    def _listed(self) -> list:
        found = self.filter.lower()

        return [row for row in self.readings.outline() if found in row[1].lower() or found in row[2].lower()]

    def _rows(self, pane: Pane) -> int:
        """How many rows the cursor moves over in a pane."""
        if pane == "outline":
            result = len(self._listed())
        else:
            sections = self.readings.space() if pane == "space" else self.readings.dials(self.selected or "")
            result = sum(len(rows) for _, rows in sections)

        return result

    def press(self, command: Command) -> None:
        """
        A command's key: the console's own commands move about the layout;
        the rest would act on a stage, and say that none is connected yet.
        """
        if command is Command.QUIT:
            self.app.exit()
        elif command in OWN:
            self._own(command)
        else:
            self.said.append((f"{command.label.lower()}: not connected to a stage yet", "warning"))

    def _own(self, command: Command) -> None:
        pane = self.focus
        if command in (Command.PREVIOUS, Command.NEXT):
            step = 1 if command is Command.NEXT else -1
            self.cursor[pane] = min(max(self.cursor[pane] + step, 0), max(self._rows(pane) - 1, 0))
            if pane == "outline":
                self.cursor["dials"] = 0
        elif command is Command.FOCUS:
            self.focus = PANES[(PANES.index(pane) + 1) % len(PANES)]
        elif command is Command.TAKE and pane == "outline":
            self.focus = "dials"
        elif command in (Command.TAKE, Command.FIND):
            self.typing = command
        elif command in (Command.GO, Command.ADD):
            self.picking, self.pick = command, 0
        elif command is Command.DONE:
            self._back()

    def _back(self) -> None:
        if self.screen is not Screen.STAGE:
            self.screen = Screen.STAGE
        elif self.focus != "outline":
            self.focus = "outline"
        elif self.filter:
            self.filter = ""

    def _choices(self) -> list[tuple[str, str]]:
        if self.picking is Command.GO:
            result = [(screen.key, screen.label) for screen in Screen]
        elif self.picking is Command.ADD:
            lines = [line for _, listed in self.readings.catalog() for line in listed]
            result = list(zip("abcdefghijklmnopqrstuvwxyz", lines, strict=False))
        else:
            result = []

        return result

    def _take(self, n: int) -> None:
        choices, command = self._choices(), self.picking
        self.picking = None
        if command is Command.GO:
            self.screen = list(Screen)[n]
        elif command is Command.ADD:
            self.said.append((f"add {choices[n][1].lower()}: not connected to a stage yet", "warning"))

    def _accept(self, buffer: Buffer) -> bool:
        text, command = buffer.text.strip(), self.typing
        self.typing = None
        if command is Command.FIND:
            self.filter = text
            self.cursor["outline"] = 0
        elif command is Command.TAKE:
            self.said.append((f"{text}: not connected to a stage yet", "warning"))

        return False

    def run(self) -> None:
        self.app.run()


def _joined(lines: list[Fragments]) -> FormattedText:
    """Lines of fragments as one formatted text, a line break between each."""
    fragments: Fragments = []
    for n, line in enumerate(lines):
        fragments += line
        if n < len(lines) - 1:
            fragments.append(("", "\n"))

    return FormattedText(fragments)


def main() -> None:
    Console(Sketch()).run()


if __name__ == "__main__":
    main()
