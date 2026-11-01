"""
The playground console: a full-screen terminal layout driven by single keys.

A status bar runs along the top, and under it a line naming the screens, the
one shown lit; `g` goes to another. The Stage screen holds three columns: the
outline of the stage on the left, the dials of the selection in the middle,
and on the right the space's own dials over a reading of what the selection
is, as mathematics. Under them run the messages and the action row, every
command with its key.

The console is a view over a stage. Its keys go to the stage's `Controls`,
and every pane is drawn from a `Readings`, so it keeps no copies or counts of
its own and decides nothing. It does only what it alone can do: ask for a
line of text, show a picker, and quit. Beside it, the picture window draws
the stage, and its keys go to the same controls.

    uv run python -m morphis.app.console            # ℝ⁴, with the picture
    uv run python -m morphis.app.console 3          # ℝ³
    uv run python -m morphis.app.console 4 --alone  # no picture window
"""

import asyncio
import sys
from time import perf_counter

from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.layout import ConditionalContainer, Dimension, HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.widgets import Frame

from morphis.app.controls import ASKS, Choice, Controls, Focus
from morphis.app.picture import key_name
from morphis.app.readout import Readings, Readout, Section
from morphis.app.vocabulary import MOD, PLACEHOLDERS, SHIFTED_DIGITS, Command, Screen
from morphis.elements import metric
from morphis.stage import Stage
from morphis.visuals import Scene


Fragments = list[tuple[str, str]]

NOTICE_STYLES = {"info": "", "warning": "ansiyellow"}

# The vocabulary's names for keys that prompt_toolkit names otherwise
TERMINAL_KEYS = {"space": " "}

# How long a lone Escape is held to see whether a sequence follows [s]
ESCAPE_WAIT = 0.05

# A pane that takes whatever height its column has left
FILL = Dimension(weight=1)

# How many messages the messages pane shows
SHOWN_NOTICES = 3

# How often the stage is stepped and the picture drawn [1/s]
FRAME_RATE = 30


def terminal_key(key: str) -> str:
    """
    Returns a vocabulary key as prompt_toolkit names it, the modifier being
    Control in a terminal.
    """
    result = TERMINAL_KEYS.get(key, key.replace(MOD, "c-"))

    return result


class Console:
    """
    The full-screen console over a stage.

    Attributes:
        controls: What every key does
        readings: What every pane is drawn from
        picking: The command whose picker is open, or none
        typing: The command a line is being typed for, or none
    """

    def __init__(self, controls: Controls, readings: Readings | None = None) -> None:
        self.controls = controls
        self.readings = readings if readings is not None else Readout(controls)
        self.picking: Command | None = None
        self.choices: list[Choice] = []
        self.pick = 0
        self.typing: Command | None = None
        self.line = Buffer(multiline=False, accept_handler=self._accept)
        controls._ask = self._ask
        controls.actions[Command.QUIT] = lambda: self.app.exit()
        self.app = self._build()

    @property
    def stage(self) -> Stage:
        return self.controls.stage

    # =========================================================================
    # Layout
    # =========================================================================

    def _build(self) -> Application:
        keys = KeyBindings()
        prompting = Condition(lambda: self.picking is not None or self.typing is not None)
        picking = Condition(lambda: self.picking is not None)
        typing = Condition(lambda: self.typing is not None)

        named = [command.key for command in Command]
        digits = [*"0123456789", *SHIFTED_DIGITS]
        for key in [*named, *digits]:
            keys.add(terminal_key(key), filter=~prompting, eager=key == Command.DONE.key)(
                lambda event, key=key: self.controls.key(key)
            )

        @keys.add("c-c")
        def _quit(event: KeyPressEvent) -> None:
            event.app.exit()

        @keys.add("up", filter=picking)
        @keys.add("down", filter=picking)
        def _move_pick(event: KeyPressEvent) -> None:
            step = 1 if event.key_sequence[0].key == "down" else -1
            self.pick = min(max(self.pick + step, 0), len(self.choices) - 1)

        @keys.add("enter", filter=picking)
        def _take_pick(event: KeyPressEvent) -> None:
            self._take(self.pick)

        @keys.add("<any>", filter=picking)
        def _take_key(event: KeyPressEvent) -> None:
            letters = [key for key, _, _ in self.choices]
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
                width=Dimension(min=30, preferred=44),
            ),
        ])
        placeholder = Frame(Window(), title=lambda: self.controls.screen.label)
        messages = Frame(Window(FormattedTextControl(self._messages), height=SHOWN_NOTICES), title="messages")
        picker = Frame(Window(FormattedTextControl(self._picker)), title=lambda: self._asked(self.picking))
        input_line = Window(
            BufferControl(buffer=self.line),
            height=1,
            get_line_prefix=lambda line, wrap: [("bold", f" {self._asked(self.typing)}: ")],
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
        result.ttimeoutlen = ESCAPE_WAIT

        return result

    def _on(self, *screens: Screen) -> Condition:
        return Condition(lambda: self.controls.screen in screens)

    def _title(self, pane: Focus) -> str:
        selected = self.stage.selected
        names = {"outline": "outline", "dials": selected.name if selected is not None else "dials", "space": "space"}
        result = f"▸ {names[pane]}" if pane == self.controls.focus else names[pane]

        return result

    # =========================================================================
    # Panes
    # =========================================================================

    def _status_bar(self) -> FormattedText:
        return FormattedText([("", " " + "   ".join(self.readings.status()))])

    def _screens(self) -> FormattedText:
        fragments: Fragments = [("", " ")]
        for screen in Screen:
            style = "reverse bold" if screen is self.controls.screen else "ansibrightblack"
            fragments += [(style, f" {screen.label} "), ("", " ")]
        fragments.append(("ansibrightblack", f"   {Command.GO.glyph()} go to"))

        return FormattedText(fragments)

    def _outline(self) -> FormattedText:
        lines = []
        for depth, name, kind, holds, selected in self.readings.outline():
            mark = "▾ " if holds else "  "
            style = self._lit("outline", selected)
            width = max(16 - 2 * depth, len(name) + 1)
            lines.append([(style, f" {'  ' * depth}{mark}{name:<{width}}"), (style or "ansibrightblack", f"{kind} ")])
        if not lines:
            lines.append([("ansibrightblack", f" {Command.ADD.glyph()} adds to the stage")])

        return _joined(lines)

    def _dials(self) -> FormattedText:
        return self._sections("dials", self.readings.dials())

    def _space(self) -> FormattedText:
        return self._sections("space", self.readings.space())

    def _sections(self, pane: Focus, sections: list[Section]) -> FormattedText:
        lines = []
        for title, rows in sections:
            lines.append([("ansibrightblack", f" {title}")])
            for name, value, lit in rows:
                lines.append([(self._lit(pane, lit), f"   {name:<12}{value} ")])

        return _joined(lines)

    def _about(self) -> FormattedText:
        return _joined([[("", f" {line}")] for line in self.readings.about()])

    def _messages(self) -> FormattedText:
        notices = self.readings.notices()[-SHOWN_NOTICES:]

        return _joined([[(NOTICE_STYLES[level], f" {text}")] for text, level in notices])

    def _action_row(self) -> FormattedText:
        marks = {"lit": "reverse", "waiting": "ansiyellow reverse", "": ""}
        fragments: Fragments = [("", " ")]
        for key, label, reads in self.readings.actions():
            fragments += [("bold", f" {key} "), (marks[reads], label), ("", " ")]
        hints = f"   {Command.FOCUS.glyph()} pane  ↑↓ move  ←→ turn  ↩ take  / find  q quit"
        fragments.append(("ansibrightblack", hints))

        return FormattedText(fragments)

    def _picker(self) -> FormattedText:
        lines = [
            [("bold", f" {key} "), ("reverse" if n == self.pick else "", f" {shown} ")]
            for n, (key, _, shown) in enumerate(self.choices)
        ]

        return _joined(lines)

    def _lit(self, pane: Focus, lit: bool) -> str:
        if not lit:
            result = ""
        elif pane == self.controls.focus:
            result = "reverse"
        else:
            result = "bold"

        return result

    # =========================================================================
    # What Only the Console Can Do
    # =========================================================================

    def _ask(self, command: Command) -> None:
        if self.controls.picks(command):
            self.picking, self.choices, self.pick = command, self.controls.choices(command), 0
        elif command in ASKS:
            self.typing = command
            self.line.set_document(Document(self.controls.draft(command)))

    def _asked(self, command: Command | None) -> str:
        return "" if command is None else self.controls.title(command)

    def _take(self, n: int) -> None:
        command, choice = self.picking, self.choices[n][1]
        self.picking = None
        self.controls.take(command, choice)

    def _accept(self, buffer: Buffer) -> bool:
        command, text = self.typing, buffer.text
        self.typing = None
        self.controls.answer(command, text)

        return False

    # =========================================================================
    # Running
    # =========================================================================

    async def run(self) -> None:
        """
        Run the console until it quits, stepping the stage and drawing its
        picture, if it has one, at the frame rate meanwhile.
        """
        ticking = asyncio.ensure_future(self._tick())
        try:
            await self.app.run_async()
        finally:
            ticking.cancel()

    async def _tick(self) -> None:
        start = last = perf_counter()
        scene = self.stage.scene
        while True:
            now = perf_counter()
            self.stage.advance(now - last)
            last = now
            if scene is not None:
                scene.capture(now - start)
                scene.process_events()
            self.app.invalidate()
            await asyncio.sleep(1.0 / FRAME_RATE)


def _joined(lines: list[Fragments]) -> FormattedText:
    """Lines of fragments as one formatted text, a line break between each."""
    fragments: Fragments = []
    for n, line in enumerate(lines):
        fragments += line
        if n < len(lines) - 1:
            fragments.append(("", "\n"))

    return FormattedText(fragments)


def main(arguments: list[str]) -> None:
    dimension = int(next((argument for argument in arguments if argument.isdigit()), 4))
    scene = None if "--alone" in arguments else Scene()
    controls = Controls(Stage(metric(space=dimension), scene))
    # The clock runs from the start, so a turned dial shows at once; nothing moves until one is
    controls.stage.play()
    if scene is not None:
        scene.on_key(lambda keysym, character, control, shift: _picture_key(controls, keysym, character, control))
    asyncio.run(Console(controls).run())
    if scene is not None:
        scene.close()


def _picture_key(controls: Controls, keysym: str, character: str, control: bool) -> None:
    """A key pressed in the picture window, to the same controls as the console's."""
    name = key_name(keysym, character, control, False)
    if name is not None:
        controls.key(name)


if __name__ == "__main__":
    main(sys.argv[1:])
