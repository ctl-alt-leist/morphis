"""
The console, driven through a pipe over a stage with no picture: its keys go
to the controls, and its panes read the stage.
"""

import asyncio

import pytest


pytest.importorskip("prompt_toolkit")

from prompt_toolkit.application import create_app_session  # noqa: E402
from prompt_toolkit.input import create_pipe_input  # noqa: E402
from prompt_toolkit.output import DummyOutput  # noqa: E402

from morphis.app.console import Console  # noqa: E402
from morphis.app.controls import Controls  # noqa: E402
from morphis.app.vocabulary import Screen  # noqa: E402
from morphis.elements import metric  # noqa: E402
from morphis.stage import Stage  # noqa: E402


DOWN, RIGHT, ESCAPE, TAB, ENTER = "\x1b[B", "\x1b[C", "\x1b", "\t", "\r"


def drive(*steps: str) -> Console:
    """Run the console over an empty ℝ⁴, send each step's keys in turn, and quit."""

    async def run() -> Console:
        with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
            console = Console(Controls(Stage(metric(space=4))))
            task = asyncio.ensure_future(console.run())
            await asyncio.sleep(0.1)
            for keys in steps:
                pipe.send_text(keys)
                await asyncio.sleep(0.1)
            pipe.send_text("q")
            await task

        return console

    return asyncio.run(run())


def test_add_and_turn():
    console = drive("a", "f", "1", "3", RIGHT + RIGHT)
    assert console.stage.selected.name == "torus"
    assert console.stage.selected.get("motion.rate.1,3") == pytest.approx(0.2)
    assert console.controls.focus == "dials"
    assert console.readings.about()[0] == "Clifford torus  S¹ × S¹"


def test_space_pane_and_typed_value():
    console = drive(TAB + TAB, DOWN, ENTER, "0.5" + ENTER)
    assert console.stage.space.get("clock.speed") == 0.5


def test_screens():
    console = drive("g", "j")
    assert console.controls.screen is Screen.JOURNAL
    console = drive("g", "j", ESCAPE)
    assert console.controls.screen is Screen.STAGE


def test_play_runs_the_clock():
    console = drive(" ", "")
    assert console.stage.playing
    assert console.stage.time > 0.0


def test_disabled_says_why():
    console = drive("s")
    assert console.readings.notices()[-1] == ("select an entry that moves to stop it", "warning")
