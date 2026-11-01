"""
The console, driven through a pipe: its keys move about its own layout, and
the commands that would act on a stage say that none is connected.
"""

import asyncio

import pytest


pytest.importorskip("prompt_toolkit")

from prompt_toolkit.application import create_app_session  # noqa: E402
from prompt_toolkit.input import create_pipe_input  # noqa: E402
from prompt_toolkit.output import DummyOutput  # noqa: E402

from morphis.app.console import Console  # noqa: E402
from morphis.app.sketch import Sketch  # noqa: E402
from morphis.app.vocabulary import Command, Screen  # noqa: E402


DOWN, ESCAPE, TAB = "\x1b[B", "\x1b", "\t"


def drive(*steps: str) -> Console:
    """Run the console, send each step's keys in turn, and quit."""

    async def run() -> Console:
        with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
            console = Console(Sketch())
            task = asyncio.ensure_future(console.app.run_async())
            await asyncio.sleep(0.1)
            for keys in steps:
                pipe.send_text(keys)
                await asyncio.sleep(0.1)
            pipe.send_text("q")
            await task

        return console

    return asyncio.run(run())


def test_outline_and_panes():
    console = drive(DOWN, TAB, DOWN + DOWN)
    assert console.selected == "torus"
    assert console.focus == "dials"
    assert console.cursor["dials"] == 2


def test_screens():
    console = drive("g", "j")
    assert console.screen is Screen.JOURNAL
    console = drive("g", "j", ESCAPE)
    assert console.screen is Screen.STAGE


def test_find():
    console = drive("/", "cube\r")
    assert console.filter == "cube"
    assert console.selected == "cube"


def test_not_connected():
    console = drive("a", "f", "s")
    assert console.picking is None
    assert console.said == [
        ("add torus: not connected to a stage yet", "warning"),
        ("stop: not connected to a stage yet", "warning"),
    ]
    assert Command.ADD.key == "a"
