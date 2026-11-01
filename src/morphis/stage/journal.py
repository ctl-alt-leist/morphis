"""
Stage - The Journal on Disk

A stage's journal is its whole history, and saving a stage is writing its
journal: one line naming the space, then one line per event, in order, as
JSON. Loading replays the events onto a stage, which gives back the state
exactly, so the journal is the one saved form.

Entries made from the catalog are rebuilt from the catalog. Entries a script
handed in are not data and cannot be written; a journal with them is loaded
onto a stage the same script has set up again, and its first events, the
script's own, are matched against what the script did.
"""

import json
from pathlib import Path

from pydantic import TypeAdapter

from morphis.elements import Metric, metric
from morphis.stage.events import AnyEvent, Event


EVENTS = TypeAdapter(AnyEvent)


def declaration(space: Metric) -> dict:
    """
    Returns the blocks a space is declared by, as plain data.
    """
    result = {
        "time": space.n_time,
        "space": space.n_space,
        "projective": space.n_projective,
        "modes": space.n_modes,
        "convention": space.convention or "mostly minus",
    }

    return result


def write(path: str | Path, space: Metric, events: list[Event]) -> None:
    """
    Write a journal: the space's declaration, then each event, one per line.
    """
    lines = [json.dumps({"space": declaration(space)}), *(event.model_dump_json() for event in events)]
    Path(path).write_text("\n".join(lines) + "\n")


def read(path: str | Path) -> tuple[Metric, list[Event]]:
    """
    Returns the space a journal was written in and its events, in order.
    """
    first, *rest = Path(path).read_text().splitlines()
    space = metric(**json.loads(first)["space"])
    events = [EVENTS.validate_json(line) for line in rest if line.strip()]

    return space, events
