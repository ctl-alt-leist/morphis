"""
The key script, driven through each front end.
"""

from morphis.app.picture import Picture
from morphis.app.tests.script import NAMED, script
from morphis.elements import metric
from morphis.stage import Stage
from morphis.stage.journal import read
from morphis.visuals import Scene


# How the picture's window reports each named key: its name, and Control held
WINDOW = {
    "up": ("Up", False),
    "down": ("Down", False),
    "left": ("Left", False),
    "right": ("Right", False),
    "enter": ("Return", False),
    "escape": ("Escape", False),
    "tab": ("Tab", False),
    "space": ("space", False),
    "mod-s": ("s", True),
}


def drive_picture(picture: Picture, keys: str) -> None:
    for character in keys:
        if character in NAMED:
            keysym, control = WINDOW[NAMED[character]]
            picture.press(keysym, "" if control else character, control)
        else:
            picture.press(character, character)


def test_picture(tmp_path):
    journal = str(tmp_path / "session.jsonl")
    picture = Picture(Stage(metric(space=4), Scene(window=False)))
    for n, (keys, run, check) in enumerate(script(journal)):
        drive_picture(picture, keys)
        if run:
            picture.stage.advance(run / picture.stage.space.get("clock.speed"))
        assert check is None or check(picture.controls), f"step {n}: {keys!r}"

    space, events = read(journal)
    assert space == picture.stage.metric
    assert len(events) == len(picture.stage.journal)
