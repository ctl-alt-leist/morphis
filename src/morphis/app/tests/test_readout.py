"""
Unit tests for the readout: how the playground is worded.
"""

from math import pi

from morphis.app.controls import Controls
from morphis.app.readout import action_row, dial_rows, outline_rows, status_line
from morphis.app.vocabulary import Command
from morphis.elements import metric
from morphis.stage import Stage


def controls_with_a_torus():
    controls = Controls(Stage(metric(space=4)))
    controls.take(Command.ADD, "Torus")

    return controls


class TestStatus:
    def test_reads_left_to_right(self):
        controls = controls_with_a_torus()
        assert status_line(controls) == "❚❚ t = 0.00   orthographic, κ = 0.00   torus"

    def test_marks(self):
        controls = controls_with_a_torus()
        controls.stage.set("space", "view.eye", 1 / 3)
        controls.stage.set("space", "clock.speed", 2.0)
        controls.key("space")
        controls.key("1")
        line = status_line(controls)
        assert line.startswith("▶ t = 0.00  ×2")
        assert "perspective, κ = 0.33" in line
        assert line.endswith("e₁·")
        controls.key("escape")
        controls.key("r")
        assert status_line(controls).endswith("r again to reset")


class TestRows:
    def test_outline(self):
        controls = controls_with_a_torus()
        controls.take(Command.ADD, "Group")
        rows = outline_rows(controls)
        assert [(depth, name, kind) for depth, _, name, kind, _, _ in rows] == [
            (0, "space", "Space"),
            (1, "torus", "Mesh"),
            (1, "group", "Group"),
        ]
        assert rows[2][4] and rows[2][5]

    def test_dials_by_their_letters(self):
        controls = controls_with_a_torus()
        controls.key("1")
        controls.key("3")
        rows = {path: (name, shown, lit) for path, name, shown, lit in dial_rows(controls)}
        assert rows["motion.rate.1,3"] == ("e₁₃", "0.00", True)
        assert rows["motion.velocity.2"][0] == "v₂"
        assert rows["look.visible"][:2] == ("shown", "on")
        assert rows["shape.major"][0] == "major"

    def test_space_dials(self):
        controls = controls_with_a_torus()
        controls.stage.set("space", "view.turn.3,4", pi / 2)
        controls.stage.select("space")
        rows = {path: (name, shown) for path, name, shown, _ in dial_rows(controls)}
        assert rows["view.eye"] == ("κ", "0.00")
        assert rows["view.turn.3,4"] == ("θ₃₄", "0.50π")
        assert rows["view.shear.x"][0] == "shear x"

    def test_action_row(self):
        controls = controls_with_a_torus()
        controls.key("h")
        row = {label: (key, reads) for key, label, reads in action_row(controls)}
        assert row["Play"] == ("␣", "")
        assert row["Hide"] == ("h", "lit")
        controls.take(Command.ADD, "Rotation")
        assert "On" in {label for _, label, _ in action_row(controls)}
