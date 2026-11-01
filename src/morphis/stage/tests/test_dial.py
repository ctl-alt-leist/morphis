"""
Unit tests for dials: domains, descriptions, and paths.
"""

from typing import Literal

import pytest
from numpy import pi
from pydantic import ValidationError

from morphis.manifold import Circle, Interval
from morphis.stage import Dial, Dials


class Look(Dials):
    opacity: float = Dial(1.0, Interval(0.0, 1.0))
    visible: bool = True
    style: Literal["solid", "sketch"] = "solid"


class Torus(Dials):
    major: float = Dial(1.0, Interval(0.2, 3.0))
    phase: float = Dial(0.0, Circle())
    spin: float = Dial(0.0, rate=True)
    look: Look = Look()


class TestDomains:
    def test_refused_outside_with_what_is_allowed(self):
        torus = Torus()
        with pytest.raises(ValidationError, match="a number from 0.2 to 3"):
            torus.major = 5.0
        assert torus.major == 1.0

    def test_refused_on_construction(self):
        with pytest.raises(ValidationError):
            Torus(major=0.1)

    def test_angle_wraps_onto_the_circle(self):
        torus = Torus()
        torus.phase = 2 * pi + 0.5
        assert torus.phase == pytest.approx(0.5)
        torus.phase = -0.5
        assert torus.phase == pytest.approx(2 * pi - 0.5)

    def test_unbounded(self):
        torus = Torus()
        torus.spin = -1e6
        assert torus.spin == -1e6


class TestDescribe:
    def test_paths_in_declaration_order(self):
        paths = [spec.path for spec in Torus().describe()]
        assert paths == ["major", "phase", "spin", "look.opacity", "look.visible", "look.style"]

    def test_kinds(self):
        specs = {spec.path: spec for spec in Torus().describe()}
        assert specs["major"].kind == "number"
        assert specs["major"].domain == Interval(0.2, 3.0)
        assert specs["phase"].kind == "angle"
        assert specs["look.visible"].kind == "toggle"
        assert specs["look.style"].kind == "choice"
        assert specs["look.style"].choices == ("solid", "sketch")

    def test_rate(self):
        specs = {spec.path: spec for spec in Torus().describe()}
        assert specs["spin"].rate
        assert not specs["major"].rate


class TestPaths:
    def test_get_and_set(self):
        torus = Torus()
        torus.set("look.opacity", 0.4)
        assert torus.get("look.opacity") == 0.4
        assert torus.look.opacity == 0.4

    def test_set_is_checked(self):
        with pytest.raises(ValidationError):
            Torus().set("look.opacity", 2.0)

    def test_unknown_path(self):
        torus = Torus()
        with pytest.raises(KeyError):
            torus.get("minor")
        with pytest.raises(KeyError):
            torus.set("shape.minor", 1.0)

    def test_values(self):
        values = Torus().values()
        assert values["major"] == 1.0
        assert values["look.style"] == "solid"
        assert len(values) == 6
