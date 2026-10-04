"""
Ink Sketches

Pen-and-ink conceptual figures: an organic enclosing space, stippled for depth,
with vectors, planes, curves, and labels drawn inside it. Objects keep their
true dimension and relationships; a Depiction states how the figure chooses to
draw them in three dimensions, and a Camera sets the view.
"""

from morphis.visuals.ink.camera import Camera as Camera
from morphis.visuals.ink.depiction import Depiction as Depiction
from morphis.visuals.ink.sketch import Sketch as Sketch
from morphis.visuals.ink.space import OrganicSpace as OrganicSpace
from morphis.visuals.ink.theme import (
    CHALKBOARD as CHALKBOARD,
    INK as INK,
    PARCHMENT as PARCHMENT,
    InkTheme as InkTheme,
    get_ink_theme as get_ink_theme,
)
