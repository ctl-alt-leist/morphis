"""
The stage: the backend of the playground, with no window, panel, or keys.

Dials are values with domains, read and set by path; Motion is the group of
rates that moves an object rigidly, extrinsic or intrinsic, by exact steps;
the Stage holds the space, the tree of entries in it, the selection, the
clock, and the journal every change is recorded in.
"""

from morphis.stage.dial import Dial as Dial, Dials as Dials, DialSpec as DialSpec
from morphis.stage.motion import Motion as Motion, Plane as Plane, planes as planes
from morphis.stage.pose import Pose as Pose
from morphis.stage.stage import Actor as Actor, Group as Group, Notice as Notice, Operation as Operation, Stage as Stage
