"""
The stage: the backend of the playground, with no window, panel, or keys.

Dials are values with domains, read and set by path; Motion is the group of
rates that moves an object rigidly, extrinsic or intrinsic, by exact steps.
"""

from morphis.stage.dial import Dial as Dial, Dials as Dials, DialSpec as DialSpec
from morphis.stage.motion import Motion as Motion, Plane as Plane, Pose as Pose, planes as planes
