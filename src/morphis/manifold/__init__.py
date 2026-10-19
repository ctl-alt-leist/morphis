"""
Manifolds

Submanifolds of flat spaces of any dimension and signature, given by
parametrized patches over coordinate domains.
"""

from morphis.manifold.atlas import Atlas as Atlas
from morphis.manifold.domain import (
    Circle as Circle,
    ComplexRegion as ComplexRegion,
    Domain as Domain,
    Interval as Interval,
)
from morphis.manifold.mesh import Mesh as Mesh
from morphis.manifold.patch import Patch as Patch
