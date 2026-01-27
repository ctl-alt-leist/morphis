"""
Geometric Algebra - Elements

Core geometric algebra objects: Vectors, MultiVectors, Frames, and Metrics.
"""

from morphis.elements.base import (
    CompositeElement as CompositeElement,
    Element as Element,
    GradedElement as GradedElement,
)
from morphis.elements.frame import Frame
from morphis.elements.metric import (
    PGA as PGA,
    STA as STA,
    GASignature as GASignature,
    GAStructure as GAStructure,
    Metric as Metric,
    degenerate_ns as degenerate_ns,
    euclidean_metric as euclidean_metric,
    euclidean_ns as euclidean_ns,
    lorentzian_metric as lorentzian_metric,
    lorentzian_ns as lorentzian_ns,
    metric as metric,
    pga_metric as pga_metric,
)
from morphis.elements.multivector import MultiVector
from morphis.elements.protocols import Graded as Graded, Spanning as Spanning, Transformable as Transformable
from morphis.elements.tensor import Tensor
from morphis.elements.vector import (
    Vector,
    basis_element as basis_element,
    basis_vector as basis_vector,
    basis_vectors as basis_vectors,
    geometric_basis as geometric_basis,
    pseudoscalar as pseudoscalar,
)


# Rebuild models to resolve forward references
Vector.model_rebuild()
MultiVector.model_rebuild()
Frame.model_rebuild()
Tensor.model_rebuild()
