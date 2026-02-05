# Morphis Package API Reference

*Auto-generated documentation*

---

## Elements

Core geometric algebra objects.

### `morphis.elements.base`

*Geometric Algebra - Element Base Classes*

```python
class IndexableMixin:
    """Mixin providing index notation dispatch for tensor contraction."""

```


```python
class Element(BaseModel):
    """Base class for all geometric algebra elements."""

    @property
    def collection(self): ...

    @property
    def dim(self): ...

    def apply_similarity(self, S: "'MultiVector'", t: "'Vector | None'" = None) -> 'Self'
        """Apply a similarity transformation to this element."""

    def __init__(self, /, **data: 'Any') -> 'None'
        """Create a new model by parsing and validating input data from keyword arguments."""
```


```python
class GradedElement(Element):
    """Base class for elements with a single grade and array data."""

    @property
    def shape(self): ...

    @property
    def ndim(self): ...

    @property
    def collection(self): ...

    @property
    def dim(self): ...

    def with_metric(self, metric: 'Metric') -> 'Self'
        """Return a new element with the specified metric context."""

    def __init__(self, /, **data: 'Any') -> 'None'
        """Create a new model by parsing and validating input data from keyword arguments."""

    def apply_similarity(self, S: "'MultiVector'", t: "'Vector | None'" = None) -> 'Self'
        """Apply a similarity transformation to this element."""
```


```python
class CompositeElement(Element):
    """Base class for elements composed of multiple grades."""

    @property
    def grades(self): ...

    @property
    def collection(self): ...

    @property
    def dim(self): ...

    def with_metric(self, metric: 'Metric') -> 'Self'
        """Return a new element with the specified metric context."""

    def __init__(self, /, **data: 'Any') -> 'None'
        """Create a new model by parsing and validating input data from keyword arguments."""

    def apply_similarity(self, S: "'MultiVector'", t: "'Vector | None'" = None) -> 'Self'
        """Apply a similarity transformation to this element."""
```


### `morphis.elements.metric`

*Geometric Algebra - Metric and Context*

```python
class GASignature(Enum):
    """Metric signature: the pattern of eigenvalues."""

```


```python
class GAStructure(Enum):
    """Geometric structure: the interpretation of GA elements."""

```


```python
class Metric(BaseModel):
    """Complete geometric context: metric tensor + signature + structure."""

    @property
    def dim(self): ...

    @property
    def euclidean_dim(self): ...

    @property
    def signature_tuple(self): ...

    def is_compatible(self, other: 'Metric') -> 'bool'
        """Check if two metrics are compatible for operations."""

    def __init__(self, /, **data: 'Any') -> 'None'
        """Create a new model by parsing and validating input data from keyword arguments."""
```

**Functions:**

```python
def metric(dim: 'int', signature: 'str' = 'euclidean', structure: 'str' = 'flat') -> 'Metric'
    """Create a metric with specified signature and structure."""

def euclidean_metric(dim: 'int') -> 'Metric'
    """Get cached Euclidean metric for d-dimensional space."""

def pga_metric(dim: 'int') -> 'Metric'
    """Get cached PGA metric for d-dimensional Euclidean space."""

def lorentzian_metric(dim: 'int') -> 'Metric'
    """Get cached Lorentzian (spacetime) metric."""
```


### `morphis.elements.tensor`

*Geometric Algebra - Tensor Base Class*

```python
class Tensor(Element):
    """A general (p,q)-tensor in geometric algebra."""

    @property
    def rank(self): ...

    @property
    def total_rank(self): ...

    @property
    def geo(self): ...

    @property
    def contravariant_shape(self): ...

    @property
    def covariant_shape(self): ...

    @property
    def shape(self): ...

    @property
    def ndim(self): ...

    @property
    def collection(self): ...

    @property
    def geometric_shape(self): ...

    @property
    def dim(self): ...

    def with_metric(self, metric: 'Metric') -> 'Self'
        """Return a new tensor with the specified metric context."""

    def __init__(self, /, **data: 'Any') -> 'None'
        """Create a new model by parsing and validating input data from keyword arguments."""

    def apply_similarity(self, S: "'MultiVector'", t: "'Vector | None'" = None) -> 'Self'
        """Apply a similarity transformation to this element."""
```


### `morphis.elements.vector`

*Geometric Algebra - Vector*

```python
class AtAccessor:
    """Accessor for lot-only (collection) slicing on a Vector."""

    def __init__(self, vector: "'Vector'")
        """Initialize self.  See help(type(self)) for accurate signature."""
```


```python
class OnAccessor:
    """Accessor for geo-only (geometric) slicing on a Vector."""

    def __init__(self, vector: "'Vector'")
        """Initialize self.  See help(type(self)) for accurate signature."""
```


```python
class Vector(IndexableMixin, Tensor):
    """A Vector (k-vector) in geometric algebra."""

    @property
    def is_blade(self): ...

    @property
    def at(self): ...

    @property
    def on(self): ...

    @property
    def real(self): ...

    @property
    def imag(self): ...

    @property
    def collection(self): ...

    @property
    def contravariant_shape(self): ...

    @property
    def covariant_shape(self): ...

    @property
    def dim(self): ...

    @property
    def geo(self): ...

    @property
    def geometric_shape(self): ...

    @property
    def ndim(self): ...

    @property
    def rank(self): ...

    @property
    def shape(self): ...

    @property
    def total_rank(self): ...

    def __init__(self, data=None, /, **kwargs)
        """Allow positional argument for data: Vector(arr, grade=1, metric=m)."""

    def reverse(self) -> 'Vector'
        """Reverse operator."""

    def rev(self) -> 'Vector'
        """Short form of reverse()."""

    def inverse(self) -> 'Vector'
        """Multiplicative inverse."""

    def inv(self) -> 'Vector'
        """Short form of inverse()."""

    def transform(self, M: 'MultiVector') -> 'None'
        """Transform this Vector in-place by a motor/versor."""

    def apply_similarity(self, S: 'MultiVector', t: "'Vector | None'" = None) -> "'Vector'"
        """Apply a similarity transformation to this Vector."""

    def with_metric(self, metric: 'Metric') -> 'Vector'
        """Return a new Vector with the specified metric context."""

    def form(self) -> 'NDArray'
        """Compute the quadratic form: v · v."""

    def norm(self) -> 'NDArray'
        """Compute the norm: sqrt(|form(v)|)."""

    def unit(self) -> 'Vector'
        """Return a unit vector (norm = 1)."""

    def conjugate(self) -> 'Vector'
        """Return Vector with complex-conjugated coefficients."""

    def conj(self) -> 'Vector'
        """Short form of conjugate()."""

    def hodge(self) -> 'Vector'
        """Return Hodge dual."""

    def sum(self, axis: 'int | tuple[int, ...] | None' = None) -> 'Vector'
        """Sum over lot axis/axes, returning a Vector with reduced lot."""

    def mean(self, axis: 'int | tuple[int, ...] | None' = None) -> 'Vector'
        """Mean over lot axis/axes, returning a Vector with reduced lot."""

    def span(self) -> 'tuple[Vector, ...]'
        """Factor this blade into its constituent grade-1 Vectors."""
```

**Functions:**

```python
def basis_vector(index: 'int', metric: 'Metric') -> 'Vector'
    """Create the i-th basis vector e_i."""

def basis_vectors(metric: 'Metric') -> 'tuple[Vector, ...]'
    """Create all dim basis vectors (e0, e1, ..., e_{d-1})."""

def basis_element(indices: 'tuple[int, ...]', metric: 'Metric') -> 'Vector'
    """Create a basis element e_{i0} ^ e_{i1} ^ ... ^ e_{ik}."""

def geometric_basis(metric: 'Metric') -> 'dict[int, tuple[Vector, ...]]'
    """Create complete geometric basis for a metric."""

def pseudoscalar(metric: 'Metric') -> 'Vector'
    """Create the pseudoscalar (volume element) e_{01...d-1}."""
```


### `morphis.elements.multivector`

*Geometric Algebra - MultiVector*

```python
class MultiVector(CompositeElement):
    """A general multivector: sum of vectors of different grades."""

    @property
    def is_even(self): ...

    @property
    def is_odd(self): ...

    @property
    def is_rotor(self): ...

    @property
    def is_motor(self): ...

    @property
    def collection(self): ...

    @property
    def dim(self): ...

    @property
    def grades(self): ...

    def __init__(self, *vectors, **kwargs)
        """Create a MultiVector from Vectors or keyword arguments."""

    def grade_select(self, k: 'int') -> 'Vector | None'
        """Extract the grade-k component, or None if not present."""

    def reverse(self) -> 'MultiVector'
        """Reverse operator."""

    def rev(self) -> 'MultiVector'
        """Short form of reverse()."""

    def inverse(self) -> 'MultiVector'
        """Multiplicative inverse."""

    def inv(self) -> 'MultiVector'
        """Short form of inverse()."""

    def form(self) -> 'NDArray'
        """Compute the quadratic form: scalar(M * ~M)."""

    def norm(self) -> 'NDArray'
        """Compute the norm: sqrt(|form(M)|)."""

    def unit(self) -> 'MultiVector'
        """Return a unit multivector (M * ~M = 1)."""

    def apply_similarity(self, S: 'MultiVector', t: "'Vector | None'" = None) -> 'MultiVector'
        """Apply a similarity transformation to this MultiVector."""

    def with_metric(self, metric: 'Metric') -> 'MultiVector'
        """Return a new MultiVector with the specified metric context."""
```


### `morphis.elements.frame`

*Geometric Algebra - Frame*

```python
class Frame(GradedElement):
    """An ordered collection of grade-1 vectors in d-dimensional space."""

    @property
    def collection(self): ...

    @property
    def dim(self): ...

    @property
    def ndim(self): ...

    @property
    def shape(self): ...

    def __init__(self, *args, **kwargs)
        """Create a Frame from array data or Vectors."""

    def vector(self, i: 'int') -> 'Vector'
        """Extract the i-th vector as a grade-1 Vector."""

    def as_vector(self) -> 'Vector'
        """View frame vectors as a batch of grade-1 vectors."""

    def transform(self, M: 'MultiVector') -> 'Frame'
        """Transform this frame by a motor/versor via sandwich product."""

    def transform_inplace(self, M: 'MultiVector') -> 'None'
        """Transform this frame in-place by a motor/versor."""

    def to_vector(self) -> 'Vector'
        """Convert frame to k-vector by wedging all vectors."""

    def unit(self) -> 'Frame'
        """Return a new frame with each vector as a unit vector."""

    def with_metric(self, metric: 'Metric') -> 'Frame'
        """Return a new Frame with the specified metric context."""

    def apply_similarity(self, S: "'MultiVector'", t: "'Vector | None'" = None) -> 'Self'
        """Apply a similarity transformation to this element."""
```


### `morphis.elements.protocols`

*Geometric Algebra - Protocol Definitions*

```python
class Graded(Protocol):
    """Protocol for objects with a single grade and array data."""

    @property
    def grade(self): ...

    @property
    def data(self): ...

    def __init__(self, *args, **kwargs)
```


```python
class Spanning(Protocol):
    """Protocol for objects that span a subspace."""

    @property
    def span(self): ...

    def __init__(self, *args, **kwargs)
```


```python
class Transformable(Protocol):
    """Protocol for objects that can be transformed by motors/versors."""

    def transform(self, motor: 'MultiVector') -> 'None'
        """Transform this object in-place by a motor/versor."""

    def __init__(self, *args, **kwargs)
```


```python
class Indexable(Protocol):
    """Protocol for objects supporting index notation for contraction."""

    @property
    def data(self): ...

    def __init__(self, *args, **kwargs)
```

---

## Operations

Geometric algebra operations and products.

### `morphis.operations.operator`

*Linear Operators - Operator Class*

```python
class Operator(IndexableMixin):
    """Linear map between geometric algebra spaces."""

    @property
    def shape(self): ...

    @property
    def output_lot(self): ...

    @property
    def input_lot(self): ...

    @property
    def input_collection(self): ...

    @property
    def output_collection(self): ...

    @property
    def input_shape(self): ...

    @property
    def output_shape(self): ...

    @property
    def dim(self): ...

    @property
    def is_outermorphism(self): ...

    @property
    def vector_map(self): ...

    @property
    def H(self): ...

    @property
    def T(self): ...

    def __init__(self, data: NDArray, numpy.dtype[~_ScalarT]], input_spec: specs.VectorSpec, output_spec: specs.VectorSpec, metric: metric.Metric)
        """Initialize linear operator."""

    def apply(self, x: vector.Vector) -> vector.Vector
        """Apply operator to input vector: y = L(x)"""

    def apply_frame(self, f) -> 'Frame'
        """Apply operator to each vector in a frame."""

    def adjoint(self) -> 'Operator'
        """Compute the adjoint (conjugate transpose) operator."""

    def adj(self) -> 'Operator'
        """Short form of adjoint()."""

    def transpose(self) -> 'Operator'
        """Compute the transpose operator (no conjugation)."""

    def trans(self) -> 'Operator'
        """Short form of transpose()."""

    def solve(self, y: vector.Vector, method: 'lstsq', 'pinv' = 'lstsq', alpha: float = 0.0, r_cond: float | None = None) -> vector.Vector
        """Solve inverse problem: find x such that L(x) = y (approximately)."""

    def pseudoinverse(self, r_cond: float | None = None) -> 'Operator'
        """Compute Moore-Penrose pseudoinverse operator."""

    def pinv(self, r_cond: float | None = None) -> 'Operator'
        """Short form of pseudoinverse()."""

    def svd(self) -> tuple['Operator', NDArray, numpy.dtype[~_ScalarT]], 'Operator']
        """Singular value decomposition: L = U * diag(S) * Vt"""

    def compose(self, other: 'Operator') -> 'Operator'
        """Compose operators: (L o M)(x) = L(M(x))"""
```


### `morphis.operations.products`

*Geometric Algebra - Products*

**Functions:**

```python
def wedge(*elements)
    """Wedge product: u ^ v ^ ... ^ w"""

def antiwedge(*elements)
    """Antiwedge (regressive) product: u ∨ v ∨ ... ∨ w"""

def geometric(u, v)
    """Geometric product: uv = sum of <uv>_r over grades r"""

def grade_project(M: 'MultiVector', k: 'int') -> 'Vector'
    """Extract grade-k component from multivector: <M>_k"""

def scalar_product(u: 'Vector', v: 'Vector') -> 'NDArray'
    """Scalar part of geometric product: <uv>_0"""

def commutator(u: 'Vector', v: 'Vector') -> 'MultiVector'
    """Commutator product: [u, v] = (1 / 2) (uv - vu)"""

def anticommutator(u: 'Vector', v: 'Vector') -> 'MultiVector'
    """Anticommutator product: u * v = (1 / 2) (uv + vu)"""

def reverse(u: 'Vector | MultiVector') -> 'Vector | MultiVector'
    """Reverse: reverse(u) = (-1)^(k (k - 1) / 2) u for grade-k vector."""

def inverse(u: 'Vector | MultiVector') -> 'Vector | MultiVector'
    """Inverse: u^(-1) = reverse(u) / (u * reverse(u))"""
```


### `morphis.operations.projections`

*Geometric Algebra - Projections and Interior Products*

**Functions:**

```python
def interior_left(u: 'Vector', v: 'Vector') -> 'Vector'
    """Compute the left interior product (left contraction) of u into v:"""

def interior_right(u: 'Vector', v: 'Vector') -> 'Vector'
    """Compute the right interior product (right contraction) of u by v:"""

def dot(u: 'Vector', v: 'Vector') -> 'NDArray'
    """Compute the inner product of two grade-1 vectors: g_{mn} u^m v^n."""

def project(u: 'Vector', v: 'Vector') -> 'Vector'
    """Project vector u onto vector v:"""

def reject(u: 'Vector', v: 'Vector') -> 'Vector'
    """Compute the rejection of vector u from vector v: the component of u"""

def interior(u: 'Vector', v: 'Vector') -> 'Vector'
    """Compute the left interior product (left contraction) of u into v:"""
```


### `morphis.operations.duality`

*Geometric Algebra - Duality Operations*

**Functions:**

```python
def right_complement(u: 'Vector') -> 'Vector'
    """Compute the right complement of a vector using the Levi-Civita symbol:"""

def left_complement(u: 'Vector') -> 'Vector'
    """Compute the left complement of a vector:"""

def hodge_dual(u: 'Vector') -> 'Vector'
    """Compute the Hodge dual of a vector:"""
```


### `morphis.operations.norms`

*Geometric Algebra - Norms*

**Functions:**

```python
def form(v: 'Vector | MultiVector') -> 'NDArray'
    """Compute the quadratic form of an element."""

def norm(v: 'Vector | MultiVector') -> 'NDArray'
    """Compute the norm of an element: sqrt(|form(v)|)."""

def unit(v: 'Vector | MultiVector') -> 'Vector | MultiVector'
    """Normalize an element to unit norm."""

def conjugate(v: 'Vector') -> 'Vector'
    """Return vector with complex-conjugated coefficients."""

def hermitian_norm(v: 'Vector') -> 'NDArray'
    """Compute Hermitian norm: sqrt of hermitian_form."""

def hermitian_form(v: 'Vector') -> 'NDArray'
    """Compute Hermitian (sesquilinear) quadratic form:"""
```


### `morphis.operations.exponential`

*Geometric Algebra - Exponentials and Logarithms*

**Functions:**

```python
def exp_vector(B: 'Vector') -> 'MultiVector'
    """Compute the exponential of a vector: exp(B)"""

def log_versor(M: 'MultiVector') -> 'Vector'
    """Extract the bivector generator from a rotor/versor: log(M)"""

def slerp(R0: 'MultiVector', R1: 'MultiVector', t: 'float | NDArray') -> 'MultiVector'
    """Spherical linear interpolation between rotors."""
```


### `morphis.operations.structure`

*Geometric Algebra - Algebraic Structure*

**Functions:**

```python
def permutation_sign(perm: tuple[int, ...]) -> int
    """Compute the sign of a permutation (+1 for even, -1 for odd). Uses the"""

def antisymmetrize(tensor: NDArray, numpy.dtype[~_ScalarT]], k: int, cdim: int = 0) -> NDArray, numpy.dtype[~_ScalarT]]
    """Antisymmetrize a tensor over its last k axes. Computes the projection onto"""

def antisymmetric_symbol(k: int, d: int) -> NDArray, numpy.dtype[~_ScalarT]]
    """Compute the k-index antisymmetric symbol eps^{m_1 ... m_k} in d dimensions."""

def levi_civita(d: int) -> NDArray, numpy.dtype[~_ScalarT]]
    """Get the Levi-Civita symbol eps^{m_1 ... m_d} for d dimensions. This is the"""

def generalized_delta(k: int, d: int) -> NDArray, numpy.dtype[~_ScalarT]]
    """Compute the generalized Kronecker delta d^{m_1 ... m_k}_{n_1 ... n_k} in d"""

def wedge_signature(grades: tuple[int, ...]) -> str
    """Einsum signature for wedge product including delta contraction."""

def wedge_normalization(grades: tuple[int, ...]) -> float
    """Compute the normalization factor for wedge product."""

def interior_left_signature(j: int, k: int) -> str
    """Einsum signature for left interior product (left contraction) of grade j"""

def interior_right_signature(j: int, k: int) -> str
    """Einsum signature for right interior product (right contraction) of grade j"""

def complement_signature(k: int, d: int) -> str
    """Einsum signature for right complement using the Levi-Civita symbol. Maps"""

def norm_squared_signature(k: int) -> str
    """Einsum signature for blade norm squared. For k = 1 returns"""

def geometric_signature(j: int, k: int, c: int) -> str
    """Einsum signature for geometric product with c contractions."""

def geometric_normalization(j: int, k: int, c: int) -> float
    """Normalization factor for geometric product with c contractions."""

def interior_signature(j: int, k: int) -> str
    """Einsum signature for left interior product (left contraction) of grade j"""
```


### `morphis.operations.factorization`

*Geometric Algebra - Vector Factorization*

**Functions:**

```python
def factor(b: 'Vector', tol: 'float | None' = None) -> 'Vector'
    """Factor a k-vector into k spanning grade-1 vectors."""

def spanning_vectors(b: 'Vector', tol: 'float | None' = None) -> 'tuple[Vector, ...]'
    """Factor a vector into its constituent grade-1 vectors."""
```


### `morphis.operations.spectral`

*Geometric Algebra - Spectral Analysis*

**Functions:**

```python
def bivector_to_skew_matrix(b: 'Vector') -> 'NDArray'
    """Convert a bivector to its d×d skew-symmetric matrix representation."""

def bivector_eigendecomposition(b: 'Vector', tol: 'float | None' = None) -> 'tuple[NDArray, list[Vector]]'
    """Decompose a bivector into orthogonal rotation planes."""

def principal_vectors(b: 'Vector', tol: 'float | None' = None) -> 'tuple[Vector, ...]'
    """Extract principal orthonormal vectors spanning a blade's subspace."""
```


### `morphis.operations.outermorphism`

*Geometric Algebra - Outermorphisms*

**Functions:**

```python
def apply_exterior_power(A: "'Operator'", blade: "'Vector'", k: 'int') -> "'Vector'"
    """Apply k-th exterior power of a vector map to a grade-k vec."""

def apply_outermorphism(A: "'Operator'", M: "'MultiVector'") -> "'MultiVector'"
    """Apply an outermorphism to a multivector."""
```


### `morphis.operations.matrix_rep`

*Geometric Algebra - Matrix Representations*

**Functions:**

```python
def vector_to_array(b: 'Vector') -> 'NDArray'
    """Flatten a blade's components to a 1D vector."""

def vector_to_vector(v: 'NDArray', grade: 'int', metric: 'Metric') -> 'Vector'
    """Reconstruct a blade from a flattened vector."""

def multivector_to_array(M: 'MultiVector') -> 'NDArray'
    """Flatten a multivector to a vector of length 2^d."""

def array_to_multivector(v: 'NDArray', metric: 'Metric') -> 'MultiVector'
    """Reconstruct a multivector from a flattened vector."""

def left_matrix(A: 'Vector | MultiVector') -> 'NDArray'
    """Compute the matrix representation of left multiplication by A."""

def right_matrix(A: 'Vector | MultiVector') -> 'NDArray'
    """Compute the matrix representation of right multiplication by A."""

def operator_to_matrix(L: 'Operator') -> 'NDArray'
    """Convert an Operator to its flattened 2D matrix form."""
```

---

## Algebra

Linear algebra utilities for operators.

### `morphis.algebra.specs`

*Linear Algebra - Vector Specifications*

```python
class VectorSpec(BaseModel):
    """Specification for a k-vector's structure in a linear operator context."""

    @property
    def geo(self): ...

    @property
    def shape(self): ...

    @property
    def geometric_shape(self): ...

    @property
    def collection(self): ...

    @property
    def total_axes(self): ...

    def __init__(self, /, **data: 'Any') -> 'None'
        """Create a new model by parsing and validating input data from keyword arguments."""
```

**Functions:**

```python
def vector_spec(grade: int, dim: int, lot: tuple[int, ...] | None = None, collection: int | None = None) -> specs.VectorSpec
    """Create a VectorSpec with convenient defaults."""
```


### `morphis.algebra.patterns`

*Linear Algebra - Einsum Pattern Generation*

**Functions:**

```python
def operator_shape(input_spec: specs.VectorSpec, output_spec: specs.VectorSpec, input_lot: tuple[int, ...] | None = None, output_lot: tuple[int, ...] | None = None, *, input_collection: tuple[int, ...] | None = None, output_collection: tuple[int, ...] | None = None) -> tuple[int, ...]
    """Compute the expected shape of operator data given specs and lot shapes."""
```


### `morphis.algebra.contraction`

*Linear Algebra - Tensor Contraction*

```python
class IndexedTensor:
    """Lightweight wrapper that pairs a tensor with index labels for contraction."""

    def __init__(self, tensor: "'Vector | Operator'", indices: 'str', output_geo_indices: 'str' = '')
        """Create an indexed tensor wrapper."""
```

**Functions:**

```python
def contract(signature: 'str', *tensors: "'Vector | Operator'") -> "'Vector'"
    """Einsum-style contraction for Morphis tensors."""
```


### `morphis.algebra.solvers`

*Linear Algebra - Structured Linear Algebra Solvers*

**Functions:**

```python
def structured_lstsq(op: 'Operator', target: vector.Vector, alpha: float = 0.0) -> vector.Vector
    """Solve least squares problem with optional Tikhonov regularization."""

def structured_pinv_solve(op: 'Operator', target: vector.Vector, r_cond: float | None = None) -> vector.Vector
    """Solve using Moore-Penrose pseudoinverse."""

def structured_pinv(op: 'Operator', r_cond: float | None = None) -> 'Operator'
    """Compute Moore-Penrose pseudoinverse operator."""

def structured_svd(op: 'Operator') -> tuple['Operator', NDArray, numpy.dtype[~_ScalarT]], 'Operator']
    """Structured singular value decomposition: L = U * diag(S) * Vt"""
```

---

## Transforms

Geometric transformations.

### `morphis.transforms.rotations`

*Geometric Algebra - Rotation and Similarity Constructors*

**Functions:**

```python
def rotor(B: 'Vector', angle: 'float | NDArray') -> 'MultiVector'
    """Create a rotor for pure rotation about the origin."""

def rotation_about_point(p: 'Vector', B: 'Vector', angle: 'float | NDArray') -> 'MultiVector'
    """Create a motor for rotation about an arbitrary center point (PGA)."""

def align_vectors(u: 'Vector', v: 'Vector') -> 'MultiVector'
    """Create a similarity versor that transforms u to v."""

def point_alignment(u1: 'Vector', u2: 'Vector', v1: 'Vector', v2: 'Vector') -> 'tuple[MultiVector, Vector]'
    """Compute similarity transform aligning two point pairs."""
```


### `morphis.transforms.actions`

*Geometric Algebra - Transformation Actions*

**Functions:**

```python
def rotate(b: 'Vector', B: 'Vector', angle: 'float | NDArray') -> 'Vector'
    """Rotate a vector by angle in the plane defined by bivector B."""

def translate(u: 'Vector', v: 'Vector') -> 'Vector'
    """Translate a vector by a direction (PGA only)."""

def transform(b: 'Vector', M: 'MultiVector') -> 'Vector'
    """Apply a motor/versor transformation to a vector via sandwich product."""
```


### `morphis.transforms.projective`

*Geometric Algebra - Projective Operations*

**Functions:**

```python
def point(x: 'NDArray', metric: 'Metric | None' = None, collection: 'tuple[int, ...] | None' = None) -> 'Vector'
    """Embed a Euclidean point into projective space. Points have unit weight"""

def direction(v: 'NDArray', metric: 'Metric | None' = None, collection: 'tuple[int, ...] | None' = None) -> 'Vector'
    """Embed a Euclidean direction into projective space. Directions have zero"""

def weight(p: 'Vector') -> 'NDArray'
    """Extract the weight (e_0 component) of a projective vector."""

def bulk(p: 'Vector') -> 'NDArray'
    """Extract the bulk (Euclidean components) of a projective vector."""

def euclidean(p: 'Vector') -> 'NDArray'
    """Project a projective point to Euclidean coordinates by dividing bulk by"""

def is_point(p: 'Vector') -> 'NDArray'
    """Check if a projective vector represents a point (nonzero weight)."""

def is_direction(p: 'Vector') -> 'NDArray'
    """Check if a projective vector represents a direction (zero weight)."""

def line(p: 'Vector', q: 'Vector') -> 'Vector'
    """Construct a line through two points as the bivector p ^ q."""

def plane(p: 'Vector', q: 'Vector', r: 'Vector') -> 'Vector'
    """Construct a plane through three points as the trivector p ^ q ^ r."""

def plane_from_point_and_line(p: 'Vector', l: 'Vector') -> 'Vector'
    """Construct a plane through a point and a line as p ^ l."""

def distance_point_to_point(p: 'Vector', q: 'Vector') -> 'NDArray'
    """Compute Euclidean distance between two points."""

def distance_point_to_line(p: 'Vector', l: 'Vector') -> 'NDArray'
    """Compute distance from a point to a line as |p ^ l| / |l|."""

def distance_point_to_plane(p: 'Vector', h: 'Vector') -> 'NDArray'
    """Compute distance from a point to a plane as |p ^ h| / |h|."""

def are_collinear(p: 'Vector', q: 'Vector', r: 'Vector', tol: 'float | None' = None) -> 'NDArray'
    """Check if three points are collinear: p ^ q ^ r = 0."""

def are_coplanar(p: 'Vector', q: 'Vector', r: 'Vector', s: 'Vector', tol: 'float | None' = None) -> 'NDArray'
    """Check if four points are coplanar: p ^ q ^ r ^ s = 0."""

def point_on_line(p: 'Vector', l: 'Vector', tol: 'float | None' = None) -> 'NDArray'
    """Check if a point lies on a line: p ^ l = 0."""

def point_on_plane(p: 'Vector', h: 'Vector', tol: 'float | None' = None) -> 'NDArray'
    """Check if a point lies on a plane: p ^ h = 0."""

def line_in_plane(l: 'Vector', h: 'Vector', tol: 'float | None' = None) -> 'NDArray'
    """Check if a line lies in a plane: l ^ h = 0."""

def translator(v: 'Vector') -> 'MultiVector'
    """Create a translator for pure translation (PGA only)."""

def screw(B: 'Vector', angle: 'float | NDArray', translation: 'Vector', center: 'Vector | None' = None) -> 'MultiVector'
    """Create a motor for screw motion (rotation + translation along axis)."""
```

---

## Visualization

Visualization and rendering tools.

### `morphis.visuals.scene`

*Scene - Unified Visualization Interface*

```python
class SceneEffect(BaseModel):
    """Effect wrapper that uses string element IDs."""

    def evaluate(self, t: 'float') -> 'float'
        """Evaluate opacity at time t."""

    def is_active(self, t: 'float') -> 'bool'

    def __init__(self, /, **data: 'Any') -> 'None'
        """Create a new model by parsing and validating input data from keyword arguments."""
```


```python
class TrackedElement(BaseModel):
    """Internal tracking for elements added to the scene."""

    def __init__(self, /, **data: 'Any') -> 'None'
        """Create a new model by parsing and validating input data from keyword arguments."""
```


```python
class Scene:
    """Unified visualization interface for static and animated scenes."""

    @property
    def theme(self): ...

    @property
    def frame_rate(self): ...

    def __init__(self, projection: 'tuple[int, ...] | None' = None, theme: 'str | Theme' = 'obsidian', size: 'tuple[int, int]' = (600, 600), frame_rate: 'int' = 30, backend: 'str' = 'pyvista', show_basis: 'bool' = True)
        """Initialize self.  See help(type(self)) for accurate signature."""

    def add(self, element: 'Element', representation: 'str | None' = None, color: 'Color | None' = None, opacity: 'float' = 1.0, **kwargs) -> 'str'
        """Add an element to the scene."""

    def remove(self, element_id: 'str') -> 'None'
        """Remove an element from the scene."""

    def set_projection(self, axes: 'tuple[int, ...]') -> 'None'
        """Set projection axes for nD -> 3D."""

    def camera(self, position: 'tuple[float, float, float] | None' = None, focal_point: 'tuple[float, float, float] | None' = None, up: 'tuple[float, float, float] | None' = None) -> 'None'
        """Set camera position and orientation."""

    def reset_camera(self) -> 'None'
        """Reset camera to fit all objects."""

    def set_clipping_range(self, near: 'float', far: 'float') -> 'None'
        """Set camera clipping range."""

    def add_light(self, position: 'tuple[float, float, float]' = (1, 1, 1), focal_point: 'tuple[float, float, float]' = (0, 0, 0), intensity: 'float' = 1.0, color: 'Color | None' = None, directional: 'bool' = True, attenuation: 'tuple[float, float, float] | None' = None) -> 'str'
        """Add a light to the scene."""

    def remove_light(self, light_id: 'str') -> 'None'
        """Remove a light from the scene."""

    def clear_lights(self) -> 'None'
        """Remove all user-added lights."""

    def fade_in(self, element: 'Element', t: 'float', duration: 'float') -> 'None'
        """Schedule a fade-in effect for an element."""

    def fade_out(self, element: 'Element', t: 'float', duration: 'float') -> 'None'
        """Schedule a fade-out effect for an element."""

    def capture(self, t: 'float') -> 'None'
        """Render current state at time t (live mode)."""

    def show(self) -> 'None'
        """Wait for user to close window."""

    def close(self) -> 'None'
        """Close the scene and clean up."""
```


### `morphis.visuals.canvas`

*3D Visualization Canvas*

```python
class ArrowStyle(BaseModel):
    """Configuration for arrow rendering proportions."""

    def __init__(self, /, **data: 'Any') -> 'None'
        """Create a new model by parsing and validating input data from keyword arguments."""
```


```python
class CurveStyle(BaseModel):
    """Configuration for curve rendering."""

    def __init__(self, /, **data: 'Any') -> 'None'
        """Create a new model by parsing and validating input data from keyword arguments."""
```


```python
class ModelStyle(BaseModel):
    """Configuration for 3D model rendering."""

    def __init__(self, /, **data: 'Any') -> 'None'
        """Create a new model by parsing and validating input data from keyword arguments."""
```


```python
class Canvas:
    """3D visualization canvas with theme support and automatic color cycling."""

    def __init__(self, theme: 'str | Theme' = Theme(name='obsidian', background=(0.12, 0.13, 0.14), e1=(0.85, 0.35, 0.3), e2=(0.4, 0.75, 0.45), e3=(0.35, 0.5, 0.9), palette=Palette(colors=((0.95, 0.55, 0.45), (0.55, 0.8, 0.7), (0.9, 0.75, 0.4), (0.5, 0.65, 0.9), (0.85, 0.5, 0.7), (0.45, 0.8, 0.85))), accent=(0.95, 0.85, 0.4), muted=(0.45, 0.47, 0.5), label=(0.82, 0.84, 0.86)), title: 'str | None' = None, size: 'tuple[int, int]' = (1200, 900), show_basis: 'bool' = True, basis_axes: 'tuple[int, int, int]' = (0, 1, 2))
        """Initialize self.  See help(type(self)) for accurate signature."""

    def basis(self, scale: 'float' = 1.0, axes: 'tuple[int, int, int]' = (0, 1, 2), labels: 'bool' = True)
        """Draw coordinate axes at origin using PyVista's native axes."""

    def set_basis_axes(self, axes: 'tuple[int, int, int]')
        """Change which basis vectors are displayed."""

    def arrow(self, start, direction, color: 'Color | None' = None, shaft_radius: 'float | None' = None)
        """Add an arrow to the scene."""

    def arrows(self, starts, directions, color: 'Color | None' = None, colors: 'Palette | list | None' = None, shaft_radius: 'float | None' = None)
        """Add multiple arrows to the scene."""

    def curve(self, points, color: 'Color | None' = None, radius: 'float | None' = None, closed: 'bool' = False)
        """Add a smooth curve through the given points."""

    def curves(self, point_sets, color: 'Color | None' = None, colors: 'Palette | list | None' = None, radius: 'float | None' = None)
        """Add multiple curves to the scene."""

    def point(self, position, color: 'Color | None' = None, radius: 'float' = 0.03)
        """Add a point (sphere) to the scene."""

    def points(self, positions, color: 'Color | None' = None, colors: 'Palette | list | None' = None, radius: 'float' = 0.03)
        """Add multiple points to the scene."""

    def plane(self, center, normal, size: 'float' = 1.0, color: 'Color | None' = None, opacity: 'float' = 0.3)
        """Add a semi-transparent plane to the scene."""

    def model(self, visual: "'VisualModel'", color: 'Color | None' = None, opacity: 'float' = 1.0, style: 'ModelStyle | None' = None)
        """Add a 3D model to the scene."""

    def reset_colors(self)
        """Reset color cycling to start of palette."""

    def camera(self, position: 'tuple[float, float, float] | None' = None, focal_point: 'tuple[float, float, float] | None' = None, view_up: 'tuple[float, float, float] | None' = None)
        """Set camera position and orientation."""

    def show(self, block: 'bool' = True)
        """Display the visualization."""
```


### `morphis.visuals.theme`

*Visualization Themes and Color Palettes*

```python
class Palette(BaseModel):
    """A color palette designed for a specific background."""

    def cycle(self, n: int) -> list[tuple[float, float, float]]
        """Return n colors, cycling through the palette."""

    def __init__(self, /, **data: 'Any') -> 'None'
        """Create a new model by parsing and validating input data from keyword arguments."""
```


```python
class Theme(BaseModel):
    """Complete visual theme for 3D rendering."""

    @property
    def basis_colors(self): ...

    @property
    def foreground(self): ...

    @property
    def axis_color(self): ...

    @property
    def grid_color(self): ...

    @property
    def text_color(self): ...

    @property
    def edge_color(self): ...

    def is_light(self) -> bool
        """Check if this is a light theme based on background luminance."""

    def __init__(self, /, **data: 'Any') -> 'None'
        """Create a new model by parsing and validating input data from keyword arguments."""
```

**Functions:**

```python
def get_theme(name: str) -> theme.Theme
    """Retrieve a theme by name."""

def list_themes() -> list[str]
    """Return list of available theme names."""
```


### `morphis.visuals.projection`

*Projection Utilities for High-Dimensional Vectors*

```python
class ProjectionConfig(BaseModel):
    """Configuration for projecting high-dimensional blades to 3D."""

    def __init__(self, /, **data: 'Any') -> 'None'
        """Create a new model by parsing and validating input data from keyword arguments."""
```

**Functions:**

```python
def project_vector(blade: vector.Vector, config: projection.ProjectionConfig) -> vector.Vector
    """Project a vector blade to lower dimension."""

def project_bivector(blade: vector.Vector, config: projection.ProjectionConfig) -> vector.Vector
    """Project a bivector blade to lower dimension."""

def project_trivector(blade: vector.Vector, config: projection.ProjectionConfig) -> vector.Vector
    """Project a trivector blade to 3D."""

def project_quadvector(blade: vector.Vector, config: projection.ProjectionConfig) -> vector.Vector
    """Project a quadvector (4-blade) to lower dimension."""

def project_blade(blade: vector.Vector, config: projection.ProjectionConfig | None = None) -> vector.Vector
    """Project a blade to lower dimension for visualization."""

def get_projection_axes(blade: vector.Vector, config: projection.ProjectionConfig | None = None) -> tuple[int, ...]
    """Get the axes that would be used for projection."""
```


### `morphis.visuals.drawing.vectors`

*Vector Drawing Functions*

```python
class VectorStyle(BaseModel):
    """Style parameters for blade rendering."""

    def __init__(self, /, **data: 'Any') -> 'None'
        """Create a new model by parsing and validating input data from keyword arguments."""
```

**Functions:**

```python
def create_vector_mesh(origin: 'ndarray', direction: 'ndarray', shaft_radius: 'float' = 0.008) -> 'tuple[PolyData | None, PolyData]'
    """Create meshes for a vector (grade 1 blade)."""

def create_bivector_mesh(origin: 'ndarray', u: 'ndarray', v: 'ndarray', shaft_radius: 'float' = 0.006, face_opacity: 'float' = 0.25) -> 'tuple[PolyData, PolyData, PolyData]'
    """Create meshes for a bivector (grade 2 blade)."""

def create_trivector_mesh(origin: 'ndarray', u: 'ndarray', v: 'ndarray', w: 'ndarray', shaft_radius: 'float' = 0.006, face_opacity: 'float' = 0.15) -> 'tuple[PolyData, PolyData, PolyData]'
    """Create meshes for a trivector."""

def create_quadvector_mesh(origin: 'ndarray', u: 'ndarray', v: 'ndarray', w: 'ndarray', x: 'ndarray', projection_axes: 'tuple[int, int, int]' = (0, 1, 2), shaft_radius: 'float' = 0.006, face_opacity: 'float' = 0.12) -> 'tuple[PolyData, PolyData, PolyData]'
    """Create meshes for a quadvector (4-blade) as a tesseract projection."""

def create_frame_mesh(origin: 'ndarray', vectors: 'ndarray', shaft_radius: 'float' = 0.008, projection_axes: 'tuple[int, int, int] | None' = None, filled: 'bool' = False) -> 'tuple[PolyData | None, PolyData | None, PolyData]'
    """Create meshes for a frame (k arrows from origin)."""

def create_blade_mesh(grade: 'int', origin: 'ndarray', vectors: 'ndarray', shaft_radius: 'float' = 0.008, edge_radius: 'float' = 0.006, projection_axes: 'tuple[int, int, int] | None' = None) -> 'tuple[PolyData | None, PolyData | None, PolyData]'
    """Create meshes for a blade of any grade."""

def draw_blade(plotter: 'Plotter', b: 'Vector', p: 'Vector | ndarray | tuple' = None, color: 'Color' = (0.85, 0.85, 0.85), tetrad: 'bool' = True, surface: 'bool' = True, label: 'bool' = False, name: 'str | None' = None, shaft_radius: 'float' = 0.008, edge_radius: 'float' = 0.006, label_offset: 'float' = 0.08)
    """Draw a blade at a given position."""

def draw_coordinate_basis(plotter: 'Plotter', scale: 'float' = 1.0, color: 'Color' = (0.85, 0.85, 0.85), tetrad: 'bool' = True, surface: 'bool' = False, label: 'bool' = True, label_offset: 'float' = 0.08, labels: 'tuple[str, str, str] | None' = None) -> 'list | None'
    """Draw the standard coordinate basis e1, e2, e3."""

def draw_basis_blade(plotter: 'Plotter', indices: 'tuple[int, ...]', position: 'ndarray | tuple' = (0, 0, 0), scale: 'float' = 1.0, color: 'Color' = (0.85, 0.85, 0.85), tetrad: 'bool' = True, surface: 'bool' = True, label: 'bool' = False, name: 'str | None' = None, label_offset: 'float' = 0.08)
    """Draw a basis k-blade (e1, e12, e123, etc.) at a given position."""

def render_scalar(blade: 'Vector', canvas, position: 'tuple[float, float, float] | None' = None, style: 'VectorStyle | None' = None) -> 'None'
    """Render scalar blade as sphere at origin (or specified position)."""

def render_vector(blade: 'Vector', canvas, projection=None, style: 'VectorStyle | None' = None) -> 'None'
    """Render vector blade as arrow from origin."""

def render_bivector(blade: 'Vector', canvas, mode: "'circle', 'parallelogram', 'plane', 'circular_arrow'" = 'circle', projection=None, style: 'VectorStyle | None' = None) -> 'None'
    """Render bivector blade with multiple visualization modes."""

def render_trivector(blade: 'Vector', canvas, mode: "'parallelepiped', 'sphere'" = 'parallelepiped', projection=None, style: 'VectorStyle | None' = None) -> 'None'
    """Render trivector blade as 3D volume element."""

def visualize_blade(blade: 'Vector', canvas=None, mode: 'str' = 'auto', projection=None, show_dual: 'bool' = False, style: 'VectorStyle | None' = None)
    """Main entry point for blade visualization."""

def visualize_blades(blades: 'list', canvas=None, style: 'VectorStyle | None' = None)
    """Visualize multiple blades on same canvas."""
```


### `morphis.visuals.contexts`

*Context-Specific Vector Visualization*

```python
class PGAStyle(VectorStyle):
    """Style parameters for PGA visualization."""

    def __init__(self, /, **data: 'Any') -> 'None'
        """Create a new model by parsing and validating input data from keyword arguments."""
```

**Functions:**

```python
def render_pga_point(blade: vector.Vector, canvas: canvas.Canvas, style: contexts.PGAStyle | None = None) -> None
    """Render PGA point (grade-1 in PGA) as sphere at Euclidean location."""

def render_pga_line(blade: vector.Vector, canvas: canvas.Canvas, style: contexts.PGAStyle | None = None) -> None
    """Render PGA line (grade-2 in PGA) as extended line segment."""

def render_pga_plane(blade: vector.Vector, canvas: canvas.Canvas, style: contexts.PGAStyle | None = None) -> None
    """Render PGA plane (grade-3 in PGA) as extended plane surface."""

def is_pga_context(vec: vector.Vector) -> bool
    """Check if blade has PGA context."""

def visualize_pga_blade(blade: vector.Vector, canvas: canvas.Canvas | None = None, style: contexts.PGAStyle | None = None) -> canvas.Canvas
    """Visualize a blade in PGA context with appropriate geometric interpretation."""

def visualize_pga_scene(*blades: vector.Vector, canvas: canvas.Canvas | None = None, style: contexts.PGAStyle | None = None) -> canvas.Canvas
    """Visualize multiple PGA blades forming a geometric scene."""
```


### `morphis.visuals.operations`

*Visualization of Geometric Algebra Operations*

```python
class OperationStyle(BaseModel):
    """Style parameters for operation visualization."""

    def __init__(self, /, **data: 'Any') -> 'None'
        """Create a new model by parsing and validating input data from keyword arguments."""
```

**Functions:**

```python
def render_join(u: vector.Vector, v: vector.Vector, canvas: canvas.Canvas | None = None, projection: projection.ProjectionConfig | None = None, style: operations.OperationStyle | None = None) -> canvas.Canvas
    """Visualize the join (wedge product) of two blades."""

def render_meet(u: vector.Vector, v: vector.Vector, canvas: canvas.Canvas | None = None, projection: projection.ProjectionConfig | None = None, style: operations.OperationStyle | None = None) -> canvas.Canvas
    """Visualize the meet (intersection) of two blades."""

def render_meet_join(u: vector.Vector, v: vector.Vector, canvas: canvas.Canvas | None = None, show: 'meet', 'join', 'both' = 'both', projection: projection.ProjectionConfig | None = None, style: operations.OperationStyle | None = None) -> canvas.Canvas
    """Visualize meet and/or join operations between two blades."""

def render_with_dual(blade: vector.Vector, canvas: canvas.Canvas | None = None, dual_type: 'right', 'left', 'hodge' = 'right', projection: projection.ProjectionConfig | None = None, style: operations.OperationStyle | None = None) -> canvas.Canvas
    """Visualize a blade alongside its dual."""
```


### `morphis.visuals.loop`

*Animation - Observer and Recorder*

```python
class Snapshot(BaseModel):
    """State of all tracked objects at a specific time."""

    def __init__(self, /, **data: 'Any') -> 'None'
        """Create a new model by parsing and validating input data from keyword arguments."""
```


```python
class AnimationTrack(BaseModel):
    """Animation-specific tracking info for a blade, frame, or model."""

    def __init__(self, /, **data: 'Any') -> 'None'
        """Create a new model by parsing and validating input data from keyword arguments."""
```


```python
class Animation:
    """Animation observer and recorder."""

    @property
    def observer(self): ...

    def __init__(self, frame_rate: int = 60, theme: str | theme.Theme = 'obsidian', size: tuple[int, int] = (1800, 1350), show_basis: bool = True, auto_camera: bool = True, fps: int | None = None)
        """Initialize self.  See help(type(self)) for accurate signature."""

    def watch(self, *targets: vector.Vector | frame.Frame | model.VisualModel, color: tuple[float, float, float] | None = None, filled: bool = False, opacity: float = 1.0) -> int | list[int]
        """Register one or more blades, frames, or models to observe."""

    def unwatch(self, *targets: vector.Vector | frame.Frame | model.VisualModel)
        """Stop watching one or more blades, frames, or models."""

    def set_vectors(self, blade: vector.Vector, vectors: NDArray, numpy.dtype[~_ScalarT]], origin: NDArray, numpy.dtype[~_ScalarT]] | None = None)
        """Set the spanning vectors for a blade directly."""

    def set_projection(self, axes: tuple[int, int, int], labels: tuple[str, str, str] | None = None)
        """Set the coordinate projection for the canvas."""

    def fade_in(self, target: vector.Vector | frame.Frame, t: float, duration: float)
        """Schedule a fade-in effect."""

    def fade_out(self, target: vector.Vector | frame.Frame, t: float, duration: float)
        """Schedule a fade-out effect."""

    def start(self, live: bool = False)
        """Start an animation session."""

    def capture(self, t: float)
        """Capture the current state of all tracked objects at time t."""

    def finish(self)
        """End an animation session (live mode)."""

    def play(self, loop: bool = False)
        """Play back recorded snapshots (batch mode)."""

    def save(self, filename: str, loop: bool = True)
        """Save the animation to a file."""

    def camera(self, position=None, focal_point=None)
        """Set camera position and/or focal point."""

    def set_basis_labels(self, labels: tuple[str, str, str])
        """Set custom labels for the coordinate basis axes."""

    def close(self)
        """Close the animation window."""

    def track(self, *targets: vector.Vector | frame.Frame | model.VisualModel, color: tuple[float, float, float] | None = None, filled: bool = False, opacity: float = 1.0) -> int | list[int]
        """Register one or more blades, frames, or models to observe."""

    def untrack(self, *targets: vector.Vector | frame.Frame | model.VisualModel)
        """Stop watching one or more blades, frames, or models."""
```

---

## Utilities

Helper utilities.

### `morphis.utils.pretty`

*Pretty printing utilities for examples and debugging.*

**Functions:**

```python
def format_matrix(arr: 'NDArray', precision: 'int' = 4, max_rows: 'int' = 8, max_cols: 'int' = 8, max_slices: 'int' = 5) -> 'str'
    """Format array with box-drawing characters for a math-style look."""

def print_matrix(arr: 'NDArray', precision: 'int' = 4) -> 'None'
    """Print array with box-drawing characters."""

def section(title: 'str', width: 'int' = 70) -> 'None'
    """Print a section header."""

def subsection(title: 'str') -> 'None'
    """Print a subsection header."""

def show_vec(name: 'str', blade: 'Vector', precision: 'int' = 4) -> 'None'
    """Print blade info with matrix-style data formatting."""

def show_array(name: 'str', arr, precision: 'int' = 4) -> 'None'
    """Print array with matrix-style formatting."""

def show_scalar(name: 'str', value, precision: 'int' = 4) -> 'None'
    """Print a scalar value."""

def show_mv(name: 'str', M: 'MultiVector', precision: 'int' = 4) -> 'None'
    """Print multivector components with matrix-style formatting."""

def format_vector(v: 'Vector', precision: 'int' = 4) -> 'str'
    """Format a Vector for display."""

def format_multivector(M: 'MultiVector', precision: 'int' = 4) -> 'str'
    """Format a MultiVector for display."""

def format_frame(F: 'Frame', precision: 'int' = 4) -> 'str'
    """Format a Frame for display."""
```


### `morphis.utils.observer`

*Observer - Watch and observe GA objects*

```python
class TrackedObject(BaseModel):
    """Internal record for a tracked object."""

    def __init__(self, /, **data: 'Any') -> 'None'
        """Create a new model by parsing and validating input data from keyword arguments."""
```


```python
class Observer:
    """Observes GA objects by holding references to them."""

    def __init__(self)
        """Initialize self.  See help(type(self)) for accurate signature."""

    def watch(self, *objects: base.Element, names: list[str] | None = None) -> 'Observer'
        """Register one or more objects to observe."""

    def unwatch(self, *objects: base.Element) -> 'Observer'
        """Stop watching one or more objects."""

    def clear(self) -> 'Observer'
        """Stop tracking all objects."""

    def get(self, obj_or_name: base.Element | str) -> numpy.ndarray | None
        """Get the current data for a tracked object."""

    def snapshot(self) -> dict[int, numpy.ndarray]
        """Get current state of all tracked objects."""

    def snapshot_named(self) -> dict[str, numpy.ndarray]
        """Get current state of named objects only."""

    def reset_baseline(self, *objects: base.Element) -> 'Observer'
        """Reset the baseline (for diff computation) to current state."""

    def diff(self, obj_or_name: base.Element | str) -> numpy.ndarray | None
        """Compute difference from baseline for an object."""

    def diff_norm(self, obj_or_name: base.Element | str) -> float | None
        """Compute norm of difference from baseline."""

    def objects(self) -> list[base.Element]
        """Return list of all tracked objects."""

    def ids(self) -> list[int]
        """Return list of all tracked object IDs."""

    def names(self) -> list[str]
        """Return list of all named objects."""

    def print_state(self, prefix: str = '')
        """Print current state of all tracked objects (for debugging)."""

    def spanning_vectors(self, obj_or_name: base.Element | str) -> tuple['Vector', ...] | None
        """Get the spanning vectors for a tracked vec."""

    def spanning_vectors_as_array(self, obj_or_name: base.Element | str) -> numpy.ndarray | None
        """Get spanning vectors as a stacked numpy array."""

    def capture_state(self, obj_or_name: base.Element | str) -> dict | None
        """Capture complete visualization state for a tracked vec."""

    def track(self, *objects: base.Element, names: list[str] | None = None) -> 'Observer'
        """Register one or more objects to observe."""

    def untrack(self, *objects: base.Element) -> 'Observer'
        """Stop watching one or more objects."""
```

