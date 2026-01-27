# Morphis Package API Reference

*Auto-generated documentation*

---

## Elements

Core geometric algebra objects.

### `morphis.elements.base`

```python
class Element(BaseModel):
    @property
    def dim(self): ...
    def __init__(self, /, **data: 'Any') -> 'None'
```

```python
class GradedElement(Element):
    @property
    def shape(self): ...
    @property
    def ndim(self): ...
    @property
    def dim(self): ...
    def with_metric(self, metric: 'Metric') -> 'Self'
    def __init__(self, /, **data: 'Any') -> 'None'
```

```python
class CompositeElement(Element):
    @property
    def grades(self): ...
    @property
    def dim(self): ...
    def with_metric(self, metric: 'Metric') -> 'Self'
    def __init__(self, /, **data: 'Any') -> 'None'
```


### `morphis.elements.metric`

```python
class GASignature(Enum):
    ...
```

```python
class GAStructure(Enum):
    ...
```

```python
class Metric(BaseModel):
    @property
    def dim(self): ...
    @property
    def euclidean_dim(self): ...
    @property
    def signature_tuple(self): ...
    def is_compatible(self, other: 'Metric') -> 'bool'
    def __init__(self, /, **data: 'Any') -> 'None'
```

**Functions:**

```python
def metric(dim: 'int', signature: 'str' = 'euclidean', structure: 'str' = 'flat') -> 'Metric'
def euclidean_metric(dim: 'int') -> 'Metric'
def pga_metric(dim: 'int') -> 'Metric'
def lorentzian_metric(dim: 'int') -> 'Metric'
```


### `morphis.elements.tensor`

```python
class Tensor(Element):
    @property
    def rank(self): ...
    @property
    def total_rank(self): ...
    @property
    def geometric_shape(self): ...
    @property
    def contravariant_shape(self): ...
    @property
    def covariant_shape(self): ...
    @property
    def shape(self): ...
    @property
    def ndim(self): ...
    @property
    def dim(self): ...
    def with_metric(self, metric: 'Metric') -> 'Self'
    def __init__(self, /, **data: 'Any') -> 'None'
```


### `morphis.elements.vector`

```python
class Vector(Tensor):
    @property
    def is_blade(self): ...
    @property
    def contravariant_shape(self): ...
    @property
    def covariant_shape(self): ...
    @property
    def dim(self): ...
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
    def reverse(self) -> 'Vector'
    def rev(self) -> 'Vector'
    def inverse(self) -> 'Vector'
    def inv(self) -> 'Vector'
    def transform(self, M: 'MultiVector') -> 'None'
    def with_metric(self, metric: 'Metric') -> 'Vector'
    def normalize(self) -> 'Vector'
    def conjugate(self) -> 'Vector'
    def conj(self) -> 'Vector'
    def hodge(self) -> 'Vector'
    def span(self) -> 'tuple[Vector, ...]'
```

**Functions:**

```python
def basis_vector(index: 'int', metric: 'Metric') -> 'Vector'
def basis_vectors(metric: 'Metric') -> 'tuple[Vector, ...]'
def basis_element(indices: 'tuple[int, ...]', metric: 'Metric') -> 'Vector'
def geometric_basis(metric: 'Metric') -> 'dict[int, tuple[Vector, ...]]'
def pseudoscalar(metric: 'Metric') -> 'Vector'
```


### `morphis.elements.multivector`

```python
class MultiVector(CompositeElement):
    @property
    def is_even(self): ...
    @property
    def is_odd(self): ...
    @property
    def is_rotor(self): ...
    @property
    def is_motor(self): ...
    @property
    def dim(self): ...
    @property
    def grades(self): ...
    def __init__(self, *vectors, **kwargs)
    def grade_select(self, k: 'int') -> 'Vector | None'
    def reverse(self) -> 'MultiVector'
    def rev(self) -> 'MultiVector'
    def inverse(self) -> 'MultiVector'
    def inv(self) -> 'MultiVector'
    def with_metric(self, metric: 'Metric') -> 'MultiVector'
```


### `morphis.elements.frame`

```python
class Frame(GradedElement):
    @property
    def dim(self): ...
    @property
    def ndim(self): ...
    @property
    def shape(self): ...
    def __init__(self, *args, **kwargs)
    def vector(self, i: 'int') -> 'Vector'
    def as_vector(self) -> 'Vector'
    def transform(self, M: 'MultiVector') -> 'Frame'
    def transform_inplace(self, M: 'MultiVector') -> 'None'
    def to_vector(self) -> 'Vector'
    def normalize(self) -> 'Frame'
    def with_metric(self, metric: 'Metric') -> 'Frame'
```


### `morphis.elements.protocols`

```python
class Graded(Protocol):
    @property
    def grade(self): ...
    @property
    def data(self): ...
    def __init__(self, *args, **kwargs)
```

```python
class Spanning(Protocol):
    @property
    def span(self): ...
    def __init__(self, *args, **kwargs)
```

```python
class Transformable(Protocol):
    def transform(self, motor: 'MultiVector') -> 'None'
    def __init__(self, *args, **kwargs)
```

---

## Operations

Geometric algebra operations and products.

### `morphis.operations.operator`

```python
class Operator:
    @property
    def shape(self): ...
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
    def apply(self, x: vector.Vector) -> vector.Vector
    def apply_frame(self, f) -> 'Frame'
    def adjoint(self) -> 'Operator'
    def adj(self) -> 'Operator'
    def transpose(self) -> 'Operator'
    def trans(self) -> 'Operator'
    def solve(self, y: vector.Vector, method: 'lstsq', 'pinv' = 'lstsq', alpha: float = 0.0, r_cond: float | None = None) -> vector.Vector
    def pseudoinverse(self, r_cond: float | None = None) -> 'Operator'
    def pinv(self, r_cond: float | None = None) -> 'Operator'
    def svd(self) -> tuple['Operator', NDArray, numpy.dtype[~_ScalarT]], 'Operator']
    def compose(self, other: 'Operator') -> 'Operator'
```


### `morphis.operations.products`

**Functions:**

```python
def wedge(*elements)
def antiwedge(*elements)
def geometric(u, v)
def grade_project(M: 'MultiVector', k: 'int') -> 'Vector'
def scalar_product(u: 'Vector', v: 'Vector') -> 'NDArray'
def commutator(u: 'Vector', v: 'Vector') -> 'MultiVector'
def anticommutator(u: 'Vector', v: 'Vector') -> 'MultiVector'
def reverse(u: 'Vector | MultiVector') -> 'Vector | MultiVector'
def inverse(u: 'Vector | MultiVector') -> 'Vector | MultiVector'
```


### `morphis.operations.projections`

**Functions:**

```python
def interior_left(u: 'Vector', v: 'Vector') -> 'Vector'
def interior_right(u: 'Vector', v: 'Vector') -> 'Vector'
def dot(u: 'Vector', v: 'Vector') -> 'NDArray'
def project(u: 'Vector', v: 'Vector') -> 'Vector'
def reject(u: 'Vector', v: 'Vector') -> 'Vector'
def interior(u: 'Vector', v: 'Vector') -> 'Vector'
```


### `morphis.operations.duality`

**Functions:**

```python
def right_complement(u: 'Vector') -> 'Vector'
def left_complement(u: 'Vector') -> 'Vector'
def hodge_dual(u: 'Vector') -> 'Vector'
```


### `morphis.operations.norms`

**Functions:**

```python
def norm(u: 'Vector') -> 'NDArray'
def norm_squared(u: 'Vector') -> 'NDArray'
def normalize(u: 'Vector') -> 'Vector'
def conjugate(u: 'Vector') -> 'Vector'
def hermitian_norm(u: 'Vector') -> 'NDArray'
def hermitian_norm_squared(u: 'Vector') -> 'NDArray'
```


### `morphis.operations.exponential`

**Functions:**

```python
def exp_vector(B: 'Vector') -> 'MultiVector'
def log_versor(M: 'MultiVector') -> 'Vector'
def slerp(R0: 'MultiVector', R1: 'MultiVector', t: 'float | NDArray') -> 'MultiVector'
```


### `morphis.operations.structure`

**Functions:**

```python
def permutation_sign(perm: tuple[int, ...]) -> int
def antisymmetrize(tensor: NDArray, numpy.dtype[~_ScalarT]], k: int, cdim: int = 0) -> NDArray, numpy.dtype[~_ScalarT]]
def antisymmetric_symbol(k: int, d: int) -> NDArray, numpy.dtype[~_ScalarT]]
def levi_civita(d: int) -> NDArray, numpy.dtype[~_ScalarT]]
def generalized_delta(k: int, d: int) -> NDArray, numpy.dtype[~_ScalarT]]
def wedge_signature(grades: tuple[int, ...]) -> str
def wedge_normalization(grades: tuple[int, ...]) -> float
def interior_left_signature(j: int, k: int) -> str
def interior_right_signature(j: int, k: int) -> str
def complement_signature(k: int, d: int) -> str
def norm_squared_signature(k: int) -> str
def geometric_signature(j: int, k: int, c: int) -> str
def geometric_normalization(j: int, k: int, c: int) -> float
def interior_signature(j: int, k: int) -> str
```


### `morphis.operations.factorization`

**Functions:**

```python
def factor(b: 'Vector', tol: 'float | None' = None) -> 'Vector'
def spanning_vectors(b: 'Vector', tol: 'float | None' = None) -> 'tuple[Vector, ...]'
```


### `morphis.operations.spectral`

**Functions:**

```python
def bivector_to_skew_matrix(b: 'Vector') -> 'NDArray'
def bivector_eigendecomposition(b: 'Vector', tol: 'float | None' = None) -> 'tuple[NDArray, list[Vector]]'
def principal_vectors(b: 'Vector', tol: 'float | None' = None) -> 'tuple[Vector, ...]'
```


### `morphis.operations.outermorphism`

**Functions:**

```python
def apply_exterior_power(A: "'Operator'", blade: "'Vector'", k: 'int') -> "'Vector'"
def apply_outermorphism(A: "'Operator'", M: "'MultiVector'") -> "'MultiVector'"
```


### `morphis.operations.matrix_rep`

**Functions:**

```python
def vector_to_array(b: 'Vector') -> 'NDArray'
def vector_to_vector(v: 'NDArray', grade: 'int', metric: 'Metric') -> 'Vector'
def multivector_to_array(M: 'MultiVector') -> 'NDArray'
def array_to_multivector(v: 'NDArray', metric: 'Metric') -> 'MultiVector'
def left_matrix(A: 'Vector | MultiVector') -> 'NDArray'
def right_matrix(A: 'Vector | MultiVector') -> 'NDArray'
def operator_to_matrix(L: 'Operator') -> 'NDArray'
```

---

## Algebra

Linear algebra utilities for operators.

### `morphis.algebra.specs`

```python
class VectorSpec(BaseModel):
    @property
    def geometric_shape(self): ...
    @property
    def total_axes(self): ...
    def vector_shape(self, collection_shape: tuple[int, ...]) -> tuple[int, ...]
    def __init__(self, /, **data: 'Any') -> 'None'
```

**Functions:**

```python
def vector_spec(grade: int, dim: int, collection: int = 1) -> specs.VectorSpec
```


### `morphis.algebra.patterns`

**Functions:**

```python
def operator_shape(input_spec: specs.VectorSpec, output_spec: specs.VectorSpec, input_collection: tuple[int, ...], output_collection: tuple[int, ...]) -> tuple[int, ...]
```


### `morphis.algebra.solvers`

**Functions:**

```python
def structured_lstsq(op: 'Operator', target: vector.Vector, alpha: float = 0.0) -> vector.Vector
def structured_pinv_solve(op: 'Operator', target: vector.Vector, r_cond: float | None = None) -> vector.Vector
def structured_pinv(op: 'Operator', r_cond: float | None = None) -> 'Operator'
def structured_svd(op: 'Operator') -> tuple['Operator', NDArray, numpy.dtype[~_ScalarT]], 'Operator']
```

---

## Transforms

Geometric transformations.

### `morphis.transforms.rotations`

**Functions:**

```python
def rotor(B: 'Vector', angle: 'float | NDArray') -> 'MultiVector'
def rotation_about_point(p: 'Vector', B: 'Vector', angle: 'float | NDArray') -> 'MultiVector'
```


### `morphis.transforms.actions`

**Functions:**

```python
def rotate(b: 'Vector', B: 'Vector', angle: 'float | NDArray') -> 'Vector'
def translate(u: 'Vector', v: 'Vector') -> 'Vector'
def transform(b: 'Vector', M: 'MultiVector') -> 'Vector'
```


### `morphis.transforms.projective`

**Functions:**

```python
def point(x: 'NDArray', metric: 'Metric | None' = None, collection: 'tuple[int, ...] | None' = None) -> 'Vector'
def direction(v: 'NDArray', metric: 'Metric | None' = None, collection: 'tuple[int, ...] | None' = None) -> 'Vector'
def weight(p: 'Vector') -> 'NDArray'
def bulk(p: 'Vector') -> 'NDArray'
def euclidean(p: 'Vector') -> 'NDArray'
def is_point(p: 'Vector') -> 'NDArray'
def is_direction(p: 'Vector') -> 'NDArray'
def line(p: 'Vector', q: 'Vector') -> 'Vector'
def plane(p: 'Vector', q: 'Vector', r: 'Vector') -> 'Vector'
def plane_from_point_and_line(p: 'Vector', l: 'Vector') -> 'Vector'
def distance_point_to_point(p: 'Vector', q: 'Vector') -> 'NDArray'
def distance_point_to_line(p: 'Vector', l: 'Vector') -> 'NDArray'
def distance_point_to_plane(p: 'Vector', h: 'Vector') -> 'NDArray'
def are_collinear(p: 'Vector', q: 'Vector', r: 'Vector', tol: 'float | None' = None) -> 'NDArray'
def are_coplanar(p: 'Vector', q: 'Vector', r: 'Vector', s: 'Vector', tol: 'float | None' = None) -> 'NDArray'
def point_on_line(p: 'Vector', l: 'Vector', tol: 'float | None' = None) -> 'NDArray'
def point_on_plane(p: 'Vector', h: 'Vector', tol: 'float | None' = None) -> 'NDArray'
def line_in_plane(l: 'Vector', h: 'Vector', tol: 'float | None' = None) -> 'NDArray'
def translator(v: 'Vector') -> 'MultiVector'
def screw(B: 'Vector', angle: 'float | NDArray', translation: 'Vector', center: 'Vector | None' = None) -> 'MultiVector'
```

---

## Visualization

Visualization and rendering tools.

### `morphis.visuals.canvas`

```python
class ArrowStyle(BaseModel):
    def __init__(self, /, **data: 'Any') -> 'None'
```

```python
class CurveStyle(BaseModel):
    def __init__(self, /, **data: 'Any') -> 'None'
```

```python
class Canvas:
    def __init__(self, theme: str | theme.Theme = Theme(name='obsidian', background=(0.12, 0.13, 0.14), e1=(0.85, 0.35, 0.3), e2=(0.4, 0.75, 0.45), e3=(0.35, 0.5, 0.9), palette=Palette(colors=((0.95, 0.55, 0.45), (0.55, 0.8, 0.7), (0.9, 0.75, 0.4), (0.5, 0.65, 0.9), (0.85, 0.5, 0.7), (0.45, 0.8, 0.85))), accent=(0.95, 0.85, 0.4), muted=(0.45, 0.47, 0.5), label=(0.82, 0.84, 0.86)), title: str | None = None, size: tuple[int, int] = (1200, 900), show_basis: bool = True, basis_axes: tuple[int, int, int] = (0, 1, 2))
    def basis(self, scale: float = 1.0, axes: tuple[int, int, int] = (0, 1, 2), labels: bool = True)
    def set_basis_axes(self, axes: tuple[int, int, int])
    def arrow(self, start, direction, color: tuple[float, float, float] | None = None, shaft_radius: float | None = None)
    def arrows(self, starts, directions, color: tuple[float, float, float] | None = None, colors: theme.Palette | list | None = None, shaft_radius: float | None = None)
    def curve(self, points, color: tuple[float, float, float] | None = None, radius: float | None = None, closed: bool = False)
    def curves(self, point_sets, color: tuple[float, float, float] | None = None, colors: theme.Palette | list | None = None, radius: float | None = None)
    def point(self, position, color: tuple[float, float, float] | None = None, radius: float = 0.03)
    def points(self, positions, color: tuple[float, float, float] | None = None, colors: theme.Palette | list | None = None, radius: float = 0.03)
    def plane(self, center, normal, size: float = 1.0, color: tuple[float, float, float] | None = None, opacity: float = 0.3)
    def reset_colors(self)
    def camera(self, position: tuple[float, float, float] | None = None, focal_point: tuple[float, float, float] | None = None, view_up: tuple[float, float, float] | None = None)
    def show(self, block: bool = True)
    def screenshot(self, filename: str, scale: int = 2)
    def close(self)
```


### `morphis.visuals.theme`

```python
class Palette(BaseModel):
    def cycle(self, n: int) -> list[tuple[float, float, float]]
    def __init__(self, /, **data: 'Any') -> 'None'
```

```python
class Theme(BaseModel):
    @property
    def basis_colors(self): ...
    @property
    def axis_color(self): ...
    @property
    def grid_color(self): ...
    @property
    def text_color(self): ...
    @property
    def edge_color(self): ...
    def is_light(self) -> bool
    def __init__(self, /, **data: 'Any') -> 'None'
```

**Functions:**

```python
def get_theme(name: str) -> theme.Theme
def list_themes() -> list[str]
```


### `morphis.visuals.projection`

```python
class ProjectionConfig(BaseModel):
    def __init__(self, /, **data: 'Any') -> 'None'
```

**Functions:**

```python
def project_vector(blade: vector.Vector, config: projection.ProjectionConfig) -> vector.Vector
def project_bivector(blade: vector.Vector, config: projection.ProjectionConfig) -> vector.Vector
def project_trivector(blade: vector.Vector, config: projection.ProjectionConfig) -> vector.Vector
def project_quadvector(blade: vector.Vector, config: projection.ProjectionConfig) -> vector.Vector
def project_blade(blade: vector.Vector, config: projection.ProjectionConfig | None = None) -> vector.Vector
def get_projection_axes(blade: vector.Vector, config: projection.ProjectionConfig | None = None) -> tuple[int, ...]
```


### `morphis.visuals.drawing.vectors`

```python
class VectorStyle(BaseModel):
    def __init__(self, /, **data: 'Any') -> 'None'
```

**Functions:**

```python
def create_vector_mesh(origin: numpy.ndarray, direction: numpy.ndarray, shaft_radius: float = 0.008) -> tuple[pyvista.core.pointset.PolyData | None, pyvista.core.pointset.PolyData]
def create_bivector_mesh(origin: numpy.ndarray, u: numpy.ndarray, v: numpy.ndarray, shaft_radius: float = 0.006, face_opacity: float = 0.25) -> tuple[pyvista.core.pointset.PolyData, pyvista.core.pointset.PolyData, pyvista.core.pointset.PolyData]
def create_trivector_mesh(origin: numpy.ndarray, u: numpy.ndarray, v: numpy.ndarray, w: numpy.ndarray, shaft_radius: float = 0.006, face_opacity: float = 0.15) -> tuple[pyvista.core.pointset.PolyData, pyvista.core.pointset.PolyData, pyvista.core.pointset.PolyData]
def create_quadvector_mesh(origin: numpy.ndarray, u: numpy.ndarray, v: numpy.ndarray, w: numpy.ndarray, x: numpy.ndarray, projection_axes: tuple[int, int, int] = (0, 1, 2), shaft_radius: float = 0.006, face_opacity: float = 0.12) -> tuple[pyvista.core.pointset.PolyData, pyvista.core.pointset.PolyData, pyvista.core.pointset.PolyData]
def create_frame_mesh(origin: numpy.ndarray, vectors: numpy.ndarray, shaft_radius: float = 0.008, projection_axes: tuple[int, int, int] | None = None, filled: bool = False) -> tuple[pyvista.core.pointset.PolyData | None, pyvista.core.pointset.PolyData | None, pyvista.core.pointset.PolyData]
def create_blade_mesh(grade: int, origin: numpy.ndarray, vectors: numpy.ndarray, shaft_radius: float = 0.008, edge_radius: float = 0.006, projection_axes: tuple[int, int, int] | None = None) -> tuple[pyvista.core.pointset.PolyData | None, pyvista.core.pointset.PolyData | None, pyvista.core.pointset.PolyData]
def draw_blade(plotter: pyvista.plotting.plotter.Plotter, b: vector.Vector, p: vector.Vector | numpy.ndarray | tuple = None, color: tuple[float, float, float] = (0.85, 0.85, 0.85), tetrad: bool = True, surface: bool = True, label: bool = False, name: str | None = None, shaft_radius: float = 0.008, edge_radius: float = 0.006, label_offset: float = 0.08)
def draw_coordinate_basis(plotter: pyvista.plotting.plotter.Plotter, scale: float = 1.0, color: tuple[float, float, float] = (0.85, 0.85, 0.85), tetrad: bool = True, surface: bool = False, label: bool = True, label_offset: float = 0.08, labels: tuple[str, str, str] | None = None) -> list | None
def draw_basis_blade(plotter: pyvista.plotting.plotter.Plotter, indices: tuple[int, ...], position: numpy.ndarray | tuple = (0, 0, 0), scale: float = 1.0, color: tuple[float, float, float] = (0.85, 0.85, 0.85), tetrad: bool = True, surface: bool = True, label: bool = False, name: str | None = None, label_offset: float = 0.08)
def render_scalar(blade: vector.Vector, canvas, position: tuple[float, float, float] | None = None, style: drawing.vectors.VectorStyle | None = None) -> None
def render_vector(blade: vector.Vector, canvas, projection=None, style: drawing.vectors.VectorStyle | None = None) -> None
def render_bivector(blade: vector.Vector, canvas, mode: 'circle', 'parallelogram', 'plane', 'circular_arrow' = 'circle', projection=None, style: drawing.vectors.VectorStyle | None = None) -> None
def render_trivector(blade: vector.Vector, canvas, mode: 'parallelepiped', 'sphere' = 'parallelepiped', projection=None, style: drawing.vectors.VectorStyle | None = None) -> None
def visualize_blade(blade: vector.Vector, canvas=None, mode: str = 'auto', projection=None, show_dual: bool = False, style: drawing.vectors.VectorStyle | None = None)
def visualize_blades(blades: list, canvas=None, style: drawing.vectors.VectorStyle | None = None)
```


### `morphis.visuals.contexts`

```python
class PGAStyle(VectorStyle):
    def __init__(self, /, **data: 'Any') -> 'None'
```

**Functions:**

```python
def render_pga_point(blade: vector.Vector, canvas: canvas.Canvas, style: contexts.PGAStyle | None = None) -> None
def render_pga_line(blade: vector.Vector, canvas: canvas.Canvas, style: contexts.PGAStyle | None = None) -> None
def render_pga_plane(blade: vector.Vector, canvas: canvas.Canvas, style: contexts.PGAStyle | None = None) -> None
def is_pga_context(vec: vector.Vector) -> bool
def visualize_pga_blade(blade: vector.Vector, canvas: canvas.Canvas | None = None, style: contexts.PGAStyle | None = None) -> canvas.Canvas
def visualize_pga_scene(*blades: vector.Vector, canvas: canvas.Canvas | None = None, style: contexts.PGAStyle | None = None) -> canvas.Canvas
```


### `morphis.visuals.operations`

```python
class OperationStyle(BaseModel):
    def __init__(self, /, **data: 'Any') -> 'None'
```

**Functions:**

```python
def render_join(u: vector.Vector, v: vector.Vector, canvas: canvas.Canvas | None = None, projection: projection.ProjectionConfig | None = None, style: operations.OperationStyle | None = None) -> canvas.Canvas
def render_meet(u: vector.Vector, v: vector.Vector, canvas: canvas.Canvas | None = None, projection: projection.ProjectionConfig | None = None, style: operations.OperationStyle | None = None) -> canvas.Canvas
def render_meet_join(u: vector.Vector, v: vector.Vector, canvas: canvas.Canvas | None = None, show: 'meet', 'join', 'both' = 'both', projection: projection.ProjectionConfig | None = None, style: operations.OperationStyle | None = None) -> canvas.Canvas
def render_with_dual(blade: vector.Vector, canvas: canvas.Canvas | None = None, dual_type: 'right', 'left', 'hodge' = 'right', projection: projection.ProjectionConfig | None = None, style: operations.OperationStyle | None = None) -> canvas.Canvas
```


### `morphis.visuals.loop`

```python
class Snapshot(BaseModel):
    def __init__(self, /, **data: 'Any') -> 'None'
```

```python
class AnimationTrack(BaseModel):
    def __init__(self, /, **data: 'Any') -> 'None'
```

```python
class Animation:
    @property
    def observer(self): ...
    def __init__(self, frame_rate: int = 60, theme: str | theme.Theme = 'obsidian', size: tuple[int, int] = (1800, 1350), show_basis: bool = True, auto_camera: bool = True, fps: int | None = None)
    def watch(self, *targets: vector.Vector | frame.Frame, color: tuple[float, float, float] | None = None, filled: bool = False) -> int | list[int]
    def unwatch(self, *targets: vector.Vector | frame.Frame)
    def set_vectors(self, blade: vector.Vector, vectors: NDArray, numpy.dtype[~_ScalarT]], origin: NDArray, numpy.dtype[~_ScalarT]] | None = None)
    def set_projection(self, axes: tuple[int, int, int], labels: tuple[str, str, str] | None = None)
    def fade_in(self, target: vector.Vector | frame.Frame, t: float, duration: float)
    def fade_out(self, target: vector.Vector | frame.Frame, t: float, duration: float)
    def start(self, live: bool = False)
    def capture(self, t: float)
    def finish(self)
    def play(self, loop: bool = False)
    def save(self, filename: str, loop: bool = True)
    def camera(self, position=None, focal_point=None)
    def set_basis_labels(self, labels: tuple[str, str, str])
    def close(self)
    def track(self, *targets: vector.Vector | frame.Frame, color: tuple[float, float, float] | None = None, filled: bool = False) -> int | list[int]
    def untrack(self, *targets: vector.Vector | frame.Frame)
```

---

## Utilities

Helper utilities.

### `morphis.utils.pretty`

**Functions:**

```python
def format_matrix(arr: numpy.ndarray, precision: int = 4) -> str
def print_matrix(arr: numpy.ndarray, precision: int = 4) -> None
def section(title: str, width: int = 70) -> None
def subsection(title: str) -> None
def show_vec(name: str, blade: vector.Vector, precision: int = 4) -> None
def show_array(name: str, arr, precision: int = 4) -> None
def show_scalar(name: str, value, precision: int = 4) -> None
def show_mv(name: str, M: multivector.MultiVector, precision: int = 4) -> None
```


### `morphis.utils.observer`

```python
class TrackedObject(BaseModel):
    def __init__(self, /, **data: 'Any') -> 'None'
```

```python
class Observer:
    def __init__(self)
    def watch(self, *objects: base.Element, names: list[str] | None = None) -> 'Observer'
    def unwatch(self, *objects: base.Element) -> 'Observer'
    def clear(self) -> 'Observer'
    def get(self, obj_or_name: base.Element | str) -> numpy.ndarray | None
    def snapshot(self) -> dict[int, numpy.ndarray]
    def snapshot_named(self) -> dict[str, numpy.ndarray]
    def reset_baseline(self, *objects: base.Element) -> 'Observer'
    def diff(self, obj_or_name: base.Element | str) -> numpy.ndarray | None
    def diff_norm(self, obj_or_name: base.Element | str) -> float | None
    def objects(self) -> list[base.Element]
    def ids(self) -> list[int]
    def names(self) -> list[str]
    def print_state(self, prefix: str = '')
    def spanning_vectors(self, obj_or_name: base.Element | str) -> tuple['Vector', ...] | None
    def spanning_vectors_as_array(self, obj_or_name: base.Element | str) -> numpy.ndarray | None
    def capture_state(self, obj_or_name: base.Element | str) -> dict | None
    def track(self, *objects: base.Element, names: list[str] | None = None) -> 'Observer'
    def untrack(self, *objects: base.Element) -> 'Observer'
```

