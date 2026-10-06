# Outermorphisms and Operator Extension

## The Mathematical Foundation

An **outermorphism** (or exomorphism) is a linear map $f: \bigwedge V \to \bigwedge W$ between exterior algebras that preserves the wedge product structure:

$$f(\mathbf{a} \wedge \mathbf{b}) = f(\mathbf{a}) \wedge f(\mathbf{b})$$

An outermorphism is **completely determined** by its action on grade-1 elements (vectors). Given a linear map $A: V \to W$ on vectors, the wedge-preservation rule forces the extension on a $k$-blade to be the wedge of the images:

$$f(\mathbf{v}_1 \wedge \mathbf{v}_2 \wedge \cdots \wedge \mathbf{v}_k) = A(\mathbf{v}_1) \wedge A(\mathbf{v}_2) \wedge \cdots \wedge A(\mathbf{v}_k)$$

This extended map is called the **$k$-th exterior power** of $A$, written $\bigwedge^k A$ or sometimes $A^{\wedge k}$.

## The Exterior Power in Components

For a linear map $A: V \to W$ represented in bases as the matrix $A^p{}_q$, the action on vectors is:

$$A(\mathbf{v})^p = A^p{}_q \, v^q$$

The extension to grade-2 (bivectors) follows from the wedge-preservation property. For a bivector $\mathbf{b} = \mathbf{u} \wedge \mathbf{v}$:

$$\begin{align}
(\bigwedge^2 A)(\mathbf{b})^{mn}
    &= (A\mathbf{u} \wedge A\mathbf{v})^{mn} \\ \\
    &= (A\mathbf{u})^m (A\mathbf{v})^n - (A\mathbf{u})^n (A\mathbf{v})^m \\ \\
    &= A^m{}_p \, u^p \, A^n{}_q \, v^q - A^n{}_p \, u^p \, A^m{}_q \, v^q \\ \\
    &= A^m{}_p \, A^n{}_q \, (u^p v^q - u^q v^p) \\ \\
    &= A^m{}_p \, A^n{}_q \, b^{pq}
\end{align}$$

The pattern generalizes naturally. For a $k$-blade with components $b^{m_1 m_2 \cdots m_k}$:

$$(\bigwedge^k A)(\mathbf{b})^{n_1 n_2 \cdots n_k} = A^{n_1}{}_{m_1} \, A^{n_2}{}_{m_2} \cdots A^{n_k}{}_{m_k} \, b^{m_1 m_2 \cdots m_k}$$

The action is $k$ copies of $A$ contracting against the $k$ indices of the blade. The antisymmetry of $b$ propagates through to the output; no explicit antisymmetrization is required when the input is already a blade.

## The Exterior Power as Matrix

When working with the **independent** components of $k$-blades (the $\binom{d}{k}$ basis elements), the exterior power $\bigwedge^k A$ becomes a $\binom{d}{k} \times \binom{d}{k}$ matrix whose entries are $k \times k$ minors of $A$:

$$[\bigwedge^k A]_{PQ} = \det(A_{P,Q})$$

where $P = (p_1, \ldots, p_k)$ and $Q = (q_1, \ldots, q_k)$ are ordered multi-indices, and $A_{P,Q}$ is the $k \times k$ submatrix of $A$ with rows $P$ and columns $Q$.

For $k = d$ (the pseudoscalar), this reduces to a $1 \times 1$ matrix containing $\det(A)$:

$$\bigwedge^d A = \det(A)$$

The action on the pseudoscalar is multiplication by the determinant, which is the algebraic statement of the determinant's geometric role as a volume scaling factor.

## Scalars Under Outermorphisms

Scalars (grade-0 elements) are invariant under outermorphisms:

$$(\bigwedge^0 A)(s) = s$$

The convention $\bigwedge^0 V = \mathbb{R}$ identifies the grade-zero piece with the scalar field, and any linear map fixes scalars. Geometrically, scalars carry no directional content, so the linear transformation has nothing directional to change.

## Design: Store the Vector Map, Compute Extensions on Demand

Three ways to give `Operator` an outermorphism action were considered:

- **Store only the grade-1 map** and compute $\bigwedge^k A$ when applying to a grade-$k$ element.
- **Store the full block-diagonal map** on the $2^d$-dimensional multivector space.
- **Store the grade-1 map and cache** each $\bigwedge^k A$ as it is first needed.

Morphis stores only the grade-1 map and computes extensions on demand:

1. **Storage**: a $d \times d$ matrix instead of a $2^d \times 2^d$ block-diagonal map
2. **Algebraic clarity**: the outermorphism *is* defined by its grade-1 action
3. **Composition**: $(f \circ g)|_{\text{grade-1}} = f|_{\text{grade-1}} \circ g|_{\text{grade-1}}$, so composing outermorphisms is matrix multiplication
4. **Redundant tensor storage**: a grade-$k$ `Vector` stores the full antisymmetric tensor of shape $(d,)^k$, so applying $k$ copies of $A$ is a single einsum with no minor computation

## Implementation

### Operator Layout

`Operator` (`operations/operator.py`) stores a linear map between `Vector` spaces described by two `VectorSpec`s:

```python
class Operator:
    data: NDArray            # (*out_lot, *in_lot, *out_geo, *in_geo)
    input_spec: VectorSpec   # grade, lot, dim
    output_spec: VectorSpec  # grade, lot, dim
    metric: Metric
```

For a grade-1 to grade-1 operator with no lot dimensions, `data` is the $d \times d$ matrix $A^p{}_q$, output index first.

### Recognizing Outermorphisms

An operator acts as an outermorphism when it maps grade 1 to grade 1:

```python
@property
def is_outermorphism(self) -> bool:
    return self.input_spec.grade == 1 and self.output_spec.grade == 1
```

`Operator.vector_map` returns the $d \times d$ matrix. It raises `ValueError` for any other grade pair and `NotImplementedError` when either spec has lot dimensions.

### Dispatch in `Operator.__mul__`

| Expression | Behavior |
|------------|----------|
| `L * v`, `v.grade == input_spec.grade` | `L.apply(v)` |
| `L * v`, other grade, `L.is_outermorphism` | `apply_exterior_power(L, v, v.grade)` |
| `L * v`, other grade, not an outermorphism | `ValueError` |
| `L * M` (`MultiVector`) | `apply_outermorphism(L, M)` |
| `L * F` (`Frame`) | `L.apply_frame(F)` |
| `L1 * L2` | `L1.compose(L2)` |
| `L * s` (number or lot-free grade-0 `Vector`) | scaled operator |

### Exterior Power (`operations/outermorphism.py`)

`exterior_power_signature(k)` builds the einsum signature from the index pools in `algebra/patterns.py` (`OUTPUT_GEOMETRIC = "WXYZ"`, `INPUT_GEOMETRIC = "abcd"`), so grades up to 4 are supported:

```python
exterior_power_signature(1)  # 'Wa,...a->...W'
exterior_power_signature(2)  # 'Wa,Xb,...ab->...WX'
exterior_power_signature(3)  # 'Wa,Xb,Yc,...abc->...WXY'
```

The ellipsis carries the input element's lot dimensions through unchanged. Signatures are cached with `lru_cache`.

`apply_exterior_power(A, blade, k)` checks that `A` is grade 1 to grade 1 and that `blade.grade == k`, then:

- $k = 0$: returns a copy (scalars are invariant)
- $k = 1$: returns `A.apply(blade)`
- $k \geq 2$: `einsum(exterior_power_signature(k), *([A.vector_map] * k), blade.data)`, wrapped as a grade-$k$ `Vector` with `A.metric`

### Applying to a MultiVector

`apply_outermorphism(A, M)` raises `TypeError` unless `A` is grade 1 to grade 1, and `NotImplementedError` if `A` has lot dimensions. It then maps each grade independently:

```python
result = {k: apply_exterior_power(A, blade, k) for k, blade in M.data.items()}
return MultiVector(data=result, metric=A.metric)
```

### Index Convention

The exterior power contracts raw storage axes. The geometric index convention (Euclidean indices from 1, Lorentzian and PGA from 0; see [Index Convention](6_index-convention.md)) is a translation applied at the user-facing boundary, in `basis_vector`, `.on[...]`, and projection, so the einsum above never sees it. The components $b^{m_1 \cdots m_k}$ in the formulas are internal slots; they coincide with the user-facing indices only up to the metric's base offset.

## MultiVector Storage: Why a Dict Remains Appropriate

`MultiVector` stores its components as a dict from grade to `Vector`:

```python
class MultiVector:
    data: dict[int, Vector]  # grade -> Vector
```

**Advantages for outermorphism application:**

1. **Grade iteration**: application loops over the grades that are present
2. **Sparsity**: rotors have only grades {0, 2}, with no storage for the others
3. **Direct einsum**: each `Vector` already holds tensor data ready for contraction

**A block-diagonal array representation would require:**

1. Basis selection (which ordered $k$-tuples represent basis $k$-blades)
2. Index offset computation for each grade
3. Conversion between tensor and coefficient representations
4. Loss of sparsity

The dict representation matches the mathematics: a multivector is a formal sum of components of different grades.

## Lot Dimensions and Outermorphisms

A grade-changing operator is a general linear map, not an outermorphism:

```python
# Maps N scalars to M bivectors: data shape (M, N, d, d)
G = Operator(
    data=G_data,
    input_spec=VectorSpec(grade=0, lot=(N,), dim=d),
    output_spec=VectorSpec(grade=2, lot=(M,), dim=d),
    metric=euclidean_metric(d),
)
```

A grade-1 to grade-1 operator with lot dimensions:

```python
# data shape (M, N, d, d): (out_lot, in_lot, out_geo, in_geo)
L = Operator(
    data=L_data,
    input_spec=VectorSpec(grade=1, lot=(N,), dim=d),
    output_spec=VectorSpec(grade=1, lot=(M,), dim=d),
    metric=euclidean_metric(d),
)
```

Acting on a grade-$k$ element of shape `(N,) + (d,) * k`, its exterior power should produce shape `(M,) + (d,) * k`: the $k$ copies of $A$ contract the geometric axes while the lot axes follow the operator's lot mapping. `L * v` works today when `v.grade == 1` (through `apply`), but the exterior power for other grades and `L * M` raise `NotImplementedError` because `vector_map` requires a lot-free operator. Supporting this needs an einsum that threads the operator's lot indices through each of the $k$ factors.

## API Summary

| Expression | Description | Requirements |
|------------|-------------|--------------|
| `L * v` | Apply to a `Vector` | Grade matches `input_spec`, or `L` is an outermorphism |
| `L * F` | Apply to a `Frame` | Grade 1 to grade 1 |
| `L * M` | Apply as outermorphism to a `MultiVector` | Grade 1 to grade 1, no lot dimensions |
| `L1 * L2` | Composition | Output of `L2` matches input of `L1` |
| `L.H` | Adjoint | Always available |
| `L.is_outermorphism` | Grade 1 to grade 1 check | Always available |
| `L.vector_map` | The $d \times d$ matrix | Grade 1 to grade 1, no lot dimensions |

## Example Usage

```python
from numpy import array, cos, pi, sin

from morphis.algebra import VectorSpec
from morphis.elements import MultiVector, Vector, basis_vectors, euclidean_metric
from morphis.operations import Operator

g = euclidean_metric(3)
e1, e2, e3 = basis_vectors(g)

theta = pi / 4
R = array([
    [cos(theta), -sin(theta), 0],
    [sin(theta), cos(theta), 0],
    [0, 0, 1],
])

L = Operator(
    data=R,
    input_spec=VectorSpec(grade=1, lot=(), dim=3),
    output_spec=VectorSpec(grade=1, lot=(), dim=3),
    metric=g,
)

v_rotated = L * e1        # grade 1, via apply
B_rotated = L * (e1 ^ e2) # grade 2, via the 2nd exterior power

s = Vector(2.0, grade=0, metric=g)
M = MultiVector(data={0: s, 1: e1, 2: e1 ^ e2}, metric=g)
M_rotated = L * M         # every grade mapped; the scalar is unchanged
```

`examples/operators.py` runs a longer version of this in its outermorphism section.

## Connection to Versors

The sandwich product $M \mathbf{x} \tilde{M}$ for a versor $M$ defines an outermorphism. The grade-1 action is:

$$\mathbf{v} \mapsto M \mathbf{v} \tilde{M}$$

The grade-1 action extends to all grades through the exterior power, and grade preservation under $M \mathbf{B} \tilde{M}$ is what identifies the sandwich product as an outermorphism whenever $M$ is a versor.

A constructor could build the operator from a versor:

```python
@classmethod
def from_versor(cls, M: MultiVector) -> Operator:
    """Outermorphism v -> M v ~M, from its action on basis vectors."""
    images = [(M * e * ~M)[1] for e in basis_vectors(M.metric)]
    A = stack([image.data for image in images], axis=-1)

    return cls(
        data=A,
        input_spec=VectorSpec(grade=1, lot=(), dim=M.dim),
        output_spec=VectorSpec(grade=1, lot=(), dim=M.dim),
        metric=M.metric,
    )
```

`basis_vectors` returns the basis in index order for every signature, so column $n$ of `A` is the image of the $n^{th}$ storage slot whatever the metric's base index.

## Summary

1. **Mathematical completeness**: linear maps on $V$ extend to $\bigwedge V$
2. **Unified API**: `L * M` works for any multivector when `L` is grade 1 to grade 1
3. **Computation on demand**: exterior powers are einsums, never precomputed
4. **Storage**: only the $d \times d$ grade-1 matrix is stored
5. **Composition**: outermorphism composition reduces to matrix multiplication

## Status

Implemented:

1. `Operator.is_outermorphism` and `Operator.vector_map`
2. `operations/outermorphism.py`: `exterior_power_signature`, `apply_exterior_power`, `apply_outermorphism`
3. `Operator.__mul__` dispatch for `MultiVector` and off-grade `Vector`
4. Tests in `operations/tests/test_outermorphism.py`
5. Outermorphism section in `examples/operators.py`

Future work:

1. `Operator.from_versor()` constructor
2. Outermorphisms of operators with lot dimensions
