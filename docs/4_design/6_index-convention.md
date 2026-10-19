# Geometric Index Convention

This document describes how geometric components are addressed in morphis, and
why the metric is the single authority for the convention.

## Motivation

The point of morphis is to let you write geometric algebra the way you write it
on paper. On paper, a purely geometric vector has components `v_1 = x`, `v_2 = y`,
`v_3 = z`, indexed from 1. A spacetime vector has a time component at index 0:
`v_0 = t`, `v_1 = x`. A Kähler space pairs each direction with its partner,
`v_1, v_1̇, v_2, v_2̇`. The labels are not a global preference; they are a
property of the space you are working in.

## Spaces Are Declared by Blocks

A space is declared by its blocks, and the block layout fixes both the storage
order and the label of every direction:

```python
metric(space=3)                          # Euclidean ℝ³
metric(space=3, projective=1)            # PGA
metric(time=1, space=3)                  # spacetime
metric(time=1, space=3, projective=1)    # spacetime PGA
metric(modes=2)                          # ℂ² as ℝ⁴
```

Storage follows (time, space, projective), except that a purely spatial space
with a single projective direction is stored (projective, space).

| Space | Order | Labels |
|-------|-------|--------|
| Euclidean | space | `1, ..., n` |
| PGA | projective, space | `0, 1, ..., n` |
| Spacetime | time, space | `0, 1, ..., n` |
| Spacetime PGA | time, space, projective | `0, 1, ..., n, n + 1` |
| Several projective | space, projective | `1, ..., n, n + 1, ...` |
| Kähler | modes | `1, Dot(1), 2, Dot(2), ...` |

Two invariants hold in every layout:

1. **The first spatial direction (x) is always index 1.**
2. **Every other direction is labeled by its position.** Time and a leading
   projective direction are 0; trailing projective directions continue the
   spatial count. A mode's partner is labeled `Dot(m)`, displayed as ṁ.

The convention is *forced*: a label the space does not carry raises
`IndexError` rather than resolving silently. One index has one meaning per
space.

## Signature and Convention

The sign convention is an argument of its own: `convention="mostly minus"`
(the default, (+, -, -, -)) or `"mostly plus"` ((-, +, +, +)). It only matters
when time is present; without time every spatial direction squares to +1.

`Metric.signature` is the inertia `(p, q, r)` of the tensor, the counts of
positive, negative, and zero eigenvalues, so it is correct for any symmetric
tensor. `Metric.convention` is read back from the sign of the time direction.
Two metrics are equal when their block layouts and tensors agree, so ℝ^{2,2}
and ℝ^{1,3} are different spaces, and so are ℝ⁴ and ℂ².

A metric that no factory produces, such as a non-diagonal tensor, is built
directly: `Metric(data=g, blocks=(("time", 1), ("space", 3)))`.

## The Metric is the Single Authority

Every geometric access point routes through the metric, so the convention lives
in exactly one place.

```python
class Metric:
    labels: tuple[int | Dot, ...]          # label of each storage slot
    kinds: tuple[str, ...]                 # block kind of each storage slot
    def to_internal(self, index) -> int: ...        # label -> storage slot
    def to_user(self, internal) -> int | Dot: ...   # storage slot -> label
    def to_internal_multi(self, indices) -> list: ...
```

The labels are derived from the block layout in one function,
`_block_labels`; nothing else in the package computes an index base.

### What routes through the metric

- **Basis construction**: `basis_vector`, `basis_vectors`, `basis_element`,
  `geometric_basis`, `pseudoscalar` take and emit labels.
  `basis_vector(1, metric(space=3))` is the x direction and
  `basis_vector(Dot(1), metric(modes=2))` is e_1̇.
- **`.on[...]`**: geometric component access. `v.on[1]` is the x component;
  `B.on[1, 2]` is the `e_12` component of a bivector, with antisymmetry
  (`B.on[2, 1] == -B.on[1, 2]`) preserved through the translation.
- **Visualization and projection**: `Scene(projection=...)`,
  `Scene.set_projection`, `ProjectionConfig.axes`, and `get_projection_axes`
  take and report labels. The default projection is `(1, 2, 3)`. Each element
  translates the axes through its own metric at projection time. Basis labels
  are built from the same indices, and `draw_blade` default names use
  `to_user`. Mesh helpers below this boundary take internal slots.
- **Display**: blade labels are produced via `to_user`, so nothing leaks a
  0-based storage index.

### What does NOT

- **Lot (collection) axes** are always standard 0-based Python indexing,
  consistent with NumPy. Labels are purely a geometric concern.
- **`.at[...]`**: lot-only access, unchanged and 0-based.
- **Plain `v[...]`**: raw positional slicing over the full `(*lot, *geo)` shape.
- **Contraction**: `contract` / `IndexedTensor` label raw array axes, which stay
  internal 0-based.
- **`metric[a, b]`**: indexes the tensor by storage slot.

## Deferred

The block layout leaves room for these without designing them now:

- **A second time direction**, labeled -1.
- **Conformal blocks**, whose null basis needs its own labels.
