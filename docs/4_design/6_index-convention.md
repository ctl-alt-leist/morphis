# Geometric Index Convention

This document describes how geometric components are addressed in morphis, and
why the metric is the single authority for the convention.

## Motivation

The point of morphis is to let you write geometric algebra the way you write it
on paper. On paper, a purely geometric vector has components `v_1 = x`, `v_2 = y`,
`v_3 = z` — indexed from 1. A spacetime vector has a time component at index 0:
`v_0 = t`, `v_1 = x`. The index base is not a global preference; it is a property
of the space you are working in.

## The Convention

Storage is always 0-based internally (a NumPy array). The user-facing geometric
index is translated to and from the internal slot by the metric, and the base
index is derived from the signature:

| Signature | Base index | Index 0 | First spatial |
|-----------|------------|---------|---------------|
| Euclidean | 1 | forbidden (raises) | `e_1 = x` |
| Lorentzian | 0 | time | `e_1 = x` |
| Degenerate (PGA) | 0 | ideal/null direction | `e_1 = x` |

Two invariants hold across every signature:

1. **The first spatial direction (x) is always index 1.** Only the presence of a
   distinguished 0th direction changes — time in Lorentzian, the ideal/null
   direction in PGA.
2. **The base index is 0 exactly when the algebra carries a distinguished
   non-spatial 0th direction.** A purely spatial Euclidean algebra starts at 1,
   and its index 0 is forbidden.

The convention is *forced*, not optional: an out-of-range geometric index raises
`IndexError` rather than silently resolving. One index has one meaning per
signature.

## The Metric is the Single Authority

Every geometric access point routes through the metric, so the convention lives
in exactly one place.

```python
class Metric:
    @property
    def base_index(self) -> int: ...       # first valid geometric index
    @property
    def max_index(self) -> int: ...        # last valid geometric index (inclusive)
    def to_internal(self, index) -> int: ...        # physics index -> storage slot
    def to_user(self, internal) -> int: ...         # storage slot -> physics index
    def to_internal_multi(self, indices) -> list: ...
```

The signature → base mapping is isolated in the module-level table
`_SIGNATURE_BASE_INDEX`. This is deliberate: a future flexible layout (see below)
replaces that one table without touching any call site.

### What routes through the metric

- **Basis construction** — `basis_vector`, `basis_vectors`, `basis_element`,
  `geometric_basis`, `pseudoscalar` take and emit user-facing geometric indices.
  `basis_vector(1, euclidean_metric(3))` is the x direction.
- **`.on[...]`** — geometric component access. `v.on[1]` is the x component;
  `B.on[1, 2]` is the `e_12` component of a bivector, with antisymmetry
  (`B.on[2, 1] == -B.on[1, 2]`) preserved through the translation.
- **Display** — when a labeled printer (`3 e_1 + ...`) is added, blade labels
  should be produced via `to_user` so nothing leaks a 0-based storage index. The
  current printer shows the raw component array and needs no translation.

### What does NOT

- **Lot (collection) axes** are always standard 0-based Python indexing,
  consistent with NumPy. The index base is purely a geometric concern.
- **`.at[...]`** — lot-only access, unchanged and 0-based.
- **Plain `v[...]`** — raw positional slicing over the full `(*lot, *geo)` shape,
  unchanged.
- **Contraction** — `contract` / `IndexedTensor` label raw array axes, which stay
  internal 0-based. The base is a boundary translation; the core array math never
  sees it.

## Deferred: Flexible Index Layout

The current scheme derives the base from the signature and supports a single
distinguished 0th direction. Two extensions are out of scope and will require the
metric to carry an explicit index *layout* — which internal slots are spatial
`e_1..e_n`, which are time, which are ideal — rather than a signature-derived
base:

- **Multiple projective dimensions**, with the extra directions parked at the
  high end while `e_1` stays x.
- **CONFORMAL / ROUND** structures, which involve null bases with their own
  conventions.

When that lands, the forced signature-driven scheme becomes the default case of
the general one, and only `_SIGNATURE_BASE_INDEX` (and the converters that read
it) change.
