# Mathematical Concepts

This section provides the mathematical foundations of geometric algebra as implemented in morphis. The documents progress from basic structures to advanced operations.

## Contents

1. **[Foundations](1_foundations.md)** — Vector spaces, tensors, exterior algebra, storage conventions
2. **[The Clifford Algebra](2_clifford-algebra.md)** — The geometric product, grading, reversion
3. **[Objects](3_objects.md)** — Vectors, blades, multivectors, versors, rotors
4. **[Products](4_products.md)** — Wedge, interior, geometric, and antiwedge products
5. **[Duality](5_duality.md)** — Complements, Hodge dual, metric-independent operations
6. **[The Metric](6_metric.md)** — Metric tensor, norms, bulk/weight decomposition
7. **[Outermorphisms](7_outermorphisms.md)** — Linear maps, exterior powers, the Operator class
8. **[Exponentials](8_exponentials.md)** — Exp/log maps, rotor construction, slerp
9. **[Transforms](9_transforms.md)** — Rotors, translators, motors, PGA

## Terminology Note

In morphis, the term **Vector** refers to a homogeneous multivector of any grade $k$ (what other texts call a "$k$-vector"). A grade-1 Vector is a traditional vector, a grade-2 Vector is a bivector, etc. This naming convention emphasizes that all grades are vectors in their respective spaces $\bigwedge^k V$.

## Quick Reference

| Mathematical Object | Morphis Class | Key Property |
|---------------------|---------------|--------------|
| $k$-vector | `Vector` | Pure grade $k$ |
| Blade (simple $k$-vector) | `Vector` with `.is_blade = True` | Factorizable |
| General multivector | `MultiVector` | Sum of grades |
| Rotor | `MultiVector` with `.is_rotor = True` | Even, $R\tilde{R} = 1$ |
| Motor | `MultiVector` with `.is_motor = True` | PGA rigid motion |
| Linear map | `Operator` | Maps between vector spaces |

## Suggested Reading Order

For newcomers to geometric algebra:
1. Foundations → Clifford Algebra → Objects → Products

For those familiar with exterior algebra:
1. Clifford Algebra → Products → Exponentials → Transforms

For PGA and transformations:
1. Transforms → Exponentials → Duality
