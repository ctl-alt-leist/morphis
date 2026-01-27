# Developer Documentation

This section covers architecture decisions, coding conventions, and development workflow for contributors to morphis.

## Contents

1. **[Architecture](1_architecture.md)** — Module structure, class hierarchy, design decisions
2. **[Style Guide](2_style-guide.md)** — Python, LaTeX, and project conventions
3. **[Unit Tests](3_unit-tests.md)** — Mathematical specification of test cases

## Quick Reference

### Module Organization

```
src/morphis/
├── elements/          # Mathematical objects (nouns)
│   ├── base.py       # Element, GradedElement, CompositeElement
│   ├── tensor.py     # Tensor base class
│   ├── vector.py     # Vector class (k-vectors)
│   ├── multivector.py
│   ├── frame.py
│   ├── metric.py
│   └── protocols.py
│
├── operations/        # Operations on elements (verbs)
│   ├── operator.py   # Operator class (linear maps)
│   ├── products.py   # geometric, wedge, reverse, inverse
│   ├── projections.py
│   ├── duality.py
│   ├── norms.py
│   ├── exponential.py
│   ├── factorization.py
│   ├── outermorphism.py
│   ├── subspaces.py
│   ├── spectral.py
│   └── matrix_rep.py
│
├── algebra/           # Algebraic infrastructure
│   ├── specs.py      # VectorSpec
│   ├── patterns.py   # Einsum patterns
│   └── solvers.py    # Linear solvers
│
├── transforms/        # Geometric transformations
│   ├── rotations.py
│   ├── actions.py
│   └── projective.py
│
├── visuals/           # Visualization
└── utils/             # Utilities
```

### Key Classes

| Class | Module | Purpose |
|-------|--------|---------|
| `Vector` | `elements.vector` | Homogeneous k-vector (pure grade) |
| `MultiVector` | `elements.multivector` | Sum of different grades |
| `Metric` | `elements.metric` | Metric tensor and signature |
| `Frame` | `elements.frame` | Ordered collection of grade-1 vectors |
| `Operator` | `operations.operator` | Linear maps between vector spaces |

### Development Commands

```bash
make install      # Full setup (venv, deps, pre-commit hooks)
make lint         # Format and lint with ruff
make test         # Run pytest
make build        # Build packages
make publish      # Tag and push to trigger release
make clean        # Remove generated files
```

### Updating API Documentation

The API documentation is auto-generated:

```bash
# Concise version
uv run python -m morphis.utils.docgen --output docs/api/api.md

# Detailed version with docstrings
uv run python -m morphis.utils.docgen --docstrings --output docs/api/api-detailed.md
```

## Design Philosophy

1. **Vector = k-vector**: The `Vector` class represents any grade k homogeneous multivector, not just grade-1. A "blade" is a Vector that satisfies `.is_blade = True`.

2. **Storage convention**: All geometric objects use `(*collection, *geometric)` shape where geometric dimensions are `(dim,) * grade`.

3. **Batch support**: All operations support leading collection (batch) dimensions via NumPy broadcasting.

4. **Metric-aware**: Operations that require metric information take it from the objects themselves.

5. **Functional + OOP**: Core operations are pure functions; classes provide convenient method syntax via `__mul__`, `__xor__`, etc.
