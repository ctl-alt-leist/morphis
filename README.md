# Morphis

A unified mathematical framework for geometric computation, providing elegant tools for working with geometric algebra,
manifolds, and their applications across mathematics and physics. The name derives from Greek *morphe* (form) —
embodying the transformation and adaptation of geometric structures across different contexts while preserving their
essential nature.

<p align="center">
  <img src="figures/rotations-4d.gif" alt="4D rotations animation" width="400">
</p>

<p align="center" width="500">
  <em>A 4D orthonormal frame rotating through bivector planes, projected to 3D.
  The view switches between e₁e₂e₃ and e₂e₃e₄ projections mid-animation.</em>
</p>

## Features

- **Geometric Algebra Core**: Vectors (k-vectors), multivectors, and operations (wedge, geometric product, duality)
- **Metric-Aware**: Objects carry their metric context (Euclidean, projective, etc.)
- **Linear Operators**: Structured linear maps between vector spaces with SVD, pseudoinverse, least squares
- **Visualization**: 3D rendering of vectors with PyVista, timeline-based animation, 4D projection
- **Motor Transforms**: Rotors and translations via sandwich product

## Documentation

- [Project Overview](docs/0_project-overview.md) — Vision and scope
- [Concepts](docs/1_concepts/) — Mathematical foundations (vectors, products, duality, transforms)
- [API Reference](docs/3_api/api.md) — Public interface
- [Architecture](docs/5_dev/1_architecture.md) — Design philosophy and decisions

## Quick Start

```python
from morphis.elements import Frame, basis_vectors, euclidean_metric
from morphis.operations import normalize
from morphis.transforms import rotor
from numpy import pi

# Create a 3D Euclidean metric and basis vectors
g = euclidean_metric(3)
e1, e2, e3 = basis_vectors(g)

# Bivector: oriented plane of rotation
b = (e1 ^ e2).normalize()

# Frame: ordered collection of vectors
F = Frame(e1, e2, e3)

# Rotor: multivector that performs rotation
R = rotor(b, pi / 4)

# Transform vector and frame via sandwich product
e1_rotated = e1.transform(R)
F_rotated = F.transform(R)
```

## Installation

Requires Python 3.12+.

```bash
pip install morphis
```

### From Source

```bash
git clone https://github.com/ctl-alt-leist/morphis.git
cd morphis
make install
```

## Project Structure

```
morphis/
├── src/morphis/
│   ├── elements/       # Core GA objects: Vector, MultiVector, Frame, Metric
│   │   └── tests/
│   ├── algebra/        # Linear algebra: VectorSpec, einsum patterns, solvers
│   │   └── tests/
│   ├── operations/     # GA operations: wedge, geometric product, duality, norms
│   │   └── tests/
│   ├── transforms/     # Rotors, translators, PGA, motor constructors
│   │   └── tests/
│   ├── visuals/        # PyVista rendering, animation, themes
│   │   └── drawing/    # Vector mesh generation
│   ├── examples/       # Runnable demos
│   └── utils/          # Easing functions, observers, pretty printing
├── docs/               # Design documents
├── pyproject.toml      # Project configuration
├── Makefile            # Development commands
└── ruff.toml           # Linting configuration
```

## Development

This project uses [uv](https://docs.astral.sh/uv/) for Python project management.

### Setup

```bash
make install    # Create venv, install dependencies, setup pre-commit hooks
```

Or step by step:

```bash
make setup      # Create venv and sync dependencies
uv run pre-commit install  # Install git hooks
```

### Common Commands

| Command       | Description                          |
|---------------|--------------------------------------|
| `make lint`   | Format and lint code with ruff       |
| `make test`   | Run tests with pytest                |
| `make build`  | Build distribution packages          |
| `make clean`  | Remove generated files and caches    |
| `make reset`  | Clean and reinstall from scratch     |

### Testing

Tests are co-located with source in `tests/` subdirectories:

```bash
make test                                # Run all tests
uv run pytest src/morphis/elements -v    # Run specific module tests
```

### Code Style

- Python 3.12+ with type hints
- Ruff for formatting and linting
- Pre-commit hooks run automatically on commit

## Release Workflow

Releases are triggered by git tags. When a tag matching `v*` is pushed, GitHub Actions
automatically builds and publishes to PyPI.

### To Release a New Version

1. Update version in `pyproject.toml`
2. Commit the change
3. Run `make publish`

```bash
# Example: releasing version 0.2.0
# Edit pyproject.toml: version = "0.2.0"
git add pyproject.toml
git commit -m "Bump version to 0.2.0"
make publish
```

The `make publish` command will:
- Read the version from `pyproject.toml`
- Create a git tag `v{version}`
- Push the tag to trigger the release workflow

## License

MIT License - see LICENSE file for details.

---

*Claude Code was used in the development of this project.*
