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

- **Geometric Algebra Core**: Blades, multivectors, and operations (wedge, geometric product, duality)
- **Metric-Aware**: Objects carry their metric context (Euclidean, projective, etc.)
- **Visualization**: 3D rendering of blades with PyVista, timeline-based animation, 4D projection
- **Motor Transforms**: Rotors and translations via sandwich product

## Installation

Requires Python 3.12+.

```bash
git clone https://github.com/ctl-alt-leist/morphis.git
cd morphis

# Setup with uv
make setup
make install
```

## Project Structure

```
morphis/
├── src/morphis/
│   ├── elements/          # Core GA objects: Blade, MultiVector, Frame, Metric
│   │   └── tests/         # Unit tests
│   ├── operations/        # GA operations: wedge, geometric product, duality, norms
│   │   └── tests/         # Unit tests
│   ├── transforms/        # Rotors, translators, PGA, motor constructors
│   │   └── tests/         # Unit tests
│   ├── visuals/           # PyVista rendering, animation, themes
│   │   └── drawing/       # Blade mesh generation
│   ├── examples/          # Runnable demos (animate_3d.py, animate_4d.py)
│   ├── utils/             # Easing functions, observers, pretty printing
│   └── _legacy/           # Backward-compatible vector math utilities
├── docs/                  # Design documents and notes
├── Makefile               # Build commands
├── pyproject.toml         # Project configuration (uv)
└── ruff.toml              # Linting configuration
```

## Development

This project uses [uv](https://docs.astral.sh/uv/) for Python project management.

### Setup

```bash
make setup      # Create virtual environment
make install    # Install package with dev dependencies
```

### Common Commands

```bash
make lint       # Format and lint code with ruff
make test       # Run tests with pytest
make clean      # Remove generated files and caches
make reset      # Clean and reinstall from scratch
```

### Pre-commit Hooks

Ruff formatting and linting run automatically on commit:

```bash
uv run pre-commit install   # Install hooks (already done if you used make install)
```

### Testing

Tests are co-located with source in `tests/` subdirectories:

```bash
make test                               # Run all tests
uv run pytest src/morphis/elements -v   # Run elements module tests
uv run pytest src/morphis/operations -v # Run operations module tests
```

### Code Style

- Python 3.12+ with type hints
- Ruff for formatting and linting
- PEP 8 compliant

## License

MIT License - see LICENSE file for details.

---

*Claude Code was used in the development of this project.*
