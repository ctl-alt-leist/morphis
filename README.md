# Morphis

A unified mathematical framework for geometric computation, providing elegant tools for working with geometric algebra,
manifolds, and their applications across mathematics and physics. The name derives from Greek *morphe* (form) —
embodying the transformation and adaptation of geometric structures across different contexts while preserving their
essential nature.

## Features

- **Geometric Algebra as Foundation**: All geometric computations use GA structures (blades, multivectors) as the
  primary representation
- **Context-Aware**: Geometric objects know their context (Euclidean, projective, conformal, spacetime) when it matters
- **Mathematical Structures**: Provides geometric and algebraic tools; applications live in examples and downstream
  packages

## Installation

Requires Python 3.12+.

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/morphis.git
cd morphis

# Setup with uv
make setup
make install
```

## Documentation

For detailed project vision, architecture, and design decisions, see
[docs/the-morphis-project.md](docs/the-morphis-project.md).

## Project Structure

```
morphis/
├── src/morphis/           # Main package source
│   ├── algebra/           # Abstract algebraic structures
│   ├── ga/                # Geometric algebra (computational core)
│   ├── geometry/          # Geometric contexts (PGA, CGA, spacetime)
│   ├── manifold/          # Differential geometry on curved spaces
│   ├── topology/          # Topological structures and invariants
│   ├── visualization/     # Geometric visualization tools
│   └── utils/             # Shared utilities
├── docs/                  # Project documentation
├── _archive/              # Archived/reference code
├── Makefile               # Build and development commands
├── pyproject.toml         # Project configuration (uv/PEP 517)
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

### Code Style

- Follows PEP 8
- Uses type hints throughout
- Pydantic models for data structures
- Ruff for formatting and linting

### Testing

Tests are co-located with source code in `tests/` subdirectories within each module:

```bash
make test                           # Run all tests
uv run pytest src/morphis/ga -v     # Run specific module tests
```

## License

MIT License - see LICENSE file for details.

---

*Claude Code was used in the development of this project.*
