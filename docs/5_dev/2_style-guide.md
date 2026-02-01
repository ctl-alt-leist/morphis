# Style Guide

## Python

### Imports

```python
# Yes
from numpy import einsum, sqrt, zeros, eye
from numpy.typing import NDArray

# No
import numpy as np
from __future__ import annotations
```

### Strings

Double quotes first, single inside:

```python
"hello"
"it's fine"
f"shape: {x.shape}"
```

### Indices

Never use `idx`. Always use a letter for loop indices.

Loop variables: prefer `k`, `m`, `n` or compound `mn`, `ab`.

Tensor indices use `a, b, c, d, m, n, p, q`—never `i, j`.

### Code Organization

#### Section Headers

Use `# ===` headers sparingly to group related functions:

```python
# Yes — groups related functions
# =============================================================================
# Norms
# =============================================================================

def form(v):
    ...

def norm(v):
    ...
```

#### Docstrings

Keep docstrings concise. Brief description, optional detail paragraph, return value.

```python
# Yes
def wedge(a: Blade, b: Blade) -> Blade:
    """
    Compute the wedge (exterior) product of two blades. The result is a blade
    of grade (grade(a) + grade(b)). The wedge product is associative and
    anticommutative for vectors.

    Returns the wedge product blade, or zero if the blades are linearly dependent.
    """
```

#### Imports

All imports at the top of the module, never inside functions.

#### Return Statements

Return statements at the bottom. Avoid early returns. For functions longer than two lines, add a blank line before return.

```python
# Yes
def compute(x, y):
    if x > 0:
        result = x + y
    else:
        result = x - y

    return result

# No — early return
def compute(x, y):
    if x > 0:
        return x + y
    return x - y
```

### Whitespace

Space around operators in math:

```python
d + 1
k - 1
shape[:-1] + (d + 1,)
```

### Differences

Always second minus first:

```python
diff = q - p      # vector from p to q
delta = b - a
```

### Einsum

```python
# Explicit indices, batch dimensions via ...
einsum("ab, ...a, ...b -> ...", g, u, v)
einsum("...a, ...b -> ...ab", u, v)
```

### Batch Support

All functions support leading batch dimensions. Use `...` in einsum and `[..., idx]` for slicing.

### Variable Naming

Blade variables: lowercase (`u, v, w, b`)

MultiVector variables: uppercase (`U, V, W, M`)

Single-letter + number: no underscore (`v1, v2`)

Word + number: use underscore (`blade_1, blade_2`)

### Abbreviated Suffixes

```python
# Yes
u_rev = reverse(u)
u_inv = inverse(u)

# No
u_reverse = reverse(u)
```

### Unified API

Public functions accept both types; use private helpers:

```python
def func(u: Blade | MultiVector) -> Blade | MultiVector:
    if isinstance(u, Blade):
        return _func_bl(u)
    return _func_mv(u)
```

### Factor Ordering

Numeric factors in front:

```python
0.5 * (uv + vu)
2 * x + 1
```

### If-Block Spacing

Empty lines between non-trivial branches:

```python
if r == 0:
    result = compute_scalar(...)
    component = wrap_scalar(result)

elif c == 0:
    delta = get_delta(...)
    result = compute_wedge(...)
    component = wrap_blade(result)
```

### Type Annotations

Use native Python 3.10+ union syntax:

```python
def func(u: Blade | None) -> Blade | MultiVector: ...
```

---

## Project Management

### uv Commands

```bash
uv venv           # Create virtual environment
uv sync           # Sync dependencies from pyproject.toml
uv run <cmd>      # Run command in venv
uv build          # Build distribution packages
uv publish        # Publish to PyPI
```

### Make Commands

```bash
make install      # Full setup (venv, deps, pre-commit hooks)
make lint         # Format and lint with ruff
make test         # Run pytest
make build        # Build packages
make publish      # Tag and push to trigger release
make clean        # Remove generated files
make reset        # Clean and reinstall
```

### Git Workflow

Branch from `main`, open PRs for review.

#### Releases

1. Update version in `pyproject.toml`
2. Commit: `git commit -am "Bump version to X.Y.Z"`
3. Run: `make publish`

This creates tag `vX.Y.Z` and pushes, triggering the GitHub Actions publish workflow.

#### Commit Messages

- Present tense, imperative mood: "Add feature" not "Added feature"
- First line: brief summary (50 chars)
- Body: explain *why*, not just *what*

---

## LaTeX

### Characters

Use Unicode Greek directly: α, β, γ, θ, ε

### Spacing

```latex
d + 1
a^m b^n
```

### Equations

Single line:

```latex
$$\mathbf{B} = B^{mn} \mathbf{e}_{mn}$$
```

Multi-line with `align`:

```latex
$$
\begin{align}
    \mathbf{a} \wedge \mathbf{b}
        &= \frac{1}{2} a^m b^n \, \varepsilon^{mn} \, \mathbf{e}_{mn} \\ \\
        &= \frac{1}{2}(a^1 b^2 - a^2 b^1) \, \mathbf{e}_{12}
\end{align}
$$
```

### Notation

| Object | Notation |
|--------|----------|
| Vector | $\mathbf{v}$ |
| Basis vector | $\mathbf{e}_m$ |
| Bivector | $\mathbf{B}$ |
| Pseudoscalar | $\mathbb{1}$ |
| Scalar | $\mathbf{1}$ |
| Levi-Civita | $\varepsilon^{mn...}$ |
| Metric | $g_{ab}$ |

### Thin Spaces

Use `\,` before differentials and between adjacent terms:

```latex
\int dx \, f(x)
a^m b^n \, \varepsilon^{mn}
```
