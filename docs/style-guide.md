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

Tensor indices use `a, b, c, d, m, n, p, q` — never `i, j`.

### Code Organization

#### Section Headers

Use `# ===` headers sparingly to group related functions, not for individual functions. Headers should mark logical
sections of a module (e.g., "Constructors", "Arithmetic", "Norms").

```python
# Yes — groups related functions
# =============================================================================
# Norms
# =============================================================================

def norm_squared(b, g):
    ...

def norm(b, g):
    ...

def normalize(b, g):
    ...

# No — header for single function
# =============================================================================
# Norm Squared
# =============================================================================

def norm_squared(b, g):
    ...
```

#### Docstrings

Keep docstrings concise. Structure: brief description, optional detail paragraph in sentence form (not one line per
sentence), and what it returns.

```python
# Yes
def wedge(a: Blade, b: Blade) -> Blade:
    """
    Compute the wedge (exterior) product of two blades. The result is a blade
    of grade (grade(a) + grade(b)). The wedge product is associative and
    anticommutative for vectors.

    Returns the wedge product blade, or zero if the blades are linearly dependent.
    """
    ...

# No — too fragmented
def wedge(a: Blade, b: Blade) -> Blade:
    """
    Compute the wedge (exterior) product of two blades.

    The wedge product is:
        - Associative
        - Anticommutative for vectors
        - Zero when applied to linearly dependent vectors

    Args:
        a: First blade.
        b: Second blade.

    Returns:
        Blade of grade grade(a) + grade(b).
    """
    ...
```

#### Imports

All imports must be at the top of the module, never inside functions.

#### Return Statements

Return statements belong at the very bottom of a function. Avoid early returns in conditionals; instead, structure logic
so the return is always the final line. For functions longer than two lines (excluding the return), add a blank line
before the return.

```python
# Yes
def compute(x, y):
    if x > 0:
        result = x + y
    else:
        result = x - y

    return result

# No — early return, not at bottom
def compute(x, y):
    if x > 0:
        return x + y
    return x - y

# Yes — short function, no blank line needed
def double(x):
    result = x * 2
    return result

# Yes — longer function, blank line before return
def process(data, factor):
    scaled = data * factor
    normalized = scaled / max(scaled)
    clipped = clip(normalized, 0, 1)

    return clipped
```

### Whitespace

Always space around operators in math:
```python
# Yes
d + 1
k - 1
shape[:-1] + (d + 1,)

# No
d+1
k-1
```

### Differences

Always second minus first:
```python
# Yes
diff = q - p      # vector from p to q
delta = b - a

# No
diff = p - q
```

### Einsum

```python
# Explicit indices, batch dimensions via ...
einsum("ab, ...a, ...b -> ...", g, u, v)
einsum("...a, ...b -> ...ab", u, v)
```

### Batch Support

All functions support leading batch dimensions. Use `...` in einsum and `[..., idx]` for slicing.

---

## LaTeX

### Characters

Use Unicode Greek directly in text: α, β, γ, θ, ε

### Spacing

```latex
# Yes
d + 1
k - 1
a^m b^n

# No  
d+1
k-1
a^mb^n
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

Use `\\ \\` for extra vertical space between major steps.

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
