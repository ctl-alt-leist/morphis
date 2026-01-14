## The Manifest Generality Challenge in PGA

### What Einsum Achieves for Vectors

The original `vectors.py` achieved something powerful:

```python
def dot(u, v, g):
    return einsum("ab, ...a, ...b -> ...", g, u, v)
```

One line. No shape inspection. No conditionals. Handles:
- `(d,)` with `(d,)` → scalar
- `(d,)` with `(n, d)` → `(n,)`  
- `(n, d)` with `(n, d)` → `(n,)`
- `(..., d)` with `(..., d)` → `(...)`

The code *is* the mathematical expression. NumPy's broadcasting and einsum's `...` absorb all the casework. This is
**manifest generality** — the generality isn't achieved through logic handling cases, it's inherent in the notation
itself.

### The k-Blade Problem

Vectors: fixed structure `(..., d)` — one index.

k-blades: grade-dependent structure `(..., d, d, ..., d)` — k indices.

To write a general `blade_norm_squared`, you must now:
1. Inspect the array to determine k
2. Build an einsum string dynamically
3. Apply 1/k! normalization

The code now contains logic *about* the mathematics rather than *being* the mathematics. Manifest generality is lost.

### Why Index Notation and Levi-Civita

The Levi-Civita symbol encodes antisymmetry as data:

$$B^{mn} = \frac{1}{2} u^m v^n \, \varepsilon^{mn}$$

Operations become tensor contractions — wedge, dual, complement — all expressible as einsum with ε. This is the right
direction. But grade-dependence persists: different k requires different einsum signatures.

### What "Elegant" Means

Elegant code should:

- **Maximize computational and notational efficiency in the fewest, simplest lines**
- **Read like the mathematical expression it implements**
- **Minimize or eliminate conditional logic**
- **Be general — and manifestly so**

"Manifestly" is key. The generality should be visible in the structure of the code, not buried in branches that handle
cases. When you read the code, you should see immediately that it works for all valid inputs.

### Existing GA Libraries

Libraries exist: `clifford`, `galgebra`, `kingdon`, `ganja.js`, etc. They implement the full algebraic machinery — basis
blades, geometric products, multivector types.

Concerns:

- **Style mismatch**: Many use camelCase, C++-like patterns, or APIs that don't feel Pythonic
- **Performance**: Unclear if they compete with raw einsum on large arrays
- **Opacity**: May hide the mathematics behind abstraction rather than expose it

However, they may offer **conceptual value**:

- Basis blade algebra without storing full dense tensors
- Sparse or symbolic representations
- Grade-aware types that carry structure

Worth examining — not to adopt wholesale, but to borrow ideas if they serve the goal.

### Constraints on Any Alternative

Whatever approach we take must:

1. **Compete with einsum speed** on large data arrays
2. **Improve elegance**, not diminish it
3. **Preserve batch semantics** — `(..., ?)` shapes just work
4. **Look like the math** — the code should be readable as its definition

### The Core Question

For vectors, einsum gives us manifest generality. Can we extend this to the full exterior algebra — or must we accept
that grade-dependent operations require grade-dependent code?

Possibilities to explore:

- Full multivector representation `(2^d,)` with grade masks?
- Grade-specific types where operations within a grade stay manifest?
- Hybrid: einsum for the numerics, thin algebraic layer for structure?
- Something from the GA libraries that we haven't considered?

The goal: code that scales to arbitrary d, handles arbitrary batch dimensions, and reads like mathematics — for all
grades, not just grade 1.
