"""Pretty printing utilities for examples and debugging."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import NDArray


if TYPE_CHECKING:
    from morphis.elements import Frame, MultiVector, Vector


# =============================================================================
# Configuration
# =============================================================================

MAX_ROWS = 8  # Max rows to display before truncating
MAX_COLS = 8  # Max columns to display before truncating
MAX_SLICES = 5  # Max slices (for 3D arrays) before truncating
PRECISION = 4


# =============================================================================
# Matrix Formatting
# =============================================================================


def _indent(text: str, prefix: str = "  ") -> str:
    """Indent each line of text."""
    return "\n".join(prefix + line for line in text.split("\n"))


def _clean_zeros(arr: NDArray, tol: float = 1e-10) -> NDArray:
    """Replace near-zero values with exact zero to avoid -0 display."""
    from numpy import abs as np_abs, where

    return where(np_abs(arr) < tol, 0.0, arr)


def format_matrix(
    arr: NDArray,
    precision: int = 4,
    max_rows: int = MAX_ROWS,
    max_cols: int = MAX_COLS,
    max_slices: int = MAX_SLICES,
) -> str:
    """
    Format array with box-drawing characters for a math-style look.

    Large arrays are truncated with "..." in the middle.

    Examples:
        1D vector:
        ┌      ┐
        │  1.0 │
        │  2.0 │
        │  3.0 │
        └      ┘

        2D matrix:
        ┌            ┐
        │  0.0   1.0 │
        │ -1.0   0.0 │
        └            ┘

        3D array (list of matrices):
        ⎡ ┌       ┐   ┌       ┐ ⎤
        ⎢ │ 0   1 │ , │ 0   0 │ ⎥
        ⎢ │ 0   0 │   │ 1   0 │ ⎥
        ⎣ └       ┘   └       ┘ ⎦
    """
    arr = _clean_zeros(arr.squeeze())

    if arr.ndim == 0:
        # Scalar
        return f"{arr:.{precision}g}"

    if arr.ndim == 1:
        # Column vector
        n = len(arr)
        if n <= max_rows:
            formatted = [f"{x:.{precision}g}" for x in arr]
        else:
            # Show first few, ..., last few
            n_show = max_rows // 2
            first = [f"{x:.{precision}g}" for x in arr[:n_show]]
            last = [f"{x:.{precision}g}" for x in arr[-n_show:]]
            formatted = first + ["⋮"] + last

        width = max(len(s) for s in formatted)
        lines = [f"│ {s:>{width}} │" for s in formatted]
        bar = " " * (width + 2)
        return "\n".join([f"┌{bar}┐", *lines, f"└{bar}┘"])

    if arr.ndim == 2:
        # Matrix
        rows, cols = arr.shape
        truncate_rows = rows > max_rows
        truncate_cols = cols > max_cols

        # Determine which rows/cols to show
        if truncate_rows:
            n_rows = max_rows // 2
            row_indices = list(range(n_rows)) + [-1] + list(range(rows - n_rows, rows))
        else:
            row_indices = list(range(rows))

        if truncate_cols:
            n_cols = max_cols // 2
            col_indices = list(range(n_cols)) + [-1] + list(range(cols - n_cols, cols))
        else:
            col_indices = list(range(cols))

        # Format cells
        formatted = []
        for r in row_indices:
            row_cells = []
            for c in col_indices:
                if r == -1:
                    row_cells.append("⋮" if c != -1 else "⋱")
                elif c == -1:
                    row_cells.append("⋯")
                else:
                    row_cells.append(f"{arr[r, c]:.{precision}g}")
            formatted.append(row_cells)

        # Calculate column widths
        n_display_cols = len(col_indices)
        col_widths = [max(len(formatted[r][c]) for r in range(len(row_indices))) for c in range(n_display_cols)]

        # Build lines
        lines = []
        for _r, row_cells in enumerate(formatted):
            row_str = "  ".join(f"{row_cells[c]:>{col_widths[c]}}" for c in range(n_display_cols))
            lines.append(f"│ {row_str} │")

        total_width = sum(col_widths) + 2 * (n_display_cols - 1) + 2
        bar = " " * total_width
        return "\n".join([f"┌{bar}┐", *lines, f"└{bar}┘"])

    if arr.ndim == 3:
        # Array of matrices - format each and display side by side
        n_matrices = arr.shape[0]
        truncate_slices = n_matrices > max_slices

        if truncate_slices:
            n_show = max_slices // 2
            slice_indices = list(range(n_show)) + [-1] + list(range(n_matrices - n_show, n_matrices))
        else:
            slice_indices = list(range(n_matrices))

        # Format each matrix (or placeholder for ellipsis)
        matrix_strs = []
        for idx in slice_indices:
            if idx == -1:
                matrix_strs.append(None)  # Placeholder for ...
            else:
                matrix_strs.append(format_matrix(arr[idx], precision, max_rows, max_cols, max_slices))

        # Get dimensions from first real matrix
        first_real = next(s for s in matrix_strs if s is not None)
        first_lines = first_real.split("\n")
        n_lines = len(first_lines)

        # Build ellipsis placeholder of same height
        ellipsis_lines = [" " for _ in range(n_lines)]
        mid = n_lines // 2
        ellipsis_lines[mid] = "⋯"

        matrix_lines = []
        for s in matrix_strs:
            if s is None:
                matrix_lines.append(ellipsis_lines)
            else:
                matrix_lines.append(s.split("\n"))

        # Build combined output with large brackets
        result = []
        for line_idx in range(n_lines):
            parts = []
            for mat_idx, mat in enumerate(matrix_lines):
                line = mat[line_idx] if line_idx < len(mat) else ""
                # Only add comma separator on last line
                if line_idx == n_lines - 1 and mat_idx < len(matrix_lines) - 1:
                    sep = " , "
                else:
                    sep = "   " if mat_idx < len(matrix_lines) - 1 else ""
                parts.append(line + sep)
            row = "".join(parts)
            # Use large bracket characters
            if line_idx == 0:
                result.append("⎡ " + row + " ⎤")
            elif line_idx == n_lines - 1:
                result.append("⎣ " + row + " ⎦")
            else:
                result.append("⎢ " + row + " ⎥")
        return "\n".join(result)

    # 4D+ - format as nested list of 3D arrays
    if arr.ndim >= 4:
        n = arr.shape[0]
        if n <= max_slices:
            indices = list(range(n))
        else:
            n_show = max_slices // 2
            indices = list(range(n_show)) + [-1] + list(range(n - n_show, n))

        parts = []
        for i in indices:
            if i == -1:
                parts.append("  ...")
            else:
                parts.append(_indent(format_matrix(arr[i], precision, max_rows, max_cols, max_slices), "  "))

        return "[\n" + ",\n".join(parts) + "\n]"

    return repr(arr)


def print_matrix(arr: NDArray, precision: int = 4) -> None:
    """Print array with box-drawing characters."""
    print(format_matrix(arr, precision))


def section(title: str, width: int = 70) -> None:
    """Print a section header."""
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def subsection(title: str) -> None:
    """Print a subsection header."""
    print()
    print(f"--- {title} ---")


def show_vec(name: str, blade: Vector, precision: int = 4) -> None:
    """Print blade info with matrix-style data formatting."""
    print(f"{name}: grade={blade.grade}, shape={blade.shape}, collection={blade.collection}")
    formatted = format_matrix(blade.data, precision)
    print(_indent(formatted))


def show_array(name: str, arr, precision: int = 4) -> None:
    """Print array with matrix-style formatting."""
    print(f"{name}:")
    formatted = format_matrix(arr, precision)
    print(_indent(formatted))


def show_scalar(name: str, value, precision: int = 4) -> None:
    """Print a scalar value."""
    if hasattr(value, "__float__"):
        print(f"{name} = {value:.{precision}g}")
    else:
        print(f"{name} = {value}")


def show_mv(name: str, M: MultiVector, precision: int = 4) -> None:
    """Print multivector components with matrix-style formatting."""
    print(f"{name}: grades={list(M.data.keys())}")
    for grade, blade in M.data.items():
        formatted = format_matrix(blade.data, precision)
        print(f"  <{name}>_{grade} =")
        print(_indent(formatted, "    "))


# =============================================================================
# Element Formatting (for __str__ methods)
# =============================================================================


def format_vector(v: Vector, precision: int = PRECISION) -> str:
    """
    Format a Vector for display.

    Shows grade, dimension, and data in mathematical notation.
    """
    lines = [f"Vector  grade={v.grade}, dim={v.dim}"]

    if v.collection:
        lines[0] += f", collection={v.collection}"

    # format_matrix now handles truncation with ...
    lines.append(_indent(format_matrix(v.data, precision)))

    return "\n".join(lines)


def format_multivector(M: MultiVector, precision: int = PRECISION) -> str:
    """
    Format a MultiVector for display.

    Shows grades and component data with matrix notation.
    """
    if not M.data:
        return "MultiVector  (empty)"

    grades = sorted(M.data.keys())
    lines = [f"MultiVector  grades={grades}, dim={M.dim}"]

    if M.collection:
        lines[0] += f", collection={M.collection}"

    for k in grades:
        vec = M.data[k]
        # format_matrix handles truncation
        mat_str = format_matrix(vec.data, precision)
        lines.append(f"  [{k}]:")
        lines.append(_indent(mat_str, "    "))

    return "\n".join(lines)


def format_frame(F: Frame, precision: int = PRECISION) -> str:
    """
    Format a Frame for display.

    Shows span, dimension, and vector data as a matrix.
    """
    lines = [f"Frame  span={F.span}, dim={F.dim}"]

    if F.collection:
        lines[0] += f", collection={F.collection}"

    # format_matrix handles truncation
    lines.append(_indent(format_matrix(F.data, precision)))

    return "\n".join(lines)
