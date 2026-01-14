"""Pretty printing utilities for examples and debugging."""

from numpy import ndarray

from morphis.elements import Blade, MultiVector


# =============================================================================
# Matrix Formatting
# =============================================================================


def _indent(text: str, prefix: str = "  ") -> str:
    """Indent each line of text."""
    return "\n".join(prefix + line for line in text.split("\n"))


def format_matrix(arr: ndarray, precision: int = 4) -> str:
    """
    Format array with box-drawing characters for a math-style look.

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
        [ ┌       ┐   ┌       ┐ ]
        [ │ 0   1 │ , │ 0   0 │ ]
        [ │ 0   0 │   │ 1   0 │ ]
        [ └       ┘   └       ┘ ]
    """
    arr = arr.squeeze()

    if arr.ndim == 0:
        # Scalar
        return f"{arr:.{precision}g}"

    if arr.ndim == 1:
        # Column vector
        formatted = [f"{x:.{precision}g}" for x in arr]
        width = max(len(s) for s in formatted)
        lines = [f"│ {s:>{width}} │" for s in formatted]
        bar = " " * (width + 2)
        return "\n".join([f"┌{bar}┐", *lines, f"└{bar}┘"])

    if arr.ndim == 2:
        # Matrix
        rows, cols = arr.shape
        formatted = [[f"{arr[r, c]:.{precision}g}" for c in range(cols)] for r in range(rows)]
        col_widths = [max(len(formatted[r][c]) for r in range(rows)) for c in range(cols)]
        lines = []
        for r in range(rows):
            row_str = "  ".join(f"{formatted[r][c]:>{col_widths[c]}}" for c in range(cols))
            lines.append(f"│ {row_str} │")
        total_width = sum(col_widths) + 2 * (cols - 1) + 2
        bar = " " * total_width
        return "\n".join([f"┌{bar}┐", *lines, f"└{bar}┘"])

    if arr.ndim == 3:
        # Array of matrices - format each and display side by side
        n_matrices = arr.shape[0]
        matrix_strs = [format_matrix(arr[i], precision) for i in range(n_matrices)]
        matrix_lines = [s.split("\n") for s in matrix_strs]
        n_lines = len(matrix_lines[0])

        # Build combined output with large brackets: ⎡⎤ top, ⎢⎥ middle, ⎣⎦ bottom
        result = []
        for line_idx in range(n_lines):
            parts = []
            for mat_idx, mat in enumerate(matrix_lines):
                # Only add comma separator on last line
                if line_idx == n_lines - 1 and mat_idx < n_matrices - 1:
                    sep = " , "
                else:
                    sep = "   " if mat_idx < n_matrices - 1 else ""
                parts.append(mat[line_idx] + sep)
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
        parts = [format_matrix(arr[i], precision) for i in range(arr.shape[0])]
        indented = [_indent(p, "  ") for p in parts]
        return "[\n" + ",\n".join(indented) + "\n]"

    return repr(arr)


def print_matrix(arr: ndarray, precision: int = 4) -> None:
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


def show_blade(name: str, blade: Blade, precision: int = 4) -> None:
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


def show_mv(name: str, mv: MultiVector, precision: int = 4) -> None:
    """Print multivector components with matrix-style formatting."""
    print(f"{name}: grades={list(mv.components.keys())}")
    for grade, blade in mv.components.items():
        formatted = format_matrix(blade.data, precision)
        print(f"  <{name}>_{grade} =")
        print(_indent(formatted, "    "))
