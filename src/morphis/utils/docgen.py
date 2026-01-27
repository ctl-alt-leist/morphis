"""
Documentation Generator for Morphis Package

Generates markdown documentation with function signatures, classes, and methods.
Can be run as a script or imported as a module.

Usage:
    python -m morphis.utils.docgen [--docstrings] [--cushion] [--output FILE]

Options:
    --docstrings    Include docstrings in output (implies --cushion)
    --cushion       Add blank lines between methods/functions
    --output FILE   Write to file instead of stdout
"""

from __future__ import annotations

import argparse
import importlib
import inspect
from io import StringIO
from typing import Any, TextIO


# =============================================================================
# Signature Extraction
# =============================================================================


def get_signature(obj: Any) -> str:
    """Get function/method signature as string."""
    try:
        sig = inspect.signature(obj)
        return str(sig)
    except (ValueError, TypeError):
        return "()"


def get_short_docstring(obj: Any) -> str | None:
    """Get first line of docstring."""
    doc = inspect.getdoc(obj)
    if doc:
        return doc.split("\n")[0].strip()
    return None


def get_source_order(cls: type) -> dict[str, int]:
    """Get the source code order of class members."""
    order = {}
    try:
        source_lines, _start_line = inspect.getsourcelines(cls)
        source = "".join(source_lines)

        # Find def and property declarations in order
        import re

        # Match 'def name(' or '@property' followed by 'def name('
        pattern = r"(?:@property\s+)?def\s+(\w+)\s*\("
        for i, match in enumerate(re.finditer(pattern, source)):
            name = match.group(1)
            if name not in order:
                order[name] = i
    except (OSError, TypeError):
        pass
    return order


def get_module_source_order(module: Any) -> dict[str, int]:
    """Get the source code order of module-level definitions."""
    order = {}
    try:
        source_lines, _ = inspect.getsourcelines(module)
        source = "".join(source_lines)

        import re

        # Match 'def name(' or 'class name'
        pattern = r"^(?:def|class)\s+(\w+)"
        for i, match in enumerate(re.finditer(pattern, source, re.MULTILINE)):
            name = match.group(1)
            if name not in order:
                order[name] = i
    except (OSError, TypeError):
        pass
    return order


# =============================================================================
# Class Formatting
# =============================================================================

# Methods inherited from Pydantic that we want to skip
PYDANTIC_METHODS = {
    "copy",
    "dict",
    "json",
    "model_copy",
    "model_dump",
    "model_dump_json",
    "model_post_init",
    "model_rebuild",
    "model_validate",
    "model_validate_json",
    "model_construct",
    "model_json_schema",
    "model_parametrized_name",
    "parse_file",
    "parse_obj",
    "parse_raw",
    "schema",
    "schema_json",
    "update_forward_refs",
    "validate",
    "from_orm",
}

# Properties inherited from Pydantic that we want to skip
PYDANTIC_PROPERTIES = {
    "model_extra",
    "model_fields_set",
    "model_computed_fields",
    "model_fields",
}


def format_class(
    cls: type,
    include_docstrings: bool = False,
    cushion: bool = False,
) -> list[str]:
    """Format a class with its methods and properties."""
    lines = []

    # Class declaration with bases
    bases = [b.__name__ for b in cls.__bases__ if b.__name__ != "object"]
    if bases:
        lines.append(f"class {cls.__name__}({', '.join(bases)}):")
    else:
        lines.append(f"class {cls.__name__}:")

    # Class docstring
    if include_docstrings:
        doc = get_short_docstring(cls)
        if doc:
            lines.append(f'    """{doc}"""')
            if cushion:
                lines.append("")

    # Get source order for sorting
    source_order = get_source_order(cls)

    # Collect members
    properties = []
    methods = []

    for name in dir(cls):
        if name.startswith("_") and name != "__init__":
            continue

        # Skip Pydantic internals
        if name in PYDANTIC_METHODS or name in PYDANTIC_PROPERTIES:
            continue

        try:
            attr = inspect.getattr_static(cls, name, None)
        except AttributeError:
            continue

        if isinstance(attr, property):
            properties.append(name)
        elif inspect.isfunction(attr) or inspect.ismethod(attr):
            methods.append((name, attr))

    # Sort by source order, fallback to alphabetical
    def sort_key(name):
        return (source_order.get(name, 999), name)

    properties.sort(key=sort_key)
    methods.sort(key=lambda x: sort_key(x[0]))

    # Format properties
    for i, name in enumerate(properties):
        if cushion and i > 0:
            lines.append("")
        lines.append("    @property")
        lines.append(f"    def {name}(self): ...")

    # Format methods
    for i, (name, method) in enumerate(methods):
        if cushion and (i > 0 or properties):
            lines.append("")
        sig = get_signature(method)
        sig = _simplify_signature(sig)
        lines.append(f"    def {name}{sig}")
        if include_docstrings:
            doc = get_short_docstring(method)
            if doc:
                lines.append(f'        """{doc}"""')

    if len(lines) == 1:  # Only class declaration
        lines.append("    ...")

    return lines


def _simplify_signature(sig: str) -> str:
    """Simplify complex type annotations in signatures."""
    import re

    # Remove 'morphis.' prefix
    sig = re.sub(r"morphis\.\w+\.", "", sig)
    # Simplify numpy array types
    sig = re.sub(r"numpy\.ndarray\[.*?\]", "NDArray", sig)
    # Simplify Literal types
    sig = re.sub(r"Literal\[([^\]]+)\]", r"\1", sig)
    return sig


# =============================================================================
# Module Analysis
# =============================================================================


def get_module_contents(module: Any) -> tuple[list, list]:
    """Get public classes and functions defined in a module, in source order."""
    classes = []
    functions = []

    # Get source order
    source_order = get_module_source_order(module)

    for name, obj in inspect.getmembers(module):
        # Skip private
        if name.startswith("_"):
            continue

        # Check if actually defined in this module
        if hasattr(obj, "__module__") and obj.__module__ != module.__name__:
            continue

        if inspect.isclass(obj):
            classes.append((name, obj))
        elif inspect.isfunction(obj):
            functions.append((name, obj))

    # Sort by source order, fallback to alphabetical
    def sort_key(item):
        name = item[0]
        return (source_order.get(name, 999), name)

    classes.sort(key=sort_key)
    functions.sort(key=sort_key)

    return classes, functions


def document_module(
    module_name: str,
    include_docstrings: bool = False,
    cushion: bool = False,
) -> list[str]:
    """Generate markdown documentation for a module."""
    try:
        module = importlib.import_module(module_name)
    except Exception as e:
        return [f"<!-- Failed to import {module_name}: {e} -->"]

    classes, functions = get_module_contents(module)

    if not classes and not functions:
        return []

    lines = []
    lines.append(f"\n### `{module_name}`\n")

    # Module docstring
    if include_docstrings and module.__doc__:
        first_line = module.__doc__.strip().split("\n")[0]
        lines.append(f"*{first_line}*\n")

    # Classes
    for i, (_name, cls) in enumerate(classes):
        if cushion and i > 0:
            lines.append("")
        lines.append("```python")
        lines.extend(format_class(cls, include_docstrings, cushion))
        lines.append("```\n")

    # Functions
    if functions:
        lines.append("**Functions:**\n")
        lines.append("```python")
        for i, (name, func) in enumerate(functions):
            if cushion and i > 0:
                lines.append("")
            sig = get_signature(func)
            sig = _simplify_signature(sig)
            lines.append(f"def {name}{sig}")
            if include_docstrings:
                doc = get_short_docstring(func)
                if doc:
                    lines.append(f'    """{doc}"""')
        lines.append("```\n")

    return lines


# =============================================================================
# Full Package Documentation
# =============================================================================


# Module groups for documentation
MODULE_GROUPS = {
    "Elements": {
        "description": "Core geometric algebra objects.",
        "modules": [
            "morphis.elements.base",
            "morphis.elements.metric",
            "morphis.elements.tensor",
            "morphis.elements.vector",
            "morphis.elements.multivector",
            "morphis.elements.frame",
            "morphis.elements.protocols",
        ],
    },
    "Operations": {
        "description": "Geometric algebra operations and products.",
        "modules": [
            "morphis.operations.operator",
            "morphis.operations.products",
            "morphis.operations.projections",
            "morphis.operations.duality",
            "morphis.operations.norms",
            "morphis.operations.exponential",
            "morphis.operations.structure",
            "morphis.operations.factorization",
            "morphis.operations.subspaces",
            "morphis.operations.spectral",
            "morphis.operations.outermorphism",
            "morphis.operations.matrix_rep",
        ],
    },
    "Algebra": {
        "description": "Linear algebra utilities for operators.",
        "modules": [
            "morphis.algebra.specs",
            "morphis.algebra.patterns",
            "morphis.algebra.solvers",
        ],
    },
    "Transforms": {
        "description": "Geometric transformations.",
        "modules": [
            "morphis.transforms.rotations",
            "morphis.transforms.actions",
            "morphis.transforms.projective",
        ],
    },
    "Visualization": {
        "description": "Visualization and rendering tools.",
        "modules": [
            "morphis.visuals.canvas",
            "morphis.visuals.theme",
            "morphis.visuals.projection",
            "morphis.visuals.drawing.vectors",
            "morphis.visuals.contexts",
            "morphis.visuals.operations",
            "morphis.visuals.loop",
        ],
    },
    "Utilities": {
        "description": "Helper utilities.",
        "modules": [
            "morphis.utils.pretty",
            "morphis.utils.observer",
        ],
    },
}


def generate_documentation(
    include_docstrings: bool = False,
    cushion: bool = False,
    output: TextIO | None = None,
) -> str:
    """Generate full package documentation."""
    out = output or StringIO()

    # If docstrings are included, auto-enable cushion
    if include_docstrings:
        cushion = True

    out.write("# Morphis Package API Reference\n\n")
    out.write("*Auto-generated documentation*\n\n")

    for group_name, group_info in MODULE_GROUPS.items():
        out.write("---\n\n")
        out.write(f"## {group_name}\n\n")
        out.write(f"{group_info['description']}\n")

        for module_name in group_info["modules"]:
            lines = document_module(module_name, include_docstrings, cushion)
            if lines:
                out.write("\n".join(lines))
                out.write("\n")

    if output is None:
        return out.getvalue()
    return ""


# =============================================================================
# CLI Entry Point
# =============================================================================


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate Morphis package documentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--docstrings",
        action="store_true",
        help="Include docstrings in output (implies --cushion)",
    )
    parser.add_argument(
        "--cushion",
        action="store_true",
        help="Add blank lines between methods/functions for readability",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file (default: stdout)",
    )

    args = parser.parse_args()

    if args.output:
        with open(args.output, "w") as f:
            generate_documentation(
                include_docstrings=args.docstrings,
                cushion=args.cushion,
                output=f,
            )
        print(f"Documentation written to {args.output}")
    else:
        doc = generate_documentation(
            include_docstrings=args.docstrings,
            cushion=args.cushion,
        )
        print(doc)


if __name__ == "__main__":
    main()
