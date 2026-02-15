"""
Morphis CLI

Command-line interface for running examples and viewing saved scenes.
"""

import runpy
import sys
from pathlib import Path


def _discover_examples() -> list[str]:
    """Discover available examples by scanning the examples directory."""
    examples_dir = Path(__file__).parent / "examples"
    examples = []
    for path in sorted(examples_dir.glob("*.py")):
        name = path.stem
        if not name.startswith("_"):
            examples.append(name)
    return examples


def main() -> None:
    """CLI entry point for morphis."""
    try:
        _main()
    except KeyboardInterrupt:
        print()
        sys.exit(0)


def _main() -> None:
    """Main CLI logic."""
    args = sys.argv[1:]

    if not args:
        _print_usage()
        return

    command = args[0]

    if command == "example":
        _run_example(args[1:])
    elif command == "view":
        _view_scene(args[1:])
    else:
        print(f"Unknown command: {command}")
        _print_usage()
        sys.exit(1)


def _print_usage() -> None:
    """Print usage information."""
    print("Usage:")
    print("  morphis example <name>   Run an example")
    print("  morphis view <file>      Open a saved .scene file")
    print()
    print("Examples:")
    for name in _discover_examples():
        print(f"  {name}")


def _run_example(args: list[str]) -> None:
    """Run a named example."""
    examples = _discover_examples()

    if not args:
        print("Usage: morphis example <name>")
        print()
        print("Available examples:")
        for name in examples:
            print(f"  {name}")
        return

    name = args[0]
    remaining = args[1:]

    if name not in examples:
        print(f"Unknown example: {name}")
        print(f"Available: {', '.join(examples)}")
        sys.exit(1)

    module = f"morphis.examples.{name}"
    sys.argv = [module] + remaining
    runpy.run_module(module, run_name="__main__", alter_sys=True)


def _view_scene(args: list[str]) -> None:
    """View a saved .scene file."""
    if not args:
        print("Usage: morphis view <file.scene>")
        sys.exit(1)

    path = args[0]

    from morphis.visuals import Scene

    scene = Scene.load(path)
    scene.show()


if __name__ == "__main__":
    main()
