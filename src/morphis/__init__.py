"""
Morphis - Geometric Algebra Library

A unified mathematical framework for geometric computation.
"""

import runpy
import sys


EXAMPLES = [
    "clifford",
    "duality",
    "exterior",
    "operators",
    "phasors",
    "rotations_3d",
    "rotations_4d",
    "transforms_pga",
]


def main() -> None:
    """CLI entry point: morphis example [name]"""
    args = sys.argv[1:]

    # morphis (no args) or morphis example (no name)
    if not args or args == ["example"]:
        print("Usage: morphis example <name>")
        print()
        print("Available examples:")
        for name in EXAMPLES:
            print(f"  {name}")
        return

    # morphis example <name> [args...]
    if args[0] == "example":
        name = args[1] if len(args) > 1 else None
        remaining = args[2:]

        if name not in EXAMPLES:
            print(f"Unknown example: {name}")
            print(f"Available: {', '.join(EXAMPLES)}")
            sys.exit(1)

        module = f"morphis.examples.{name}"
        sys.argv = [module] + remaining
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    else:
        print(f"Unknown command: {args[0]}")
        print("Usage: morphis example <name>")
        sys.exit(1)


if __name__ == "__main__":
    main()
