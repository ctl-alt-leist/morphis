"""
Stage - The Catalog

What can be added to a stage from a front end, where there is no Python to
hand an object in: a picker of entries, in sections, each a function from the
space to a new object with its default dials. A new kind of object is offered
by registering an entry, with no change to the stage or any front end.

    @entry("Vector", section="Elements")
    def _vector(space: Metric) -> Addition:
        return Addition(basis_vector(1, space))

    stage.create("Vector")        # added as "vector", selected-ready

An expression is written rather than chosen: a line of text evaluated in the
space, its basis vectors named by their labels (e1, e2, and e1dot for the
partner of e1 in a Kähler space) and the blades of single-digit labels by
theirs (e12, e123), with pi, sqrt, sin, cos, and exp at hand:

    e1 ^ e3
    0.5 * (e1 + e2)
    cos(pi / 6) * e12 + sin(pi / 6) * e34

The line is kept as the actor's shape dial, so editing it rebuilds the
element, and a line that does not give a vector, bivector, or trivector of
the space is refused where it is typed.
"""

from collections.abc import Callable
from itertools import permutations
from typing import Any, Literal, NamedTuple

from numpy import cos, pi, sin, sqrt
from pydantic import Field, ValidationInfo, field_validator

from morphis.elements import Frame, Metric, Vector, basis_vector, basis_vectors
from morphis.operations.exponential import exp_vector
from morphis.stage.dial import Dial, Dials


class Addition(NamedTuple):
    """What a catalog entry adds: the object, its dials, its own law, and how it is drawn."""

    obj: Any
    dials: Dials | None = None
    step: Callable | None = None
    options: dict = {}


class Entry(NamedTuple):
    """
    One line of the catalog.

    Attributes:
        name: The name it is offered under, and the name given to what it adds
        section: The heading it is listed under
        kind: Whether it adds an actor or a group
        make: Builds the addition for a space; none for a group
    """

    name: str
    section: str
    kind: Literal["actor", "group"]
    make: Callable[[Metric], Addition] | None


CATALOG: dict[str, Entry] = {}


def entry(name: str, section: str) -> Callable[[Callable[[Metric], Addition]], Callable[[Metric], Addition]]:
    """
    Register a catalog entry: a function from the space to an addition.
    """

    def register(make: Callable[[Metric], Addition]) -> Callable[[Metric], Addition]:
        CATALOG[name] = Entry(name, section, "actor", make)

        return make

    return register


def sections() -> dict[str, list[str]]:
    """
    Returns the catalog's entry names under their sections, in the order registered.
    """
    result: dict[str, list[str]] = {}
    for line in CATALOG.values():
        result.setdefault(line.section, []).append(line.name)

    return result


# =============================================================================
# Expressions
# =============================================================================


class Expression(Dials):
    """
    The line of text an element is written as.

    Attributes:
        metric: The space the line is read in
        text: The line, checked by evaluating it where it is set
    """

    metric: Metric = Field(exclude=True, repr=False)
    text: str = Dial("e1")

    @field_validator("text")
    @classmethod
    def _readable(cls, text: str, info: ValidationInfo) -> str:
        evaluate(text, info.data["metric"])

        return text


def evaluate(text: str, metric: Metric) -> Vector:
    """
    Returns the element a line of text gives in a space.

    Raises:
        ValueError: If the line cannot be read, or does not give a vector,
            bivector, or trivector of the space
    """
    try:
        result = eval(text, {"__builtins__": {}}, namespace(metric))  # noqa: S307 (the user's own line, in their own session)
    except Exception as error:
        raise ValueError(f"cannot read {text!r}: {error}") from error

    if not (isinstance(result, Vector) and result.grade in (1, 2, 3) and result.metric == metric):
        raise ValueError(f"{text!r} does not give a vector, bivector, or trivector of the space")

    return result


def namespace(metric: Metric) -> dict[str, Any]:
    """
    Returns the names a line is read with: the basis vectors and the blades of
    single-digit labels, and a few functions.
    """
    vectors = dict(zip(metric.labels, basis_vectors(metric), strict=True))
    names = {_name(label): vector for label, vector in vectors.items()}
    digits = [label for label in vectors if isinstance(label, int) and 0 <= label <= 9]
    for grade in (2, 3):
        for labels in permutations(digits, grade):
            blade = vectors[labels[0]]
            for label in labels[1:]:
                blade = blade ^ vectors[label]
            names["e" + "".join(str(label) for label in labels)] = blade
    result = {**names, "pi": pi, "sqrt": sqrt, "sin": sin, "cos": cos, "exp": _exp}

    return result


def _name(label: Any) -> str:
    return f"e{label}" if isinstance(label, int) else f"e{label.mode}dot"


def _exp(x: Any) -> Any:
    from numpy import exp

    return exp_vector(x) if isinstance(x, Vector) else exp(x)


def expression(dials: Expression) -> Vector:
    """The element an expression's line gives."""
    return evaluate(dials.text, dials.metric)


# =============================================================================
# The Entries
# =============================================================================


@entry("Vector", section="Elements")
def _vector(space: Metric) -> Addition:
    first = next((label for label, kind in zip(space.labels, space.kinds, strict=True) if kind == "space"), 1)

    return Addition(basis_vector(first, space))


@entry("Bivector", section="Elements")
def _bivector(space: Metric) -> Addition:
    first, second = space.labels[-2:]

    return Addition(basis_vector(first, space) ^ basis_vector(second, space))


@entry("Frame", section="Elements")
def _frame(space: Metric) -> Addition:
    return Addition(Frame(*basis_vectors(space)), options={"filled": True})


@entry("Expression", section="Elements")
def _expression(space: Metric) -> Addition:
    return Addition(expression, Expression(metric=space))


CATALOG["Group"] = Entry("Group", "Groups", "group", None)
