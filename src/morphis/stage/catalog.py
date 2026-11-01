"""
Stage - The Catalog

What can be added to a stage from a front end, where there is no Python to
hand an object in: a picker of entries, in sections, each a function from the
space to a new object with its default dials. A new kind of object is offered
by registering an entry, with no change to the stage or any front end.

    @entry("Frame", section="Elements")
    def _frame(space: Metric) -> Addition:
        return Addition(Frame(*basis_vectors(space)))

    stage.create("Frame")        # added as "frame", selected-ready

A vector is any k-vector, written rather than chosen: its grade is picked
when it is added, and it is a line of text evaluated in the space, its basis
vectors named by their labels (e1, e2, and e1dot for the partner of e1 in a
Kähler space) and the blades of single-digit labels by theirs, of any grade
and in any order (e12, e31, e123), with pi, sqrt, sin, cos, and exp at hand:

    e1 ^ e3
    0.5 * (e1 + e2)
    cos(pi / 6) * e12 + sin(pi / 6) * e34

The grade and the line are the actor's shape dials, so editing either
rebuilds the element. A line that does not give a k-vector of the grade is
refused where it is typed; a new grade starts the line again at the first
blade of that grade.
"""

from collections.abc import Callable
from typing import Any, Literal, NamedTuple

from numpy import cos, pi, sin, sqrt
from pydantic import Field, ValidationInfo, field_validator, model_validator

from morphis.elements import Frame, Metric, Vector, basis_vectors
from morphis.manifold import Circle, Interval, Mesh, Patch
from morphis.operations.exponential import exp_vector
from morphis.stage.dial import Dial, Dials
from morphis.stage.motion import planes


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
        kind: Whether it adds an actor, a group, or an operation
        make: Builds the addition for a space, or for an operation the values
            its dials start at, by path; none for a group
        applies: Whether it can be added in a space
        asks: The path of a choice dial a front end asks for as soon as the
            entry is added, since it has no sensible default, or none
    """

    name: str
    section: str
    kind: Literal["actor", "group", "operation"]
    make: Callable[[Metric], Any] | None
    applies: Callable[[Metric], bool] = lambda space: True
    asks: str | None = None


CATALOG: dict[str, Entry] = {}


def entry(
    name: str, section: str, applies: Callable[[Metric], bool] = lambda space: True, asks: str | None = None
) -> Callable[[Callable[[Metric], Addition]], Callable[[Metric], Addition]]:
    """
    Register a catalog entry: a function from the space to an addition.
    """

    def register(make: Callable[[Metric], Addition]) -> Callable[[Metric], Addition]:
        CATALOG[name] = Entry(name, section, "actor", make, applies, asks)

        return make

    return register


def operation(
    name: str, applies: Callable[[Metric], bool] = lambda space: True
) -> Callable[[Callable[[Metric], dict]], Callable[[Metric], dict]]:
    """
    Register an operation in the catalog: a function from the space to the
    values its dials start at, by path.
    """

    def register(make: Callable[[Metric], dict]) -> Callable[[Metric], dict]:
        CATALOG[name] = Entry(name, "Operations", "operation", make, applies)

        return make

    return register


def sections(space: Metric | None = None) -> dict[str, list[str]]:
    """
    Returns the catalog's entry names under their sections, in the order
    registered: those that apply in a space, given one.
    """
    result: dict[str, list[str]] = {}
    for line in CATALOG.values():
        if space is None or line.applies(space):
            result.setdefault(line.section, []).append(line.name)

    return result


# =============================================================================
# Expressions
# =============================================================================


# The grades a k-vector can be drawn at in a picture of three dimensions
DRAWN_GRADES = (1, 2, 3)


class Expression(Dials):
    """
    A k-vector, as its grade and the line of text it is written as.

    Attributes:
        metric: The space the line is read in
        grade: The grade of the k-vector, at most the dimension of the space
        text: The line, checked where it is set to give a k-vector of the grade
    """

    metric: Metric = Field(exclude=True, repr=False)
    grade: Literal[DRAWN_GRADES] = 1
    text: str = Dial("e1")

    @field_validator("grade")
    @classmethod
    def _within_space(cls, grade: int, info: ValidationInfo) -> int:
        dim = info.data["metric"].dim
        if grade > dim:
            raise ValueError(f"grade at most {dim} in this space")

        return grade

    @field_validator("text")
    @classmethod
    def _readable(cls, text: str, info: ValidationInfo) -> str:
        evaluate(text, info.data["metric"], info.data.get("grade"))

        return text

    @model_validator(mode="after")
    def _of_the_grade(self) -> "Expression":
        # A new grade starts the line again, written to the dict so the line is not checked twice
        if evaluate(self.text, self.metric).grade != self.grade:
            self.__dict__["text"] = first_blade(self.metric, self.grade)

        return self

    def about(self) -> list[str]:
        """What the expression is: its own line."""
        return [f"written  {self.text}"]


def first_blade(metric: Metric, grade: int) -> str:
    """
    Returns the line of the first blade of a grade: the first spatial
    directions, or the first of the space when it has too few, by the blade's
    own name (e12) when their labels are digits, and wedged otherwise.
    """
    spatial = [label for label, kind in zip(metric.labels, metric.kinds, strict=True) if kind in ("space", "modes")]
    labels = spatial[:grade] if len(spatial) >= grade else list(metric.labels[:grade])
    if all(isinstance(label, int) and label <= 9 for label in labels):
        result = "e" + "".join(str(label) for label in labels)
    else:
        result = " ^ ".join(_name(label) for label in labels)

    return result


def evaluate(text: str, metric: Metric, grade: int | None = None) -> Vector:
    """
    Returns the element a line of text gives in a space.

    Raises:
        ValueError: If the line cannot be read, or does not give a k-vector of
            the space that can be drawn, of the grade if one is given
    """
    try:
        result = eval(text, {"__builtins__": {}}, Names(metric))  # noqa: S307 (the user's own line, in their own session)
    except Exception as error:
        raise ValueError(f"cannot read {text!r}: {error}") from error

    if not (isinstance(result, Vector) and result.grade in DRAWN_GRADES and result.metric == metric):
        raise ValueError(f"{text!r} does not give a vector, bivector, or trivector of the space")

    if grade is not None and result.grade != grade:
        raise ValueError(f"{text!r} is of grade {result.grade}, not {grade}")

    return result


class Names(dict):
    """
    The names a line is read with: the basis vectors and a few functions, and
    the blade of any string of distinct single-digit labels, made when it is
    first read rather than listed, since there are factorially many.
    """

    def __init__(self, metric: Metric) -> None:
        vectors = dict(zip(metric.labels, basis_vectors(metric), strict=True))
        names = {_name(label): vector for label, vector in vectors.items()}
        super().__init__({**names, "pi": pi, "sqrt": sqrt, "sin": sin, "cos": cos, "exp": _exp})
        self.digits = {str(label): vector for label, vector in vectors.items() if isinstance(label, int) and label <= 9}

    def __missing__(self, name: str) -> Vector:
        written = name.removeprefix("e")
        is_blade = name.startswith("e") and len(written) > 1 and len(set(written)) == len(written)
        if not (is_blade and set(written) <= set(self.digits)):
            raise KeyError(name)
        result = self.digits[written[0]]
        for digit in written[1:]:
            result = result ^ self.digits[digit]

        return result


def _name(label: Any) -> str:
    return f"e{label}" if isinstance(label, int) else f"e{label.mode}dot"


def _exp(x: Any) -> Any:
    from numpy import exp

    return exp_vector(x) if isinstance(x, Vector) else exp(x)


def expression(dials: Expression) -> Vector:
    """The element an expression's line gives."""
    return evaluate(dials.text, dials.metric, dials.grade)


# =============================================================================
# The Entries
# =============================================================================


@entry("Vector", section="Elements", asks="shape.grade")
def _vector(space: Metric) -> Addition:
    return Addition(expression, Expression(metric=space, text=first_blade(space, 1)))


@entry("Frame", section="Elements")
def _frame(space: Metric) -> Addition:
    return Addition(Frame(*basis_vectors(space)), options={"filled": True})


# =============================================================================
# Surfaces
# =============================================================================


def _spatial(space: Metric) -> list[int]:
    """The storage positions of the directions a surface is drawn in: the spatial ones, in order."""
    return [n for n, kind in enumerate(space.kinds) if kind in ("space", "modes")]


def _surface(space: Metric, shape: tuple[int, int], domain: Any, components: Callable) -> Mesh:
    """A surface as a mesh: components along the spatial directions, zero along the rest."""
    places = _spatial(space)

    def embedded(u, v):
        point = [0.0 * u] * space.dim
        for place, component in zip(places, components(u, v), strict=False):
            point[place] = component

        return point

    return Patch(embedded, domain, space).mesh(shape)


def _drawn_in(count: int) -> Callable[[Metric], bool]:
    return lambda space: len(_spatial(space)) >= count and not space.is_projective


class Sphere(Dials):
    metric: Metric = Field(exclude=True, repr=False)
    radius: float = Dial(1.0, Interval(0.1, 5.0))
    resolution: int = Dial(24, Interval(4, 96))

    def about(self) -> list[str]:
        """What the sphere is, as mathematics."""
        return ["sphere  S²", "x = r (sin θ cos φ, sin θ sin φ, cos θ)", f"r = {self.radius:.2f}"]


class Torus(Dials):
    """
    A torus: a ring of the major radius swept by a circle of the minor one in
    three dimensions, and from four dimensions on the flat Clifford torus, the
    product of two circles of those radii in orthogonal planes.
    """

    metric: Metric = Field(exclude=True, repr=False)
    major: float = Dial(1.0, Interval(0.1, 5.0))
    minor: float = Dial(0.4, Interval(0.05, 5.0))
    resolution: int = Dial(32, Interval(4, 96))

    def about(self) -> list[str]:
        """What the torus is, as mathematics."""
        if len(_spatial(self.metric)) >= 4:
            lines = ["Clifford torus  S¹ × S¹", "x = (a cos θ, a sin θ, b cos φ, b sin φ)", "flat"]
        else:
            lines = ["ring torus  S¹ × S¹", "x = ((a + b cos φ) cos θ, (a + b cos φ) sin θ, b sin φ)"]
        result = [*lines, f"a = {self.major:.2f}   b = {self.minor:.2f}"]

        return result


class Square(Dials):
    metric: Metric = Field(exclude=True, repr=False)
    side: float = Dial(2.0, Interval(0.1, 10.0))
    resolution: int = Dial(8, Interval(1, 64))

    def about(self) -> list[str]:
        """What the plane is, as mathematics."""
        return ["plane  e₁₂", "x = (u, v),  |u|, |v| ≤ ℓ / 2", f"ℓ = {self.side:.2f}"]


def sphere(dials: Sphere) -> Mesh:
    r, n = dials.radius, dials.resolution

    return _surface(
        dials.metric,
        (n, 2 * n),
        Interval(0.0, pi) * Circle(),
        lambda θ, φ: [r * sin(θ) * cos(φ), r * sin(θ) * sin(φ), r * cos(θ)],
    )


def torus(dials: Torus) -> Mesh:
    a, b, n = dials.major, dials.minor, dials.resolution
    if len(_spatial(dials.metric)) >= 4:
        components = lambda θ, φ: [a * cos(θ), a * sin(θ), b * cos(φ), b * sin(φ)]  # noqa: E731
    else:
        components = lambda θ, φ: [(a + b * cos(φ)) * cos(θ), (a + b * cos(φ)) * sin(θ), b * sin(φ)]  # noqa: E731

    return _surface(dials.metric, (n, n), Circle() * Circle(), components)


def square(dials: Square) -> Mesh:
    half, n = 0.5 * dials.side, dials.resolution

    return _surface(dials.metric, (n + 1, n + 1), Interval(-half, half) * Interval(-half, half), lambda u, v: [u, v])


@entry("Sphere", section="Surfaces", applies=_drawn_in(3))
def _sphere(space: Metric) -> Addition:
    return Addition(sphere, Sphere(metric=space), options={"opacity": 0.6})


@entry("Torus", section="Surfaces", applies=_drawn_in(3))
def _torus(space: Metric) -> Addition:
    return Addition(torus, Torus(metric=space), options={"opacity": 0.6})


@entry("Plane", section="Surfaces", applies=_drawn_in(2))
def _plane(space: Metric) -> Addition:
    return Addition(square, Square(metric=space), options={"opacity": 0.6})


# =============================================================================
# Operations
# =============================================================================


def _planes_of(kind: str) -> Callable[[Metric], list]:
    return lambda space: [plane for plane in planes(space) if plane.kind == kind]


# Operations start at rest, as every addition does: nothing moves until a dial is turned.


@operation("Rotation", applies=lambda space: bool(_planes_of("rotation")(space)))
def _rotation(space: Metric) -> dict:
    return {}


@operation("Translation")
def _translation(space: Metric) -> dict:
    return {}


@operation("Boost", applies=lambda space: bool(_planes_of("boost")(space)))
def _boost(space: Metric) -> dict:
    return {}


CATALOG["Group"] = Entry("Group", "Groups", "group", None)
