"""
Stage - Dials

A dial is a value the playground can turn: a parameter with a domain. The
domains are the manifold code's own, so a panel of dials is a chart on a
domain:

    Interval(start, stop)    a number between the two, refused outside them
    Circle()                 an angle, wrapped onto the circle
    no domain                any number

and a boolean is a toggle, a Literal a choice among its values.

A group of dials is a Dials model. It knows nothing of any panel or key: it
validates every value against its domain where the value is set, describes
its dials as plain data a front end draws from, and reads and writes them by
path, so a dial is addressed the same way by a panel, a key, a script, and a
saved state:

    class Torus(Dials):
        major: float = Dial(1.0, Interval(0.2, 3.0))
        minor: float = Dial(0.4, Interval(0.05, 1.0))

    torus = Torus()
    torus.set("major", 1.4)
    torus.describe()      # what each dial is, for a front end to draw

A dial is absolute or a rate. An absolute dial is the value itself; a rate is
a time derivative, and the picture follows its history rather than its value.
"""

from typing import Any, Literal, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator
from pydantic.fields import FieldInfo

from morphis.manifold.domain import Interval


Kind = Literal["number", "angle", "choice", "toggle"]


def Dial(default: Any, domain: Interval | None = None, rate: bool = False, description: str | None = None) -> Any:
    """
    A field of a Dials model, with the domain its value must lie in.

    Args:
        default: The value the dial starts at
        domain: An interval the value is refused outside of, or a periodic
            interval it is wrapped onto; none for any number
        rate: Whether the dial is a time derivative rather than a value
        description: What the dial means, for the schema

    Returns:
        The field, to assign in a Dials class body
    """
    extra = {"dial": {"domain": None if domain is None else domain.model_dump(), "rate": rate}}
    result = Field(default=default, description=description, json_schema_extra=extra)

    return result


class DialSpec(BaseModel):
    """
    What a dial is, as plain data a front end draws from.

    Attributes:
        path: Where the dial is, from the group described (`shape.major`)
        kind: Drawn as a number, an angle, a choice, or a toggle
        domain: The interval the value lies in, if any
        choices: The values of a choice
        rate: Whether the dial is a time derivative
        quantity: What a rate is a rate of, worded ("angular rate"), or ""
    """

    model_config = ConfigDict(frozen=True)

    path: str
    kind: Kind
    domain: Interval | None = None
    choices: tuple[Any, ...] = ()
    rate: bool = False
    quantity: str = ""


class Dials(BaseModel):
    """
    A group of dials: every field is a dial, or a nested group of them.

    Values are checked against their domains on construction and on every
    assignment, so a value outside its domain is refused where it is set,
    with the range it must lie in. A field excluded from dumping is context
    the group carries, not a dial.
    """

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    @field_validator("*")
    @classmethod
    def _within_domain(cls, value: Any, info: ValidationInfo) -> Any:
        domain = _domain(cls.model_fields[info.field_name])
        if domain is not None and not domain.periodic and not domain.start <= value <= domain.stop:
            raise ValueError(f"a number from {domain.start:g} to {domain.stop:g}")

        if domain is not None and domain.periodic:
            result = domain.start + (value - domain.start) % domain.length
        else:
            result = value

        return result

    # =========================================================================
    # Paths
    # =========================================================================

    def get(self, path: str) -> Any:
        """
        Returns the value of the dial at a path, its keys joined by dots.
        """
        head, _, rest = path.partition(".")
        if rest:
            result = self._group(head).get(rest)
        else:
            result = self._read(head)

        return result

    def set(self, path: str, value: Any) -> None:
        """
        Set the dial at a path, its keys joined by dots. A value outside the
        dial's domain is refused with a ValueError saying what is allowed.
        """
        head, _, rest = path.partition(".")
        if rest:
            self._group(head).set(rest, value)
        else:
            self._write(head, value)

    def values(self) -> dict[str, Any]:
        """
        Returns every dial's value, by path.
        """
        result = {spec.path: self.get(spec.path) for spec in self.describe()}

        return result

    def describe(self, prefix: str = "") -> list[DialSpec]:
        """
        Returns what each dial is, in declaration order, nested groups in
        place, every path prefixed by the given one.
        """
        result = []
        for name, field in type(self).model_fields.items():
            value = getattr(self, name)
            if isinstance(value, Dials):
                result += value.describe(f"{prefix}{name}.")
            elif not field.exclude:
                result.append(_spec(f"{prefix}{name}", field))

        return result

    def _group(self, key: str) -> "Dials":
        group = getattr(self, key, None) if key in type(self).model_fields else None
        if not isinstance(group, Dials):
            raise KeyError(f"no group of dials named {key!r}")

        return group

    def _read(self, key: str) -> Any:
        self._check(key)

        return getattr(self, key)

    def _write(self, key: str, value: Any) -> None:
        self._check(key)
        setattr(self, key, value)

    def _check(self, key: str) -> None:
        field = type(self).model_fields.get(key)
        if field is None or field.exclude:
            raise KeyError(f"no dial named {key!r}")


# =============================================================================
# Fields
# =============================================================================


def _dial(field: FieldInfo) -> dict:
    extra = field.json_schema_extra if isinstance(field.json_schema_extra, dict) else {}

    return extra.get("dial", {})


def _domain(field: FieldInfo) -> Interval | None:
    domain = _dial(field).get("domain")
    result = None if domain is None else Interval(domain["start"], domain["stop"], periodic=domain["periodic"])

    return result


def _spec(path: str, field: FieldInfo) -> DialSpec:
    domain = _domain(field)
    annotation = field.annotation
    if annotation is bool:
        kind, choices = "toggle", ()
    elif get_origin(annotation) is Literal:
        kind, choices = "choice", get_args(annotation)
    elif domain is not None and domain.periodic:
        kind, choices = "angle", ()
    else:
        kind, choices = "number", ()
    result = DialSpec(path=path, kind=kind, domain=domain, choices=choices, rate=_dial(field).get("rate", False))

    return result
