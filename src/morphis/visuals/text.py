"""
Text Annotations for Scene Visualization

Text provides 3D text annotations that can be added to scenes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import array, zeros
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator

from morphis.visuals.theme import Color


if TYPE_CHECKING:
    from morphis.elements.vector import Vector


class Text:
    """
    3D text annotation for scenes.

    Text objects can be positioned in 3D space and styled with font
    settings. They are added to scenes like other elements.

    Attributes:
        content: The text string to display
        position: 3D position (Vector or tuple)
        font_size: Font size in points
        font_family: Font family name
        color: Text color (None = theme foreground)
        anchor: Text anchor point ("center", "left", "right", "top", "bottom")

    Examples:
        # Simple label
        label = Text("v₁", position=(1, 0, 0))
        scene.add(label)

        # Styled text
        title = Text(
            "Rotation",
            position=(0, 0, 2),
            font_size=16,
            color=(1, 0.8, 0.2),
        )
        scene.add(title)

        # Using Vector position
        offset = v1.data + array([0.1, 0, 0])
        label = Text("v₁", position=offset)
    """

    def __init__(
        self,
        content: str,
        position: "Vector | tuple | NDArray" = (0, 0, 0),
        font_size: int = 12,
        font_family: str = "arial",
        color: Color | None = None,
        anchor: str = "center",
    ):
        """
        Create a text annotation.

        Args:
            content: Text string to display
            position: 3D position (can be Vector, tuple, or array)
            font_size: Font size in points
            font_family: Font family name
            color: RGB color tuple (None = use theme foreground)
            anchor: Anchor point for positioning
        """
        self.content = content
        self._position = self._normalize_position(position)
        self.font_size = font_size
        self.font_family = font_family
        self.color = color
        self.anchor = anchor

    @staticmethod
    def _normalize_position(pos) -> NDArray:
        """Convert position to 3D numpy array."""
        from morphis.elements.vector import Vector

        if isinstance(pos, Vector):
            data = pos.data
        else:
            data = array(pos, dtype=float)

        # Ensure 3D
        if len(data) < 3:
            result = zeros(3)
            result[: len(data)] = data
            return result

        return data[:3]

    @property
    def position(self) -> NDArray:
        """Current 3D position."""
        return self._position

    @position.setter
    def position(self, value):
        """Set position."""
        self._position = self._normalize_position(value)

    def __repr__(self) -> str:
        return f"Text({self.content!r}, position={tuple(self._position)})"


class TextStyle(BaseModel):
    """Style configuration for text rendering."""

    model_config = ConfigDict(frozen=True)

    font_size: int = 12
    font_family: str = "arial"
    color: Color | None = None
    anchor: str = "center"
    bold: bool = False
    italic: bool = False

    @field_validator("anchor")
    @classmethod
    def _validate_anchor(cls, v: str) -> str:
        valid = {"center", "left", "right", "top", "bottom", "top-left", "top-right", "bottom-left", "bottom-right"}
        if v not in valid:
            raise ValueError(f"anchor must be one of {valid}")
        return v
