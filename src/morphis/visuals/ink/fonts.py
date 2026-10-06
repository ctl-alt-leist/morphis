"""
Font Registration

Many system serif families (Palatino, Charter, Iowan Old Style on macOS) ship as
.ttc collections. Matplotlib registers only the first face of a collection, so
the italic that mathematics needs is invisible to it. Here each face of such a
collection is extracted once to a user cache and registered, making the
family's italic and bold available to mathtext.
"""

from __future__ import annotations

from pathlib import Path

from fontTools.ttLib import TTCollection
from matplotlib import font_manager


FONT_CACHE = Path.home() / ".cache" / "morphis" / "fonts"

# Ships with matplotlib, so figures render anywhere when the theme font is missing
FALLBACK_FONT = "DejaVu Serif"

_registered: set[str] = set()


def find_font(family: str) -> Path | None:
    """Path of the font file for a family, or None when it is not installed."""
    try:
        source = Path(font_manager.findfont(font_manager.FontProperties(family=family), fallback_to_default=False))
    except ValueError:
        source = None

    return source


def register_font(family: str) -> None:
    """Make every face of a font family available to matplotlib, extracting collections if needed."""
    source = find_font(family) if family not in _registered else None
    is_collection = source is not None and source.suffix.lower() == ".ttc"

    if is_collection:
        FONT_CACHE.mkdir(parents=True, exist_ok=True)
        for face in TTCollection(str(source)).fonts:
            names = face["name"]
            target = FONT_CACHE / f"{names.getDebugName(1)}-{names.getDebugName(2)}.ttf".replace(" ", "_")
            if not target.exists():
                face.save(str(target))
            font_manager.fontManager.addfont(str(target))

    _registered.add(family)


def font_settings(family: str) -> dict[str, str]:
    """
    Matplotlib rc settings that set text and mathtext in one serif family.

    Falls back to DejaVu Serif, which ships with matplotlib, when the family is
    not installed, so a figure still renders on a machine without it.
    """
    chosen = family if find_font(family) is not None else FALLBACK_FONT
    register_font(chosen)
    settings = {
        "font.family": chosen,
        "mathtext.fontset": "custom",
        "mathtext.rm": chosen,
        "mathtext.it": f"{chosen}:style=italic",
        "mathtext.bf": f"{chosen}:weight=bold",
        "mathtext.fallback": "stix",
    }

    return settings
