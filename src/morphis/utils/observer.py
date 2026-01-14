"""
Observer - Watch and observe GA objects

A general-purpose observer that holds references to GA objects and can
read their current state at any time. The observer is passive - it never
modifies the objects, only reads them.

This is useful for:
- Animation systems that need to snapshot object state over time
- Debugging to watch how objects change
- Analysis that needs to watch multiple related objects

Example:
    from morphis.utils.observer import Observer
    from morphis.elements import basis_vectors, euclidean

    e1, e2, e3 = basis_vectors(euclidean(3))
    q = e1 ^ e2 ^ e3

    obs = Observer()
    obs.watch(e1, e2, e3, q)

    # ... modify objects ...
    q.data[...] = transformed.data

    # Observer sees the changes
    print(obs.snapshot())
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator

from numpy import copy as np_copy, ndarray, stack
from numpy.linalg import norm

from morphis.elements import GAModel


if TYPE_CHECKING:
    from morphis.elements import Blade


@dataclass
class TrackedObject:
    """Internal record for a tracked object."""

    obj: GAModel
    obj_id: int
    name: str | None
    baseline: ndarray | None = None  # For computing diffs
    cached_spanning_vectors: list["Blade"] | None = field(default=None, repr=False)


class Observer:
    """
    Observes GA objects by holding references to them.

    The observer is completely passive - it reads object state but never
    modifies it. Your code controls the objects; the observer just watches.

    Example:
        obs = Observer()
        obs.watch(blade1, blade2)

        for t in times:
            # Your code modifies objects
            blade1.data[...] = transform(blade1)

            # Observer reads current state
            state = obs.snapshot()
    """

    def __init__(self):
        self._tracked: dict[int, TrackedObject] = {}
        self._names: dict[str, int] = {}  # name -> obj_id for lookup

    def watch(self, *objects: GAModel, names: list[str] | None = None) -> "Observer":
        """
        Register one or more objects to observe.

        Args:
            *objects: GA objects to watch (Blade, MultiVector, etc.)
            names: Optional list of names for the objects (for easier lookup)

        Returns:
            Self for chaining

        Example:
            obs.watch(e1, e2, e3)
            obs.watch(q, names=["trivector"])
        """
        if names is not None and len(names) != len(objects):
            raise ValueError(f"Got {len(names)} names for {len(objects)} objects")

        for i, obj in enumerate(objects):
            obj_id = id(obj)
            name = names[i] if names else None

            # Store baseline (initial state) for diff computation
            baseline = self._get_data(obj)
            if baseline is not None:
                baseline = np_copy(baseline)

            self._tracked[obj_id] = TrackedObject(
                obj=obj,
                obj_id=obj_id,
                name=name,
                baseline=baseline,
            )

            if name:
                self._names[name] = obj_id

        return self

    # Alias for backward compatibility
    track = watch

    def unwatch(self, *objects: GAModel) -> "Observer":
        """
        Stop watching one or more objects.

        Returns:
            Self for chaining
        """
        for obj in objects:
            obj_id = id(obj)
            if obj_id in self._tracked:
                tracked = self._tracked.pop(obj_id)
                if tracked.name and tracked.name in self._names:
                    del self._names[tracked.name]

        return self

    # Alias for backward compatibility
    untrack = unwatch

    def clear(self) -> "Observer":
        """
        Stop tracking all objects.

        Returns:
            Self for chaining
        """
        self._tracked.clear()
        self._names.clear()
        return self

    def _get_data(self, obj: GAModel) -> ndarray | None:
        """Extract the data array from a GA object."""
        if hasattr(obj, "data"):
            return obj.data
        elif hasattr(obj, "components"):
            # MultiVector - return dict of component data
            # For simplicity, just return the first component's data
            for blade in obj.components.values():
                return blade.data
        return None

    def get(self, obj_or_name: GAModel | str) -> ndarray | None:
        """
        Get the current data for a tracked object.

        Args:
            obj_or_name: The object or its name

        Returns:
            Current data array (copy), or None if not found
        """
        if isinstance(obj_or_name, str):
            obj_id = self._names.get(obj_or_name)
            if obj_id is None:
                return None
        else:
            obj_id = id(obj_or_name)

        if obj_id not in self._tracked:
            return None

        tracked = self._tracked[obj_id]
        data = self._get_data(tracked.obj)
        return np_copy(data) if data is not None else None

    def snapshot(self) -> dict[int, ndarray]:
        """
        Get current state of all tracked objects.

        Returns:
            Dict mapping object id to current data (copies)
        """
        result = {}
        for obj_id, tracked in self._tracked.items():
            data = self._get_data(tracked.obj)
            if data is not None:
                result[obj_id] = np_copy(data)
        return result

    def snapshot_named(self) -> dict[str, ndarray]:
        """
        Get current state of named objects only.

        Returns:
            Dict mapping name to current data (copies)
        """
        result = {}
        for _obj_id, tracked in self._tracked.items():
            if tracked.name:
                data = self._get_data(tracked.obj)
                if data is not None:
                    result[tracked.name] = np_copy(data)
        return result

    def reset_baseline(self, *objects: GAModel) -> "Observer":
        """
        Reset the baseline (for diff computation) to current state.

        If no objects specified, resets all tracked objects.

        Returns:
            Self for chaining
        """
        if not objects:
            objects = [t.obj for t in self._tracked.values()]

        for obj in objects:
            obj_id = id(obj)
            if obj_id in self._tracked:
                tracked = self._tracked[obj_id]
                data = self._get_data(tracked.obj)
                tracked.baseline = np_copy(data) if data is not None else None

        return self

    def diff(self, obj_or_name: GAModel | str) -> ndarray | None:
        """
        Compute difference from baseline for an object.

        Args:
            obj_or_name: The object or its name

        Returns:
            (current - baseline) array, or None if not tracked
        """
        if isinstance(obj_or_name, str):
            obj_id = self._names.get(obj_or_name)
            if obj_id is None:
                return None
        else:
            obj_id = id(obj_or_name)

        if obj_id not in self._tracked:
            return None

        tracked = self._tracked[obj_id]
        if tracked.baseline is None:
            return None

        current = self._get_data(tracked.obj)
        if current is None:
            return None

        return current - tracked.baseline

    def diff_norm(self, obj_or_name: GAModel | str) -> float | None:
        """
        Compute norm of difference from baseline.

        Args:
            obj_or_name: The object or its name

        Returns:
            ||current - baseline||, or None if not tracked
        """
        d = self.diff(obj_or_name)
        return float(norm(d)) if d is not None else None

    def objects(self) -> list[GAModel]:
        """Return list of all tracked objects."""
        return [t.obj for t in self._tracked.values()]

    def ids(self) -> list[int]:
        """Return list of all tracked object IDs."""
        return list(self._tracked.keys())

    def names(self) -> list[str]:
        """Return list of all named objects."""
        return list(self._names.keys())

    def __len__(self) -> int:
        """Number of tracked objects."""
        return len(self._tracked)

    def __contains__(self, obj: GAModel) -> bool:
        """Check if an object is being tracked."""
        return id(obj) in self._tracked

    def __iter__(self) -> Iterator[GAModel]:
        """Iterate over tracked objects."""
        return iter(t.obj for t in self._tracked.values())

    def __getitem__(self, key: str | int) -> GAModel | None:
        """
        Get tracked object by name or id.

        Args:
            key: Object name (str) or object id (int)

        Returns:
            The tracked object, or None if not found
        """
        if isinstance(key, str):
            obj_id = self._names.get(key)
            if obj_id is None:
                return None
        else:
            obj_id = key

        if obj_id in self._tracked:
            return self._tracked[obj_id].obj
        return None

    def print_state(self, prefix: str = ""):
        """Print current state of all tracked objects (for debugging)."""
        print(f"{prefix}Observer tracking {len(self)} objects:")
        for _obj_id, tracked in self._tracked.items():
            name_str = f" ({tracked.name})" if tracked.name else ""
            data = self._get_data(tracked.obj)
            obj_type = type(tracked.obj).__name__
            print(f"{prefix}  [{obj_type}]{name_str}: {data}")

    def spanning_vectors(self, obj_or_name: GAModel | str) -> tuple["Blade", ...] | None:
        """
        Get the spanning vectors for a tracked blade.

        For a k-blade B = v₁ ∧ v₂ ∧ ... ∧ vₖ, returns (v₁, v₂, ..., vₖ).
        This is the primary way to extract visualization-ready data from
        tracked blades.

        Args:
            obj_or_name: The blade or its name

        Returns:
            Tuple of grade-1 Blades that span the blade, or None if not a Blade
        """
        from morphis.elements import Blade

        if isinstance(obj_or_name, str):
            obj_id = self._names.get(obj_or_name)
            if obj_id is None:
                return None
        else:
            obj_id = id(obj_or_name)

        if obj_id not in self._tracked:
            return None

        tracked = self._tracked[obj_id]

        if not isinstance(tracked.obj, Blade):
            return None

        return tracked.obj.spanning_vectors()

    def spanning_vectors_as_array(self, obj_or_name: GAModel | str) -> ndarray | None:
        """
        Get spanning vectors as a stacked numpy array.

        Returns shape (k, dim) where k is the blade grade and dim is the vector
        space dimension. This is convenient for direct use in visualization code.

        Args:
            obj_or_name: The blade or its name

        Returns:
            Array of shape (k, dim) containing the spanning vectors, or None
        """
        vectors = self.spanning_vectors(obj_or_name)
        if vectors is None or len(vectors) == 0:
            return None

        return stack([v.data for v in vectors])

    def capture_state(self, obj_or_name: GAModel | str) -> dict | None:
        """
        Capture complete visualization state for a tracked blade.

        This is the primary method for animation systems to extract all
        needed data for rendering a blade at the current time.

        Args:
            obj_or_name: The blade or its name

        Returns:
            Dict with keys:
                - 'grade': Blade grade
                - 'dim': Vector space dimension
                - 'data': Raw blade data (copy)
                - 'spanning_vectors': Array of shape (k, dim)
                - 'context': Geometric context (PGA, etc.) or None
            Or None if not a tracked Blade
        """
        from morphis.elements import Blade

        if isinstance(obj_or_name, str):
            obj_id = self._names.get(obj_or_name)
            if obj_id is None:
                return None
        else:
            obj_id = id(obj_or_name)

        if obj_id not in self._tracked:
            return None

        tracked = self._tracked[obj_id]

        if not isinstance(tracked.obj, Blade):
            return None

        blade = tracked.obj
        vectors = blade.spanning_vectors()

        return {
            "grade": blade.grade,
            "dim": blade.dim,
            "data": np_copy(blade.data),
            "spanning_vectors": stack([v.data for v in vectors]) if vectors else None,
            "context": blade.context,
        }
