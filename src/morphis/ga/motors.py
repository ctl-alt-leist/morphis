"""
Geometric Algebra - Motors (PGA Transformations)

Motors represent rigid transformations in projective geometric algebra.
A motor is a MultiVector with grades {0, 2} that operates via the sandwich
product: p' = M p M†

All operations support collection dimensions via einsum broadcasting.
"""

from numpy import array, cos, ndarray, newaxis, ones, sin, zeros
from numpy.typing import NDArray

from morphis.ga.context import degenerate
from morphis.ga.geometric import geometric, inverse, reverse
from morphis.ga.model import Blade, Metric, MultiVector, pga, scalar_blade
from morphis.geometry.projective import euclidean as to_euclidean, point


class Motor(MultiVector):
    """
    Motor for rigid transformations in PGA.

    A motor unifies rotations, translations, and screw motions as a single
    algebraic object. Internally it's a MultiVector with grades {0, 2}.

    The bivector part decomposes as:
        - Euclidean: B^{mn} e_{mn} (m,n ≠ 0) → rotation plane
        - Degenerate: T^{0m} e_{0m} → translation direction

    All transformations operate via sandwich product: p' = M p M†
    """

    def __init__(self, components: dict[int, Blade], dim: int, cdim: int):
        """
        Initialize motor from components.

        Args:
            components: Dict mapping grade to Blade (must have keys {0, 2})
            dim: PGA dimension (d+1 for d-dimensional Euclidean space)
            cdim: Number of collection dimensions
        """
        # Validate structure
        if set(components.keys()) - {0, 2}:
            raise ValueError(f"Motor must have only grades {{0, 2}}, got {components.keys()}")

        super().__init__(components=components, dim=dim, cdim=cdim)

        # Set all components to PGA context
        for blade in self.components.values():
            blade.context = degenerate.projective

    @classmethod
    def rotor(cls, B: Blade, angle: float | NDArray) -> "Motor":
        """
        Pure rotation about origin.

        M = exp(-B θ/2) = cos(θ/2) - sin(θ/2) B

        Args:
            B: Euclidean bivector (grade-2, no e_0 components) defining plane
            angle: Rotation angle in radians (supports collection dims)

        Returns:
            Motor representing pure rotation

        Example:
            # Single rotation
            B = bivector_blade([[0,0,0,0,0,1]], dim=4)  # e_{12} plane
            M = Motor.rotor(B, pi/2)

            # Collection of rotations with different angles
            B = bivector_blade(data, dim=4, cdim=2)  # (n, m, d, d)
            angles = linspace(0, 2*pi, n*m).reshape(n, m)
            M = Motor.rotor(B, angles)  # Broadcasting
        """
        angle = array(angle)
        dim = B.dim

        # Determine collection dimensions
        # If angle is scalar (ndim=0), use B's cdim
        # If angle has dimensions, it adds collection dimensions
        if angle.ndim == 0:
            cdim = B.cdim
            half_angle = angle / 2
        else:
            # angle provides collection dimensions
            cdim = angle.ndim
            half_angle = angle / 2

        # Scalar part: cos(θ/2)
        scalar_data = cos(half_angle)

        # Bivector part: -sin(θ/2) B
        # Need to broadcast sin(half_angle) with B.data
        if angle.ndim == 0:
            bivector_data = -sin(half_angle) * B.data
        else:
            # Expand sin(half_angle) to broadcast with B
            sin_expanded = sin(half_angle)
            for _ in range(B.grade):
                sin_expanded = sin_expanded[..., newaxis]
            bivector_data = -sin_expanded * B.data

        components = {
            0: scalar_blade(scalar_data, dim=dim, cdim=cdim),
            2: Blade(data=bivector_data, grade=2, dim=dim, cdim=cdim),
        }

        return cls(components=components, dim=dim, cdim=cdim)

    @classmethod
    def translator(cls, displacement: NDArray, dim: int = None, cdim: int = 0) -> "Motor":
        """
        Pure translation.

        M = 1 + (1/2) t^m e_{0m}

        The sandwich product T p T† translates point p by displacement t.
        Uses positive sign for the bivector part to achieve correct translation
        direction in point-based PGA.

        Args:
            displacement: Translation vector (shape: (..., d))
            dim: PGA dimension (d+1, inferred if not provided)
            cdim: Number of collection dimensions

        Returns:
            Motor representing pure translation

        Example:
            # Single translation
            M = Motor.translator([1, 0, 0])

            # Collection of translations
            displacements = array([[[1,0,0]], [[0,1,0]], [[0,0,1]]])
            M = Motor.translator(displacements, cdim=1)  # (3, 3)
        """
        displacement = array(displacement)
        d = displacement.shape[-1]
        dim = d + 1 if dim is None else dim

        if displacement.shape[-1] != dim - 1:
            raise ValueError(f"Displacement has {displacement.shape[-1]} components, expected {dim - 1}")

        # Infer cdim from displacement shape if not provided
        if cdim == 0 and displacement.ndim > 1:
            cdim = displacement.ndim - 1

        # Scalar part: 1
        shape_0 = displacement.shape[:-1] if cdim > 0 else ()
        scalar_data = ones(shape_0)

        # Bivector part: (1/2) t^m e_{0m}
        # This is the degenerate part of the bivector
        # Shape: (..., dim, dim) where only [0, m] entries are nonzero
        shape_2 = displacement.shape[:-1] + (dim, dim)
        bivector_data = zeros(shape_2)
        for m in range(1, dim):
            bivector_data[..., 0, m] = 0.5 * displacement[..., m - 1]

        components = {
            0: scalar_blade(scalar_data, dim=dim, cdim=cdim),
            2: Blade(data=bivector_data, grade=2, dim=dim, cdim=cdim),
        }

        return cls(components=components, dim=dim, cdim=cdim)

    @classmethod
    def rotation_about_point(
        cls,
        p: Blade,
        B: Blade,
        angle: float | NDArray,
    ) -> "Motor":
        """
        Rotation about arbitrary center point.

        Implemented as: translate to origin, rotate, translate back.
        This is equivalent to the line exponential M = exp(-(p ^ B) theta / 2).

        Args:
            p: PGA point (grade-1) for rotation center
            B: Euclidean bivector defining rotation plane
            angle: Rotation angle in radians

        Returns:
            Motor representing rotation about center

        Example:
            p = point([1, 0, 0])  # Center at x=1
            B = bivector_blade([[0, 0, 0, 0, 0, 1]], dim=4)  # e_{12}
            M = Motor.rotation_about_point(p, B, pi / 2)
        """
        # Extract center coordinates (project from PGA to Euclidean)
        c = to_euclidean(p)  # Shape: (..., d)

        # Create three motors: T1 (to origin), R (rotate), T2 (back)
        T1 = cls.translator(-c, dim=p.dim)  # Translate to origin
        R = cls.rotor(B, angle)  # Rotate
        T2 = cls.translator(c, dim=p.dim)  # Translate back

        # Compose: T2 * R * T1
        return T2.compose(R).compose(T1)

    @classmethod
    def screw(
        cls,
        B: Blade,
        angle: float | NDArray,
        translation: NDArray,
        center: NDArray | None = None,
    ) -> "Motor":
        """
        Screw motion: rotation in plane B + translation.

        The screw motion rotates by `angle` in the plane defined by bivector B
        and translates by `translation`. This is the general rigid motion.

        In proper GA, the rotation plane is the fundamental object (bivector),
        not a derived axis. The translation direction is explicit.

        Args:
            B: Bivector defining the rotation plane
            angle: Rotation angle in radians
            translation: Translation vector
            center: Optional center point (default: origin)

        Returns:
            Motor representing the screw motion

        Example:
            # Rotation in e12 plane + translation along e3
            B = e1 ^ e2  # or use bivector_blade
            M = Motor.screw(B, angle=pi/2, translation=[0, 0, 1])

            # With explicit center
            M = Motor.screw(B, angle=pi, translation=[0, 0, 2],
                            center=[1, 0, 0])
        """
        translation = array(translation, dtype=float)

        # Create rotor and translator
        R = cls.rotor(B, angle)
        T = cls.translator(translation, dim=B.dim)

        # Compose: T * R (translate after rotate)
        if center is not None:
            center = array(center, dtype=float)
            T_to = cls.translator(-center, dim=B.dim)
            T_back = cls.translator(center, dim=B.dim)
            return T_back.compose(T).compose(R).compose(T_to)

        return T.compose(R)

    @classmethod
    def from_line(cls, L: Blade, param: float | NDArray) -> "Motor":
        """
        General motor from line exponential.

        M = exp(-L s/2)

        Args:
            L: Line (grade-3 in PGA encoding screw or rotation+center)
            param: Motion parameter s

        Returns:
            Motor
        """
        raise NotImplementedError(
            "Motor.from_line() requires logarithm/exponential operations on multivectors. "
            "Use specific constructors (rotor, translator, rotation_about_point) instead."
        )

    @classmethod
    def from_bivector_angle(cls, B: Blade, angle: float | NDArray, p: Blade | None = None) -> "Motor":
        """
        Construct motor from bivector and angle with optional center.

        Args:
            B: Bivector defining plane of rotation
            angle: Rotation angle in radians
            p: Optional PGA point for rotation center

        Returns:
            Motor (rotor if p is None, full motor otherwise)
        """
        if p is None:
            return cls.rotor(B, angle)

        return cls.rotation_about_point(p, B, angle)

    def apply(self, p: Blade, g: Metric | None = None) -> Blade:
        """
        Apply motor to PGA points via sandwich product.

        p' = M p M†

        Args:
            p: PGA point or points (grade-1, shape: (..., dim))
            g: PGA metric (defaults to pga(dim - 1))

        Returns:
            Transformed point(s) with same shape

        Example:
            p = point([[0, 0, 0], [1, 0, 0]])  # 2 points
            M = Motor.translator([1, 0, 0])
            p_new = M.apply(p)  # Both translated
        """
        g = pga(self.dim - 1) if g is None else g

        # Convert point blade to MultiVector for geometric product
        p_mv = MultiVector(components={p.grade: p}, dim=p.dim, cdim=p.cdim)

        # Sandwich product: M p M†
        M_rev = reverse(self)
        temp = geometric(self, p_mv, g)
        result = geometric(temp, M_rev, g)

        # Extract grade-1 component (the transformed point)
        transformed = result.grade_select(1)

        if transformed is None:
            raise ValueError("Motor transformation did not produce a grade-1 result")

        return transformed

    def apply_to_euclidean(self, v: NDArray, g: Metric | None = None) -> NDArray:
        """
        Apply motor to Euclidean vectors (convenience wrapper).

        Handles PGA embedding/projection automatically:
        1. Embed v as PGA points: p = e_0 + v^m e_m
        2. Apply motor: p' = M p M†
        3. Project back: v' = p'^m / p'^0

        Args:
            v: Euclidean vectors (shape: (..., d))
            g: PGA metric

        Returns:
            Transformed Euclidean vectors (shape: (..., d))

        Example:
            vectors = array([[0, 0, 0], [1, 1, 1]])
            M = Motor.rotor(B, pi / 4)
            v_rotated = M.apply_to_euclidean(vectors)
        """
        v = array(v)
        # Calculate cdim: all dimensions except the last are collection dimensions
        cdim = v.ndim - 1 if v.ndim > 1 else 0

        # Embed as PGA points
        p = point(v, cdim=cdim)

        # Apply motor
        p_transformed = self.apply(p, g)

        # Project back to Euclidean
        return to_euclidean(p_transformed)

    def compose(self, other: "Motor", g: Metric | None = None) -> "Motor":
        """
        Compose with another motor via geometric product.

        M_composed = self * other

        Applied right-to-left: (M2 * M1)(p) = M2(M1(p))

        The result is projected onto grades {0, 2} since motors are defined
        as elements with these grades only. Higher grade components (e.g.,
        grade-4 from bivector products in 4D) are discarded.

        Args:
            other: Motor to compose with
            g: PGA metric

        Returns:
            Composed motor

        Example:
            M1 = Motor.translator([1, 0, 0])
            M2 = Motor.rotor(B, pi / 2)
            M = M2.compose(M1)  # Translate then rotate
        """
        g = pga(self.dim - 1) if g is None else g
        result_mv = geometric(self, other, g)

        # Project onto motor grades {0, 2}
        motor_components = {k: v for k, v in result_mv.components.items() if k in {0, 2}}

        return Motor(components=motor_components, dim=self.dim, cdim=max(self.cdim, other.cdim))

    def inverse(self, g: Metric | None = None) -> "Motor":
        """
        Compute motor inverse.

        For motors (versors), M^(-1) = M† typically.

        Args:
            g: PGA metric

        Returns:
            Inverse motor
        """
        g = pga(self.dim - 1) if g is None else g
        M_inv = inverse(self, g)

        return Motor(components=M_inv.components, dim=self.dim, cdim=self.cdim)

    def to_matrix(self, g: Metric | None = None) -> NDArray:
        """
        Convert motor to transformation matrix for efficient bulk operations.

        Returns (dim x dim) matrix M such that: p' = M @ p

        Args:
            g: PGA metric

        Returns:
            Transformation matrix (shape: (dim, dim) or (..., dim, dim))
        """
        g = pga(self.dim - 1) if g is None else g
        dim = self.dim

        # Apply motor to basis vectors and extract matrix
        # Create identity matrix as set of points
        shape = self.collection_shape + (dim, dim)
        matrix = zeros(shape)

        for k in range(dim):
            # Create k-th basis point
            basis_data = zeros(self.collection_shape + (dim,))
            basis_data[..., k] = 1.0
            basis_point = Blade(data=basis_data, grade=1, dim=dim, cdim=self.cdim)

            # Transform it
            transformed = self.apply(basis_point, g)

            # Extract as column
            matrix[..., :, k] = transformed.data

        return matrix

    def __mul__(self, other):
        """
        Override multiplication for motor composition or point transformation.

        M1 * M2 -> composed motor
        M * p -> transformed point
        """
        if isinstance(other, Motor):
            return self.compose(other)
        elif isinstance(other, Blade):
            return self.apply(other)
        else:
            # Fallback to MultiVector multiplication
            return super().__mul__(other)

    def __call__(self, target: Blade | NDArray) -> Blade | NDArray:
        """
        Make motor callable for transformation.

        Args:
            target: PGA point (Blade) or Euclidean vector (NDArray)

        Returns:
            Transformed target (same type as input)

        Example:
            M = Motor.translator([1, 0, 0])
            p = point([0, 0, 0])
            p_new = M(p)  # Callable syntax
        """
        if isinstance(target, Blade):
            return self.apply(target)

        if isinstance(target, ndarray):
            return self.apply_to_euclidean(target)

        raise TypeError(f"Cannot apply motor to {type(target)}")

    def __invert__(self) -> "Motor":
        """
        Reverse operator: ~M

        For motors, the reverse is equivalent to the inverse (for normalized motors).
        Returns M† where M M† = 1.
        """
        M_rev = reverse(self)
        return Motor(components=M_rev.components, dim=self.dim, cdim=self.cdim)

    def __pow__(self, exponent: int) -> "Motor":
        """
        Power operation for motors.

        For motors (which are versors), the inverse is computed via
        the geometric product. Motors should typically be normalized,
        in which case inverse equals reverse.

        Currently supports:
            motor**(-1) - multiplicative inverse
            motor**(1)  - identity (returns self)

        Args:
            exponent: Integer power (only -1 and 1 supported)

        Returns:
            Motor: Result of power operation

        Raises:
            NotImplementedError: For unsupported exponents
            ValueError: If motor is not invertible
        """
        if exponent == -1:
            return self.inverse()
        elif exponent == 1:
            return self
        else:
            raise NotImplementedError(
                f"Power {exponent} not implemented. Only motor**(-1) for multiplicative inverse is supported."
            )

    def __repr__(self) -> str:
        return f"Motor(dim={self.dim}, cdim={self.cdim}, grades={self.grades})"
