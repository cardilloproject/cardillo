import numpy as np

zeros3 = np.zeros(3, dtype=float)
eye3 = np.eye(3, dtype=float)


class Shape:
    """Geometric primitive attached to a subsystem.

    The shape frame S is located at r_OS = r_OP(t, q, xi, B_r_CS) of the
    subsystem and oriented by A_IS = A_IB(t, q, xi) @ A_BS.

    Parameters
    ----------
    subsystem : object
        Subsystem carrying the shape, e.g., RigidBody, PointMass or Frame.
    B_r_CS : np.ndarray (3,)
        Position of the shape frame relative to the reference point of the
        subsystem in body-fixed coordinates.
    A_BS : np.ndarray (3, 3)
        Orientation of the shape frame relative to the body-fixed basis.
    xi : float or None
        Parameter of the attachment point for parametrized subsystems
        (not yet supported by CollisionWorld).
    name : str
    """

    kind = None

    def __init__(self, subsystem, B_r_CS=zeros3, A_BS=eye3, xi=None, name="shape"):
        self.subsystem = subsystem
        self.B_r_CS = np.asarray(B_r_CS, dtype=float)
        self.A_BS = np.asarray(A_BS, dtype=float)
        self.xi = xi
        self.name = name


class SphereShape(Shape):
    """Sphere with given radius around the origin of the shape frame. A radius
    of zero gives a point."""

    kind = "sphere"

    def __init__(self, subsystem, radius, **kwargs):
        super().__init__(subsystem, **kwargs)
        assert radius >= 0, "radius must be non-negative"
        self.radius = float(radius)


class PlaneShape(Shape):
    """Plane through the origin of the shape frame. Its normal is the e_z-axis
    of the shape frame, the half-space below it is solid. The e_x- and
    e_y-axes are the friction directions."""

    kind = "plane"


class CapsuleShape(Shape):
    """Capsule: all points within the distance ``radius`` of a line segment.

    The segment is centered at the origin of the shape frame and points along
    the e_z-axis of the shape frame, i.e., its end points are
    ``r_OS +- half_length * A_IS[:, 2]``. Use `SphereShape` for
    ``half_length = 0``.
    """

    kind = "capsule"

    def __init__(self, subsystem, radius, half_length, **kwargs):
        super().__init__(subsystem, **kwargs)
        assert radius >= 0, "radius must be non-negative"
        assert half_length > 0, "half_length must be positive, use SphereShape"
        self.radius = float(radius)
        self.half_length = float(half_length)
