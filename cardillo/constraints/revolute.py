import numpy as np
from cardillo.constraints._base import PositionOrientationBase


class Revolute(PositionOrientationBase):
    r"""Constraint representation of revolute joint.

    Parameters
    ----------
    subsystem1 : object
        RigidBody or CosseratRod
    subsystem2 : object
        RigidBody or CosseratRod
    axis : int
        Integer number between 0 and 2 to indicate which axis is the free rotation axis.
        0 : e_x^B-direction.
        1 : e_y^B-direction.
        2 : e_z^B-direction.
    angle0 : float
        Value of the joint angle in initial configuration q0.
    r_OJ0 : np.ndarray (3,)
        Initial position vector of joint.
    A_IJ0 : np.ndarray (3, 3)
        Initial orientation of joint basis. Defines axis of rotation together with 'axis'.
    xi1 : #TODO
    xi2 : #TODO
    name : str
        Name of contribution.
    """

    def __init__(
        self,
        subsystem1,
        subsystem2,
        axis,
        angle0=0.0,
        r_OJ0=None,
        A_IJ0=None,
        xi1=None,
        xi2=None,
        name="revolute_joint",
    ):
        self.name = name
        self.axis = axis
        self.angle0 = angle0
        self.plane_axes = np.roll([0, 1, 2], -axis)[1:]
        projection_pairs_rotation = [
            (axis, self.plane_axes[0]),
            (axis, self.plane_axes[1]),
        ]

        # aliases for nicer interface in post-processing
        self.angle = self.l
        self.angle_dot = self.l_dot

        super().__init__(
            subsystem1,
            subsystem2,
            projection_pairs_rotation=projection_pairs_rotation,
            r_OJ0=r_OJ0,
            A_IJ0=A_IJ0,
            xi1=xi1,
            xi2=xi2,
        )

    def assembler_callback(self):
        self.n_full_rotations = 0
        self.previous_quadrant = 1
        super().assembler_callback()

    def _compute_quadrant(self, x, y):
        if x > 0 and y >= 0:
            return 1
        elif x <= 0 and y > 0:
            return 2
        elif x < 0 and y <= 0:
            return 3
        elif x >= 0 and y < 0:
            return 4
        else:
            raise RuntimeError("You should never be here!")

    def l(self, t, q):
        A_IJ1 = self.A_IJ1(t, q)
        A_IJ2 = self.A_IJ2(t, q)

        a, b = self.plane_axes

        e_a1 = A_IJ1[:, a]
        e_b1 = A_IJ1[:, b]
        e_a2 = A_IJ2[:, a]

        # projections
        y = e_a2 @ e_b1
        x = e_a2 @ e_a1

        quadrant = self._compute_quadrant(x, y)

        # check if a full rotation happens
        if self.previous_quadrant == 4 and quadrant == 1:
            self.n_full_rotations += 1
        elif self.previous_quadrant == 1 and quadrant == 4:
            self.n_full_rotations -= 1
        self.previous_quadrant = quadrant

        # compute rotation angle without singularities
        angle = self.angle0 + self.n_full_rotations * 2 * np.pi
        if quadrant == 1:
            angle += np.arctan(y / x)
        elif quadrant == 2:
            angle += 0.5 * np.pi + np.arctan(-x / y)
        elif quadrant == 3:
            angle += np.pi + np.arctan(-y / -x)
        else:
            angle += 1.5 * np.pi + np.arctan(x / -y)

        return angle

    def l_dot(self, t, q, u):
        e_c1 = self.A_IJ1(t, q)[:, self.axis]
        return (self.Omega2(t, q, u) - self.Omega1(t, q, u)) @ e_c1

    def l_dot_q(self, t, q, u):
        e_c1 = self.A_IJ1(t, q)[:, self.axis]
        e_c1_q1 = self.A_IJ1_q1(t, q)[:, self.axis]

        return np.concatenate(
            [
                (self.Omega2(t, q, u) - self.Omega1(t, q, u)) @ e_c1_q1
                - e_c1 @ self.Omega1_q1(t, q, u),
                e_c1 @ self.Omega2_q2(t, q, u),
            ]
        )

    def l_dot_u(self, t, q, u):
        e_c1 = self.A_IJ1(t, q)[:, self.axis]
        return e_c1 @ np.concatenate([-self.J_R1(t, q), self.J_R2(t, q)], axis=1)

    def l_q(self, t, q):
        A_IJ1 = self.A_IJ1(t, q)
        A_IJ2 = self.A_IJ2(t, q)
        A_IJ1_q1 = self.A_IJ1_q1(t, q)
        A_IJ2_q2 = self.A_IJ2_q2(t, q)

        a, b = self.plane_axes

        e_a1 = A_IJ1[:, a]
        e_b1 = A_IJ1[:, b]
        e_a2 = A_IJ2[:, a]

        e_a1_q1 = A_IJ1_q1[:, a]
        e_b1_q1 = A_IJ1_q1[:, b]
        e_a2_q2 = A_IJ2_q2[:, a]

        # projections
        y = e_a2 @ e_b1
        x = e_a2 @ e_a1

        x_q = np.concatenate((e_a2 @ e_a1_q1, e_a1 @ e_a2_q2))
        y_q = np.concatenate((e_a2 @ e_b1_q1, e_b1 @ e_a2_q2))

        return (x * y_q - y * x_q) / (x**2 + y**2)

    def W_l(self, t, q):
        J_R1 = self.J_R1(t, q)
        J_R2 = self.J_R2(t, q)
        e_c1 = self.A_IJ1(t, q)[:, self.axis]
        return np.concatenate([-J_R1.T @ e_c1, J_R2.T @ e_c1]).reshape(self._nu, 1)

    def W_l_q(self, t, q):
        nq1 = self._nq1
        nu1 = self._nu1

        J_R1 = self.J_R1(t, q)
        J_R2 = self.J_R2(t, q)
        J_R1_q1 = self.J_R1_q1(t, q)
        J_R2_q2 = self.J_R2_q2(t, q)

        e_c1 = self.A_IJ1(t, q)[:, self.axis]
        e_c1_q1 = self.A_IJ1_q1(t, q)[:, self.axis]

        # dense blocks
        W_angle_q = np.zeros((self._nu, 1, self._nq))
        W_angle_q[:nu1, 0, :nq1] = (
            np.einsum("i,ijk->jk", -e_c1, J_R1_q1) - J_R1.T @ e_c1_q1
        )
        W_angle_q[nu1:, 0, :nq1] = J_R2.T @ e_c1_q1
        W_angle_q[nu1:, 0, nq1:] = np.einsum("i,ijk->jk", e_c1, J_R2_q2)

        return W_angle_q

    def reset(self):
        self.n_full_rotations = 0
        self.previous_quadrant = 1
