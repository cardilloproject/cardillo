import numpy as np
import numpy.typing as npt
from scipy.spatial import ConvexHull

from cardillo.math import norm, cross3
from cardillo.discrete import RigidBodyQuaternion


class ConvexRigidBody(RigidBodyQuaternion):
    def __init__(
        self,
        points: npt.ArrayLike,
        rho: float = None,
        mass: float = None,
        q0: np.ndarray = None,
        u0: np.ndarray = None,
    ):
        """body of type Rigid_body_quaternion with arbitrary convex surface described by a pointcloud

        Args:
            points (np.ndarray):        pointcloud which is used by a convex hull algorithm to determine the shape of the body. The algorithm ensures that all points lie on or within the convex hull
            rho (float, optional):      density of the body in kg/m^3 (see https://numpy-stl.readthedocs.io/en/latest/_modules/stl/base.html#BaseMesh.get_mass_properties_with_density). Set either mass or rho. Defaults to None.
            mass (float, optional):     total mass of the body. Set either mass or rho. Defaults to None.
            q0 (np.ndarray, optional):  Initial position and rotation. Defaults to None.
            u0 (np.ndarray, optional):  Initial velocity and angular velocity. Defaults to None.

        Raises:
            TypeError: Either one of density and mass has to be of type None
            TypeError: mass and density cannot both hav type None
        """
        self.mesh = Mesh(points)
        mass, K_theta_S, self.A_KK0 = self.mesh.mass_properties(rho, mass)
        self.volume = self.mesh.convex_hull.volume

        # halfplane equations defining polytope
        self.A, self.b = self.mesh.get_halfplane_equations()

        super().__init__(mass, K_theta_S, q0=q0, u0=u0)

    def A_IK(self, t, q, frame_ID=None):
        return super().A_IK(t, q, frame_ID) @ self.A_KK0.T

    def A_IK_q(self, t, q, frame_ID=None):
        A_IK_q = super().A_IK_q(t, q, frame_ID)
        return np.einsum("ijk,jl->ilk", A_IK_q, self.A_KK0.T)

    # transform into new K system
    def K_Omega(self, t, q, u, frame_ID=None):
        return self.A_KK0 @ super().K_Omega(t, q, u, frame_ID)

    def K_Omega_q(self, t, q, u, frame_ID=None):
        return self.A_KK0 @ super().K_Omega_q(t, q, u, frame_ID)

    def K_Psi(self, t, q, u, u_dot, frame_ID=None):
        return self.A_KK0 @ super().K_Psi(t, q, u, u_dot, frame_ID)

    def K_J_R(self, t, q, frame_ID=None):
        return self.A_KK0 @ super().K_J_R(t, q, frame_ID)

    def K_J_R_q(self, t, q, frame_ID=None):
        return self.A_KK0 @ super().K_J_R_q(t, q, frame_ID)

    def export(self, sol_i, base_export=False, **kwargs):
        if base_export:
            points, cells, point_data, cell_data = super().export(sol_i)
        else:
            points, vel, acc = [], [], []
            for point in self.mesh.points:
                points.append(self.r_OP(sol_i.t, sol_i.q[self.qDOF], K_r_SP=point))
                vel.append(
                    self.v_P(
                        sol_i.t, sol_i.q[self.qDOF], sol_i.u[self.uDOF], K_r_SP=point
                    )
                )
                if sol_i.u_dot is not None:
                    acc.append(
                        self.a_P(
                            sol_i.t,
                            sol_i.q[self.qDOF],
                            sol_i.u[self.uDOF],
                            sol_i.u_dot[self.uDOF],
                            K_r_SP=point,
                        )
                    )
            cells = [("triangle", self.mesh.simplices)]

            normals = np.array(
                [
                    self.A_IK(sol_i.t, sol_i.q[self.qDOF]) @ self.A[j, :]
                    for j in range(self.A.shape[0])
                ]
            )

            cell_data = dict(normals=[normals])

            if sol_i.u_dot is not None:
                point_data = dict(v=vel, a=acc)
            else:
                point_data = dict(v=vel)
        return points, cells, point_data, cell_data


class Mesh:
    def __init__(self, points: npt.ArrayLike) -> None:
        """wrapper class for mesh created by convex hull

        Args:
            points (np.ndarray): point cloud which is used as basis for convex hull
        """
        self.convex_hull = ConvexHull(points)
        self.__counterclockwise_ordering()
        self.points = self.convex_hull.points
        self.vertices = self.convex_hull.vertices
        self.simplices = self.convex_hull.simplices
        self.normals = self.convex_hull.equations[:, :-1]
        self.offsets = self.convex_hull.equations[:, -1]
        self.neighbors = self.convex_hull.neighbors

    def __counterclockwise_ordering(self):
        """convex hull creates outward pointing unit normals as part of the hyperplane equations but does not ensure, that the points in a simplex are ordered counterclockwise. As this is important for contact detection, this is done manually here."""
        for simplex, equation in zip(
            self.convex_hull.simplices, self.convex_hull.equations
        ):
            # check if normal calculated using simplex points points in same direction as hyperplane equation normal
            t1 = (
                self.convex_hull.points[simplex[1]]
                - self.convex_hull.points[simplex[0]]
            )
            t2 = (
                self.convex_hull.points[simplex[2]]
                - self.convex_hull.points[simplex[0]]
            )
            a = cross3(t1, t2)
            n = a / norm(a)
            # if n and hyperplane normal point in opposite direction sum will be 0:
            if np.isclose(norm(n + equation[:-1]), 0):
                simplex[1], simplex[2] = simplex[2], simplex[1]

    def mass_properties(self, rho=None, mass=None):
        """This function calculates the volume of the body described by the convex hull, its center of mass based on either rho or mass parameter and the body's inertia matrix w.r.t. to the center of mass. A new frame is introduced in the center of mass whose axes align with the principal axes of the inertia matrix.

        Returns:
            calculated mass, in case rho was specified
            diagonalized inertia matrix
            rotation matrix to new frame

        Args:
            rho (_type_, optional): _description_. Defaults to None.
            mass (_type_, optional): _description_. Defaults to None.
        """

        def __f1(w):
            return w[0] + w[1] + w[2]

        def __f2(w):
            return w[0] ** 2 + w[0] * w[1] + w[1] ** 2 + w[2] * __f1(w)

        def __f3(w):
            return (
                w[0] ** 3
                + w[0] ** 2 * w[1]
                + w[0] * w[1] ** 2
                + w[1] ** 3
                + w[2] * __f2(w)
            )

        def __gi(w, i):
            return __f2(w) + w[i] * __f1(w) + w[i] ** 2

        # [1, x, y, z, x^2, y^2, z^2, xy, yz, xz]
        #  0  1  2  3   4    5    6    7   8   9
        integrals = np.zeros(10)
        for i, simplex in enumerate(self.simplices):
            x, y, z = self.points[simplex].T
            t1 = self.points[simplex[1]] - self.points[simplex[0]]
            t2 = self.points[simplex[2]] - self.points[simplex[0]]
            n_ = self.normals[i] * norm(cross3(t1, t2))
            integrals += np.array(
                [
                    n_[0] * __f1(x),
                    n_[0] * __f2(x),
                    n_[1] * __f2(y),
                    n_[2] * __f2(z),
                    n_[0] * __f3(x),
                    n_[1] * __f3(y),
                    n_[2] * __f3(z),
                    n_[0] * (y[0] * __gi(x, 0) + y[1] * __gi(x, 1) + y[2] * __gi(x, 2)),
                    n_[1] * (z[0] * __gi(y, 0) + z[1] * __gi(y, 1) + z[2] * __gi(y, 2)),
                    n_[2] * (x[0] * __gi(z, 0) + x[1] * __gi(z, 1) + x[2] * __gi(z, 2)),
                ]
            )
        integrals /= np.array([6, 24, 24, 24, 60, 60, 60, 120, 120, 120])

        volume = integrals[0]

        if rho is not None and mass is not None:
            raise TypeError("Either one of density and mass has to be of type None")
        elif mass is not None:
            rho = mass / volume
        elif rho is not None:
            mass = rho * volume
        else:
            raise TypeError("mass and density cannot both have type None")

        center_of_mass = integrals[1:4] / volume
        xs2, ys2, zs2 = center_of_mass**2
        inertia = rho * np.array(
            [
                [integrals[5] + integrals[6], integrals[7], integrals[9]],
                [integrals[7], integrals[4] + integrals[6], -integrals[8]],
                [integrals[9], -integrals[8], integrals[4] + integrals[5]],
            ]
        ) - mass * np.array(
            [
                [
                    ys2 + zs2,
                    center_of_mass[0] * center_of_mass[1],
                    center_of_mass[0] * center_of_mass[2],
                ],
                [
                    center_of_mass[0] * center_of_mass[1],
                    xs2 + zs2,
                    -center_of_mass[1] * center_of_mass[2],
                ],
                [
                    center_of_mass[0] * center_of_mass[2],
                    -center_of_mass[1] * center_of_mass[2],
                    xs2 + ys2,
                ],
            ]
        )

        self.translate(-center_of_mass)
        inertia, A_KK0 = self.__diagonalize_inertia_matrix(inertia)

        return mass, inertia, A_KK0

    def get_halfplane_equations(self):
        return self.normals, self.offsets

    def translate(self, r_OS):
        # self.convex_hull.points += r_OS
        self.points += r_OS
        # new offsets:
        self.offsets = np.array(
            [
                np.dot(self.points[simplex[0]], normal)
                for simplex, normal in zip(self.simplices, self.normals)
            ]
        )

    def __diagonalize_inertia_matrix(self, inertia_matrix):
        ew, ev = np.linalg.eigh(inertia_matrix)
        inertia_matrix_new = np.diag(ew)
        # rotation matrix in new K system:
        A_KK0 = ev.T
        # rotate convex hull points
        self.points = (A_KK0 @ self.points.T).T
        # adapt hyperplane equations:
        self.normals = (A_KK0 @ self.normals.T).T
        return inertia_matrix_new, A_KK0

    def v0(self):
        return self.points[self.simplices[:, 0]]

    def v1(self):
        return self.points[self.simplices[:, 1]]

    def v2(self):
        return self.points[self.simplices[:, 2]]
