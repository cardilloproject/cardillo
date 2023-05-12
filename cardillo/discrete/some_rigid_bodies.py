import numpy as np
from cardillo.visualization import vtk_sphere
from stl import mesh
import meshio


def Ball(RigidBodyParametrization):
    class _Ball(RigidBodyParametrization):
        def __init__(self, mass, radius, q0=None, u0=None):
            K_theta_S = 2 / 5 * mass * radius**2 * np.eye(3)
            self.radius = radius
            super().__init__(mass, K_theta_S, q0, u0)

        def export(self, sol_i, resolution=20, base_export=False, **kwargs):
            if base_export:
                points, cells, point_data, cell_data = super().export(sol_i)
            else:
                points_sphere, cell, point_data = vtk_sphere(self.radius)
                cells = [cell]
                points, vel, acc = [], [], []
                for point in points_sphere:
                    points.append(self.r_OP(sol_i.t, sol_i.q[self.qDOF], K_r_SP=point))
                    vel.append(
                        self.v_P(
                            sol_i.t,
                            sol_i.q[self.qDOF],
                            sol_i.u[self.uDOF],
                            K_r_SP=point,
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
                point_data.update({"v": vel})
                if sol_i.u_dot is not None:
                    point_data.update({"a": acc})

            return points, cells, point_data, None

    return _Ball


def Box(RigidBodyParametrization):
    class _Box(RigidBodyParametrization):
        def __init__(
            self, dimensions, mass=None, density=None, K_theta_S=None, q0=None, u0=None
        ):
            assert len(dimensions) == 3, "3 dimensions are needed to generate a box"
            self.a, self.b, self.c = dimensions

            p1 = np.array([0.5 * self.a, -0.5 * self.b, 0.5 * self.c])
            p2 = np.array([0.5 * self.a, 0.5 * self.b, 0.5 * self.c])
            p3 = np.array([0.5 * self.a, 0.5 * self.b, -0.5 * self.c])
            p4 = np.array([0.5 * self.a, -0.5 * self.b, -0.5 * self.c])
            p5 = np.array([-0.5 * self.a, -0.5 * self.b, 0.5 * self.c])
            p6 = np.array([-0.5 * self.a, 0.5 * self.b, 0.5 * self.c])
            p7 = np.array([-0.5 * self.a, 0.5 * self.b, -0.5 * self.c])
            p8 = np.array([-0.5 * self.a, -0.5 * self.b, -0.5 * self.c])
            self.points = np.vstack((p1, p2, p3, p4, p5, p6, p7, p8))

            if density is not None:
                mass = density * self.a * self.b * self.c
            elif mass is not None:
                mass = mass
            else:
                raise TypeError("mass and density cannot both have type None")

            K_theta_S = (
                (mass / 12)
                * np.array(
                    [
                        [self.b**2 + self.c**2, 0, 0],
                        [0, self.a**2 + self.c**2, 0],
                        [0, 0, self.a**2 + self.b**2],
                    ]
                )
                if K_theta_S is None
                else K_theta_S
            )

            super().__init__(mass, K_theta_S, q0, u0)

        def export(self, sol_i, base_export=False, **kwargs):
            if base_export:
                points, cells, point_data, cell_data = super().export(sol_i)
                return points, cells, point_data, cell_data
            else:
                points, vel, acc = [], [], []
                for point in self.points:
                    points.append(self.r_OP(sol_i.t, sol_i.q[self.qDOF], K_r_SP=point))

                    vel.append(
                        self.v_P(
                            sol_i.t,
                            sol_i.q[self.qDOF],
                            sol_i.u[self.uDOF],
                            K_r_SP=point,
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
                cells = [("hexahedron", [[0, 1, 2, 3, 4, 5, 6, 7]])]

                if sol_i.u_dot is not None:
                    point_data = dict(v=vel, a=acc)
                else:
                    point_data = dict(v=vel)
            return points, cells, point_data, None

    return _Box


def Cylinder(RigidBodyParametrization):
    class _Cylinder(RigidBodyParametrization):
        def __init__(self, length, radius, density, q0=None, u0=None):
            volume = length * np.pi * radius**2
            mass = volume * density
            K_theta_S = (mass / 12) * np.diag(
                [
                    3 * radius**2 + length**2,
                    3 * radius**2 + length**2,
                    6 * radius**2,
                ]
            )

            super().__init__(mass, K_theta_S, q0, u0)

        def export(self, sol_i, base_export=False, **kwargs):
            if base_export:
                points, cells, point_data, cell_data = super().export(sol_i)
                return points, cells, point_data, cell_data
            else:
                raise NotImplementedError

    return _Cylinder


def FromSTL(RigidBodyParametrization):
    class _FromSTL(RigidBodyParametrization):
        def __init__(self, path, density, q0=None, u0=None):
            self.path = path
            self.mesh = mesh.Mesh.from_file(path)
            volume, cog, inertia = self.mesh.get_mass_properties()
            mass = density * volume

            # TODO: Compute correct K_theta_S using cog!
            K_theta_S = density * inertia

            self.meshio_mesh = meshio.read(path)

            super().__init__(mass, K_theta_S, q0, u0)

        def export(self, sol_i, base_export=False, **kwargs):
            if base_export:
                points, cells, point_data, cell_data = super().export(sol_i)
                return points, cells, point_data, cell_data
            else:
                points, vel, acc = [], [], []
                for i, point in enumerate(self.meshio_mesh.points):
                    points.append(self.r_OP(sol_i.t, sol_i.q[self.qDOF], K_r_SP=point))

                    vel.append(
                        self.v_P(
                            sol_i.t,
                            sol_i.q[self.qDOF],
                            sol_i.u[self.uDOF],
                            K_r_SP=point,
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

                if sol_i.u_dot is not None:
                    point_data = dict(v=vel, a=acc)
                else:
                    point_data = dict(v=vel)

                cells = self.meshio_mesh.cells

            return points, cells, point_data, None

    return _FromSTL
