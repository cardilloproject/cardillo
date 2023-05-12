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
        def __init__(self, length, radius, density, axis=1, q0=None, u0=None):
            self.length = length
            self.radius = radius
            self.density = density
            assert axis in [0, 1, 2]
            self.axis = axis

            volume = length * np.pi * radius**2
            mass = volume * density
            diag = np.array(
                [
                    6 * radius**2,
                    3 * radius**2 + length**2,
                    3 * radius**2 + length**2,
                ],
                dtype=float,
            )
            K_theta_S = np.diag(np.roll(diag, shift=axis))

            super().__init__(mass, K_theta_S, q0, u0)

        def export(self, sol_i, base_export=False, **kwargs):
            if base_export:
                points, cells, point_data, cell_data = super().export(sol_i)
                return points, cells, point_data, cell_data
            else:
                ##################################
                # compute position and orientation
                ##################################
                r_OP0 = self.r_OP(sol_i.t, sol_i.q[self.qDOF])
                A_IK0 = self.A_IK(sol_i.t, sol_i.q[self.qDOF])
                d1, d2, d3 = np.roll(A_IK0, shift=self.axis, axis=0).T
                r_OP1 = r_OP0 + d1 * self.length

                ##################################
                # build Bezier volume
                ##################################
                outer_radius = 2 * self.radius
                a = 2 * np.sqrt(3) * self.radius

                # minimal number of points that define the wedge
                # layer 0
                P00 = r_OP0 - self.radius * d3
                P30 = r_OP0 + d2 * a / 2 - self.radius * d3
                P40 = r_OP0 + d3 * outer_radius
                # layer 1
                P01 = r_OP1 - self.radius * d3
                P31 = r_OP1 + d2 * a / 2 - self.radius * d3
                P41 = r_OP1 + d3 * outer_radius

                def compute_missing_points(P0, P3, P4):
                    P5 = 2 * P0 - P3
                    P1 = 0.5 * (P3 + P4)
                    P0 = 0.5 * (P5 + P3)
                    P2 = 0.5 * (P4 + P5)

                    # add correct weights to the points
                    dim = len(P0)
                    points_weights = np.zeros((6, dim + 1), dtype=float)
                    points_weights[0] = np.array([*P0, 1])
                    points_weights[1] = np.array([*P1, 1])
                    points_weights[2] = np.array([*P2, 1])
                    points_weights[3] = np.array([*P3, 1 / 2])
                    points_weights[4] = np.array([*P4, 1 / 2])
                    points_weights[5] = np.array([*P5, 1 / 2])

                    return points_weights

                # create correct VTK ordering, see
                # https://coreform.com/papers/implementation-of-rational-bezier-cells-into-VTK-report.pdf:
                vtk_points_weights = []

                # compute all missing points of both layers
                points_layer0 = compute_missing_points(P00, P30, P40)
                points_layer1 = compute_missing_points(P01, P31, P41)

                #######################
                # 1. vertices (corners)
                #######################

                # bottom
                for j in range(3):
                    vtk_points_weights.append(points_layer0[j])

                # top
                for j in range(3):
                    vtk_points_weights.append(points_layer1[j])

                ##########
                # 2. edges
                ##########

                # bottom
                for j in range(3, 6):
                    vtk_points_weights.append(points_layer0[j])

                # top
                for j in range(3, 6):
                    vtk_points_weights.append(points_layer1[j])

                # number of points per layer
                n_layer = 6

                # polynomial degree in d1 direction
                p_zeta = 1

                # number of points per cell
                n_cell = (p_zeta + 1) * n_layer

                higher_order_degrees = [(np.array([2, 2, p_zeta]),) for _ in range(1)]

                cells = [
                    (
                        "VTK_BEZIER_WEDGE",
                        np.arange(i * n_cell, (i + 1) * n_cell)[None],
                    )
                    for i in range(1)
                ]

                vtk_points_weights = np.array(vtk_points_weights)
                vtk_points = vtk_points_weights[:, :3]

                point_data = {
                    "RationalWeights": vtk_points_weights[:, 3],
                }

                cell_data = {
                    "HigherOrderDegrees": higher_order_degrees,
                }

                return vtk_points, cells, point_data, cell_data

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
