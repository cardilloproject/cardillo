import numpy as np
from cardillo.visualization import vtk_sphere
import meshio


def Rectangle(Base):
    class _Rectangle(Base):
        def __init__(self, **kwargs):
            """Create a rectangle, defined by its normal vector and dimensions.
            The normal vector is aligned with one of the rectangles three body-fixed axes. Orientation of the body-fixed frameis dependent of chosen Base.
            Args:
                axis:       one of (0, 1, 2)
                dimensions: 2d tuple defining length and width
                **kwargs:   dependent on Base
            """
            axis = kwargs.pop("axis", 2)
            assert axis in (0, 1, 2)

            dimensions = kwargs.pop("dimensions", (1, 1))
            assert len(dimensions) == 2

            self.rolled_axis = np.roll([0, 1, 2], -axis)
            self.dimensions = dimensions

            super().__init__(**kwargs)

        def export(self, sol_i, base_export=False, **kwargs):
            if base_export:
                return super().export(sol_i, **kwargs)
            else:
                r_OP = self.r_OP(sol_i.t)
                A_IK = self.A_IK(sol_i.t)
                n, t1, t2 = A_IK[:, self.rolled_axis].T
                points = [
                    r_OP + t1 * self.dimensions[0] + t2 * self.dimensions[1],
                    r_OP + t1 * self.dimensions[0] - t2 * self.dimensions[1],
                    r_OP - t1 * self.dimensions[0] - t2 * self.dimensions[1],
                    r_OP - t1 * self.dimensions[0] + t2 * self.dimensions[1],
                ]
                cells = [("quad", [np.arange(4)])]
                t1t2 = np.vstack([t1, t2]).T
                point_data = dict(n=[n, n, n, n], t=[t1t2, t1t2, t1t2, t1t2])
            return points, cells, point_data, None

    return _Rectangle


def Ball(Base):
    class _Ball(Base):
        def __init__(self, **kwargs):
            self.radius = kwargs.pop("radius")
            mass = kwargs.pop("mass", None)
            density = kwargs.pop("density", None)

            if density is not None:
                mass = density * (4 / 3) * np.pi * self.radius**3
            elif mass is None:
                raise TypeError("mass and density cannot both have type None")

            K_Theta_S = 2 / 5 * mass * self.radius**2 * np.eye(3)

            kwargs.update({"mass": mass, "K_Theta_S": K_Theta_S})

            super().__init__(**kwargs)

        def export(self, sol_i, base_export=False, **kwargs):
            if base_export:
                return super().export(sol_i, **kwargs)
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


def Cuboid(Base):
    class _Box(Base):
        def __init__(self, **kwargs):
            """Generate a box shaped object with Base chosen on generation. Inertia K_Theta_S if needed by Base is computed.

            Args:
                dimensions:     length, width and heigth of Cuboid
                mass:           mass of Cuboid. Either mass or density must be set
                density:        density of Cuboid. Either mass or density must be set
                kwargs:         further arguments needed by Base
            """
            self.dimensions = kwargs.pop("dimensions")
            mass = kwargs.pop("mass", None)
            density = kwargs.pop("density", None)

            assert (
                len(self.dimensions) == 3
            ), "3 dimensions are needed to generate a box"
            self.a, self.b, self.c = self.dimensions

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
            elif mass is None:
                raise TypeError("mass and density cannot both have type None")

            K_Theta_S = (mass / 12) * np.array(
                [
                    [self.b**2 + self.c**2, 0, 0],
                    [0, self.a**2 + self.c**2, 0],
                    [0, 0, self.a**2 + self.b**2],
                ]
            )

            kwargs.update({"mass": mass, "K_Theta_S": K_Theta_S})

            super().__init__(**kwargs)

        def export(self, sol_i, base_export=False, **kwargs):
            if base_export:
                return super().export(sol_i, **kwargs)
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


def Cylinder(Base):
    class _Cylinder(Base):
        def __init__(self, **kwargs):
            self.length = kwargs.pop("length")
            self.radius = kwargs.pop("radius")
            mass = kwargs.pop("mass", None)
            density = kwargs.pop("density", None)
            self.axis = kwargs.pop("axis")
            assert self.axis in [0, 1, 2]

            volume = self.length * np.pi * self.radius**2
            if density is not None:
                mass = density * volume
            elif mass is None:
                raise TypeError("mass and density cannot both have type None")

            diag = np.array(
                [
                    6 * self.radius**2,
                    3 * self.radius**2 + self.length**2,
                    3 * self.radius**2 + self.length**2,
                ],
                dtype=float,
            )
            K_Theta_S = np.diag(np.roll(diag, shift=self.axis))

            kwargs.update({"mass": mass, "K_Theta_S": K_Theta_S})

            super().__init__(**kwargs)

        def export(self, sol_i, base_export=False, **kwargs):
            if base_export:
                return super().export(sol_i, **kwargs)
            else:
                ##################################
                # compute position and orientation
                ##################################
                r_OS = self.r_OP(sol_i.t, sol_i.q[self.qDOF])
                A_IK0 = self.A_IK(sol_i.t, sol_i.q[self.qDOF])
                d1, d2, d3 = np.roll(A_IK0, shift=self.axis, axis=0).T
                r_OP0 = r_OS - 0.5 * d1 * self.length
                r_OP1 = r_OS + 0.5 * d1 * self.length

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


def FromSTL(Base):
    class _FromSTL(Base):
        def __init__(self, scale=1.0, **kwargs):
            self.path = kwargs.pop("path")
            self.K_r_SP = kwargs.pop("K_r_SP")

            self.meshio_mesh = meshio.read(self.path)
            self.meshio_mesh.points *= scale

            super().__init__(**kwargs)

        def export(self, sol_i, base_export=False, **kwargs):
            if base_export:
                return super().export(sol_i, **kwargs)
            else:
                points, vel, acc = [], [], []
                for K_r_PQ in self.meshio_mesh.points:
                    # center of mass (S) over stl origin (P) to arbitrary stl point (Q)
                    K_r_SQ = self.K_r_SP + K_r_PQ
                    points.append(self.r_OP(sol_i.t, sol_i.q[self.qDOF], K_r_SP=K_r_SQ))

                    vel.append(
                        self.v_P(
                            sol_i.t,
                            sol_i.q[self.qDOF],
                            sol_i.u[self.uDOF],
                            K_r_SP=K_r_SQ,
                        )
                    )

                    if sol_i.u_dot is not None:
                        acc.append(
                            self.a_P(
                                sol_i.t,
                                sol_i.q[self.qDOF],
                                sol_i.u[self.uDOF],
                                sol_i.u_dot[self.uDOF],
                                K_r_SP=K_r_SQ,
                            )
                        )

                if sol_i.u_dot is not None:
                    point_data = dict(v=vel, a=acc)
                else:
                    point_data = dict(v=vel)

                cells = self.meshio_mesh.cells

            return points, cells, point_data, None

    return _FromSTL
