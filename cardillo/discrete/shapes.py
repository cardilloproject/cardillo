import numpy as np
from cardillo.visualization import vtk_sphere
import meshio
import trimesh

def Meshed(Base):
    class _Meshed(Base):
        def __init__(
            self, trimesh_obj, K_r_SP, A_KM=np.eye(3), **kwargs
        ):
            """Generate an object (typically with Base `Frame` or `RigidBody`) from a given Trimesh object
            Args:
                trimesh_obj: instance of trimesh defining the mesh
                K_r_SP (np.ndarray): offset center of mass (S) from STL origin (P) in body fixed K-frame
                A_KM (np.ndarray): tansformation from mesh-fixed frame (M) to body-fixed frame (K)
            """
            self.K_r_SP = K_r_SP
            self.A_KM = A_KM

            assert isinstance(trimesh_obj, trimesh.Trimesh)
            if hasattr(trimesh_obj, "to_mesh"):
                # primitives are converted to mesh
                self.self.visual_mesh = trimesh_obj.to_mesh()
            else:
                self.self.visual_mesh = trimesh_obj

            # vectors from mesh origin to vertices represented in mesh-fixed frame
            M_r_PQ_i = self.visual_mesh.vertices.view(np.ndarray) 

            # vectors (transposed) from center of mass (S) of body to vertices represented in body-fixed frame
            self.K_r_SQi_T = self.K_r_SP[:, None] + self.A_KM @ M_r_PQ_i.T
            super().__init__(**kwargs)

        def export(self, sol_i, base_export=False, **kwargs):
            if base_export:
                return super().export(sol_i, **kwargs)
            else:
                r_OS = self.r_OP(sol_i.t)
                A_IK = self.A_IK(sol_i.t)
                points = (r_OS[:, None] + A_IK @ self.K_r_SQi_T).T

                cells = [
                    ("triangle", self.visual_mesh.faces),
                ]

            return points, cells, None, None

    return _Meshed

def RectangleTrimesh(Base):
    class _Rectangle(Base):
        def __init__(self, **kwargs):
            axis = kwargs.pop("axis", 2)
            assert axis in (0, 1, 2)

            dimensions = kwargs.pop("dimensions", (1, 1))
            # dimensions = kwargs.pop("dimensions", (2, 0.5))
            assert len(dimensions) == 2

            # 0: [1, 2, 0]
            # 1: [2, 0, 1]
            # 2: [0, 1, 2]
            self.rolled_axis = np.roll([1, 2, 0], -axis)
            self.dimensions = dimensions

            A_IK = kwargs.pop("A_IK", np.eye(3))[:, self.rolled_axis]
            transform = np.eye(4)
            transform[:3, :3] = A_IK
            self.visual_primitive = trimesh.primitives.Box(
                extents=[*dimensions, 0], transform=transform
            )
            self.visual_mesh = self.visual_primitive.to_mesh()

            # # debug
            # axis = trimesh.creation.axis()
            # axis.apply_transform(transform)
            # scene = trimesh.Scene()
            # scene.add_geometry(self.trimesh)
            # scene.add_geometry(axis)
            # scene.show()

            kwargs.update({"A_IK": A_IK})
            super().__init__(**kwargs)

        def export(self, sol_i, base_export=False, **kwargs):
            if base_export:
                return super().export(sol_i, **kwargs)
            else:
                r_OS = self.r_OP(sol_i.t)
                A_IK = self.A_IK(sol_i.t)
                t1, t2, n = A_IK.T

                K_r_SPs = self.visual_mesh.vertices.view(np.ndarray)
                points = (r_OS[:, None] + A_IK @ K_r_SPs.T).T

                points = np.concatenate((points, [r_OS, r_OS + t1]))
                faces = self.visual_mesh.faces

                # # export lines for coordinate system
                # lines = np.array([
                #     r_OS,
                #     r_OS + t1,
                #     r_OS + t2,
                #     r_OS + n,
                # ])
                # offset = len(points)
                # line_cells = [
                #     [offset, offset + 1],
                #     [offset, offset + 2],
                #     [offset, offset + 3],
                # ]
                # points = np.concatenate((points, lines))

                cells = [
                    ("triangle", faces),
                    # ("line", line_cells),
                ]

                # fmt: off
                point_data = dict(
                    t1=len(points) * [t1,],
                    t2=len(points) * [t2,],
                    n=len(points) * [n,],
                )

                cell_data = dict(
                    t1=[len(faces) * [t1,]],
                    t2=[len(faces) * [t2,]],
                    n=[len(faces) * [n,]],
                )
                # fmt: on

            return points, cells, point_data, cell_data

    return _Rectangle


def Rectangle(Base):
    class _Rectangle(Base):
        def __init__(self, **kwargs):
            """Create a rectangle, defined by its normal vector and dimensions.
            The normal vector is aligned with one of the rectangles three
            body-fixed axes. Orientation of the body-fixed frame is dependent
            of chosen Base.

            Args:
            -----
                axis:       one of (0, 1, 2)
                dimensions: 2d tuple defining length and width
                **kwargs:   dependent on Base
            """
            axis = kwargs.pop("axis", 2)
            assert axis in (0, 1, 2)

            dimensions = kwargs.pop("dimensions", (1, 1))
            assert len(dimensions) == 2

            # 0: [1, 2, 0]
            # 1: [2, 0, 1]
            # 2: [0, 1, 2]
            self.rolled_axis = np.roll([1, 2, 0], -axis)
            self.dimensions = dimensions

            kwargs.update({"A_IK": kwargs.pop("A_IK", np.eye(3))[:, self.rolled_axis]})
            # print(f"axis: {axis}; rolled_axis: {self.rolled_axis}; A_IK0:\n{kwargs['A_IK']}")
            super().__init__(**kwargs)

        def export(self, sol_i, base_export=False, **kwargs):
            if base_export:
                return super().export(sol_i, **kwargs)
            else:
                r_OP = self.r_OP(sol_i.t)
                A_IK = self.A_IK(sol_i.t)[:, self.rolled_axis]
                t1, t2, n = A_IK.T
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
            """Create a ball shaped rigid body, defined by a radius and either mass or density.
            Inertia matrix in body fixed K-frame is computed based on chosen radius.

            Args:
                radius:     radius of ball
                mass:       mass of ball. Either mass or density need to be set
                density:    density of ball. Either mass or density need to be set
                **kwargs:   dependent on Base. Note: K_Theta_S is provided by Ball
            """
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
                points, vel = [], []
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
                point_data.update({"v": vel})

                return points, cells, point_data, None

    return _Ball


def Cuboid(Base):
    class _Cuboid(Base):
        def __init__(self, **kwargs):
            """Generate a box shaped object with Base chosen on generation. Inertia K_Theta_S if needed by Base is computed.

            Args:
                dimensions:     length, width and heigth of Cuboid
                mass:           mass of Cuboid. Either mass or density must be set
                density:        density of Cuboid. Either mass or density must be set
                **kwargs:         further arguments needed by Base
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
                points, vel = [], []
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

                cells = [("hexahedron", [[0, 1, 2, 3, 4, 5, 6, 7]])]
                point_data = dict(v=vel)
            return points, cells, point_data, None

    return _Cuboid


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
            if mass is None:
                if density is None:
                    raise TypeError("mass and density cannot both have type None")
                else:
                    mass = density * volume

            K_Theta_S = kwargs.pop("K_Theta_S", None)
            if K_Theta_S is None:
                diag = mass * np.array(
                    [
                        (1 / 2) * self.radius**2,
                        (1 / 12) * (3 * self.radius**2 + self.length**2),
                        (1 / 12) * (3 * self.radius**2 + self.length**2),
                    ],
                    dtype=float,
                )
                # raise RuntimeError("Check if this np.roll behaves as expected!")
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
                A_IK = self.A_IK(sol_i.t, sol_i.q[self.qDOF])

                # raise RuntimeError("Check if this np.roll behaves as expected!")
                rolled_axes = np.roll([0, 1, 2], -self.axis)
                d1, d2, d3 = A_IK[:, rolled_axes].T
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


def Tetrahedron(Base):
    class _Tetrahedron(Base):
        def __init__(self, **kwargs):
            """Generate a tetrahedron shaped object with Base chosen on generation. Inertia K_Theta_S if needed by Base is computed.

            Args:
                edge:       length of each edge
                mass:       mass of Tetrahedron; either mass or density must be set
                density:    density of Cuboid; either mass or density must be set
                **kwargs:   further arguments needed by Base
            """
            self.edge = kwargs.pop("edge")
            mass = kwargs.pop("mass", None)
            density = kwargs.pop("density", None)

            # see https://de.wikipedia.org/wiki/Tetraeder
            h_D = self.edge * np.sqrt(3) / 2
            h_P = self.edge * np.sqrt(2 / 3)
            r_OS = np.array([0, h_D / 3, h_P / 4])
            p1 = np.array([-self.edge / 2, 0, 0]) - r_OS
            p2 = np.array([+self.edge / 2, 0, 0]) - r_OS
            p3 = np.array([0, h_D, 0]) - r_OS
            p4 = np.array([0, h_D / 3, h_P]) - r_OS
            self.points = np.vstack((p1, p2, p3, p4))

            # see https://de.wikipedia.org/wiki/Liste_von_Tr%C3%A4gheitstensoren#Platonische_K%C3%B6rper
            if density is not None:
                mass = density * self.edge**3 * np.sqrt(2) / 12
            elif mass is None:
                raise TypeError("mass and density cannot both have type None")

            K_Theta_S = self.edge**2 * (mass / 20) * np.eye(3)

            kwargs.update({"mass": mass, "K_Theta_S": K_Theta_S})

            super().__init__(**kwargs)

        def export(self, sol_i, base_export=False, **kwargs):
            if base_export:
                return super().export(sol_i, **kwargs)
            else:
                points, vel = [], []
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

                cells = [("tetra", [[0, 1, 2, 3]])]
                point_data = dict(v=vel)
            return points, cells, point_data, None

    return _Tetrahedron


