import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import meshio
from cardillo.math.rotations import A_IK_basic
from cardillo.discretization.indexing import flat3D


class BezierMesh3D:
    def __init__(self, p, q, r):
        self.p = p
        self.q = q
        self.r = r
        self.vtk_cell_type = "VTK_BEZIER_HEXAHEDRON"

    def split_vtk(self, point_weights):
        """Rearranges either a Point Array with dimensions
        (p+1)x(q+1)x(r+1)x(dim+1) to vtk ordering, see vtk:

        References:
        -----------

        vtk: https://blog.kitware.com/modeling-arbitrary-order-lagrange-finite-elements-in-the-visualization-toolkit/
        """
        # extract dimensions
        p, q, r, _ = point_weights.shape

        #######################
        # 1. vertices (corners)
        #######################
        # fmt: off
        # points.extend(
        vertices = [
            point_weights[ 0,  0,  0],
            point_weights[-1,  0,  0],
            point_weights[-1, -1,  0],
            point_weights[ 0, -1,  0],
            point_weights[ 0,  0, -1],
            point_weights[-1,  0, -1],
            point_weights[-1, -1, -1],
            point_weights[ 0, -1, -1],
        ]
        # fmt: on

        ##########
        # 2. edges
        ##########
        # fmt: off
        edges = []
        for iz in [0, -1]:
            edges.extend(point_weights[1:-1,    0, iz])
            edges.extend(point_weights[  -1, 1:-1, iz])
            edges.extend(point_weights[1:-1,   -1, iz])
            edges.extend(point_weights[   0, 1:-1, iz])
        for ix in [0, -1]:
            edges.extend(point_weights[ix, 0, 1:-1])
        for ix in [0, -1]:
            edges.extend(point_weights[ix, -1, 1:-1])
        # fmt: on

        ##########
        # 3. faces
        ##########
        # yz
        faces = []
        for ix in [0, -1]:
            for iz in range(r - 2):
                faces.extend(point_weights[ix, 1:-1, iz + 1])
        # xz
        for iy in [0, -1]:
            for iz in range(r - 2):
                faces.extend(point_weights[1:-1, iy, iz + 1])
        # xy
        for iz in [0, -1]:
            for iy in range(q - 2):
                faces.extend(point_weights[1:-1, iy + 1, iz])

        ###########
        # 4. volume
        ###########
        volume = []
        for iz in range(r - 2):
            for iy in range(q - 2):
                for ix in range(p - 2):
                    volume.extend([point_weights[ix + 1, iy + 1, iz + 1]])

        return vertices, edges, faces, volume

    def points_weights_vtk(self, P):
        vertices, edges, faces, volume = self.split_vtk(P)
        return np.array(vertices + edges + faces + volume)

    # TODO: This is only a single Bezier element!
    def export(self, points_weights, file):
        nxi, neta, nzeta, dim1 = points_weights.shape
        nxi -= self.p
        neta -= self.q
        nzeta -= self.r

        n_patches = nxi * neta * nzeta
        patch_size = (self.p + 1) * (self.q + 1) * (self.r + 1)

        vtk_points_weights = np.zeros((n_patches * patch_size, dim1))
        vtk_cells = []
        vtk_HigherOrderDegree = []

        exit()

        for i in range(nxi):
            for j in range(neta):
                for k in range(nzeta):
                    idx = flat3D(i, j, k, (nxi, neta))
                    point_range = np.arange(idx * patch_size, (idx + 1) * patch_size)

                    vtk_points_weights[point_range] = self.points_weights_vtk(
                        points_weights[i, j, k]
                    )

        vtk_points_weights = self.points_weights_vtk(points_weights)
        vtk_points = vtk_points_weights[:, :3]
        vtk_weights = vtk_points_weights[:, 3]

        cells = [(self.vtk_cell_type, np.arange(len(vtk_points))[None])]

        higher_order_degrees = [
            np.array([p, q, r])[None],
        ]

        # return points, cells, None, None
        point_data = {"RationalWeights": vtk_weights}
        # point_data = None
        cell_data = {"HigherOrderDegrees": higher_order_degrees}

        meshio.write_points_cells(
            filename=file,
            points=vtk_points,
            cells=cells,
            point_data=point_data,
            cell_data=cell_data,
            binary=False,
        )


def test_volume():
    p = q = r = 1
    points_weights = np.zeros((2, 2, 3, 4), dtype=float)
    nxi, neta, nzeta, _ = points_weights.shape

    for i in range(nxi):
        points_weights[0, 0, i] = np.array([0, 0, i, 1])
        points_weights[1, 0, i] = np.array([1, 0, i, 1])

        points_weights[0, 1, i] = np.array([0, 1, i, 1])
        points_weights[1, 1, i] = np.array([1, 1, i, 1])

    # p = 2
    # q = r = 1
    # points_weights = np.zeros((p + 1, q + 1, r + 1, 3), dtype=float)

    # # layer 0
    # points_weights[0, 0, 0] = np.array([0, 0, 0])
    # points_weights[1, 0, 0] = np.array([1, 0, 0])
    # points_weights[2, 0, 0] = np.array([2, 0, 0])

    # points_weights[0, 1, 0] = np.array([0, 1, 0])
    # points_weights[1, 1, 0] = np.array([1, 1.5, 0])
    # points_weights[2, 1, 0] = np.array([2, 0.75, 0])

    # # layer 1
    # points_weights[0, 0, 1] = np.array([0, 0, 1])
    # points_weights[1, 0, 1] = np.array([1, 0, 1])
    # points_weights[2, 0, 1] = np.array([2, 0, 1])

    # points_weights[0, 1, 1] = np.array([0, 1, 1])
    # points_weights[1, 1, 1] = np.array([1, 1.5, 1])
    # points_weights[2, 1, 1] = np.array([2, 0.75, 1])

    # p = 3
    # q = r = 1
    # points_weights = np.zeros((p + 1, q + 1, r + 1, 3), dtype=float)

    # # layer 0
    # points_weights[0, 0, 0] = np.array([0, 0, 0])
    # points_weights[1, 0, 0] = np.array([1, 0, 0])
    # points_weights[2, 0, 0] = np.array([2, 0, 0])
    # points_weights[3, 0, 0] = np.array([3, 0, 0])

    # points_weights[0, 1, 0] = np.array([0, 1, 0])
    # points_weights[1, 1, 0] = np.array([1, 1.5, 0])
    # points_weights[2, 1, 0] = np.array([2, 1.0, 0])
    # points_weights[3, 1, 0] = np.array([3, 0.5, 0])

    # # layer 1
    # points_weights[0, 0, 1] = np.array([0, 0, 1])
    # points_weights[1, 0, 1] = np.array([1, 0, 1])
    # points_weights[2, 0, 1] = np.array([2, 0, 1])
    # points_weights[3, 0, 1] = np.array([3, 0, 1])

    # points_weights[0, 1, 1] = np.array([0, 1, 1])
    # points_weights[1, 1, 1] = np.array([1, 1.5, 1])
    # points_weights[2, 1, 1] = np.array([2, 1.0, 1])
    # points_weights[3, 1, 1] = np.array([3, 0.5, 1])

    # p = 3
    # q = 2
    # r = 1
    # Ps = np.zeros((p + 1, q + 1, r + 1, 3), dtype=float)

    # # layer 0
    # Ps[0, 0, 0] = np.array([0, 0, 0])
    # Ps[1, 0, 0] = np.array([1, 0, 0])
    # Ps[2, 0, 0] = np.array([2, 0, 0])
    # Ps[3, 0, 0] = np.array([3, 0, 0])

    # Ps[0, 1, 0] = np.array([0, 0.25, 0])
    # Ps[1, 1, 0] = np.array([1, 0.25, 0])
    # Ps[2, 1, 0] = np.array([2, 0.25, 0])
    # Ps[3, 1, 0] = np.array([3, 0.25, 0])

    # Ps[0, 2, 0] = np.array([0, 1, 0])
    # Ps[1, 2, 0] = np.array([1, 1.5, 0])
    # Ps[2, 2, 0] = np.array([2, 1.0, 0])
    # Ps[3, 2, 0] = np.array([3, 0.5, 0])

    # # layer 1
    # Ps[0, 0, 1] = np.array([0, 0, 1])
    # Ps[1, 0, 1] = np.array([1, 0, 1])
    # Ps[2, 0, 1] = np.array([2, 0, 1])
    # Ps[3, 0, 1] = np.array([3, 0, 1])

    # Ps[0, 1, 1] = np.array([0, 0.25, 1])
    # Ps[1, 1, 1] = np.array([1, 0.25, 1])
    # Ps[2, 1, 1] = np.array([2, 0.25, 1])
    # Ps[3, 1, 1] = np.array([3, 0.25, 1])

    # Ps[0, 2, 1] = np.array([0, 1, 1])
    # Ps[1, 2, 1] = np.array([1, 1.5, 1])
    # Ps[2, 2, 1] = np.array([2, 1.0, 1])
    # Ps[3, 2, 1] = np.array([3, 0.5, 1])

    mesh = BezierMesh3D(p, q, r)

    here = Path(__file__)
    path = here.parent / str(here.stem + ".vtu")
    mesh.export(points_weights, path)


def VTK_LAGRANGE_WEDGE():
    h = 3

    PW0 = np.array([0, 0, 0, 1], dtype=float)
    PW1 = np.array([1, 0, 0, 1], dtype=float)
    PW2 = np.array([0, 1, 0, 1], dtype=float)

    PW3 = np.array([0, 0, h, 1], dtype=float)
    PW4 = np.array([1, 0, h, 1], dtype=float)
    PW5 = np.array([0, 1, h, 1], dtype=float)

    PW6 = np.array([0, 0, 2 * h, 1], dtype=float)
    PW7 = np.array([1, 0, 2 * h, 1], dtype=float)
    PW8 = np.array([0, 1, 2 * h, 1], dtype=float)

    vtk_points_weights = np.array(
        [
            # bottom
            PW0,
            PW1,
            PW2,
            # top
            PW6,
            PW7,
            PW8,
            # first
            PW3,
            PW4,
            PW5,
        ]
    )
    # vtk_points_weights = []
    # vtk_points_weights.append(PW0)
    # vtk_points_weights.append(PW1)
    # vtk_points_weights.append(PW2)
    # vtk_points_weights.append(PW3)
    # vtk_points_weights.append(PW4)
    # vtk_points_weights.append(PW5)

    vtk_points = vtk_points_weights[:, :3]
    vtk_weights = vtk_points_weights[:, 3]

    # cells = [("VTK_LAGRANGE_WEDGE", np.arange(len(vtk_points))[None])]
    cells = [("VTK_BEZIER_WEDGE", np.arange(len(vtk_points))[None])]
    # cells = [("VTK_BEZIER_HEXAHEDRON", np.arange(len(vtk_points))[None])]

    higher_order_degrees = [
        np.array([1, 1, 2])[None],
    ]

    point_data = {"RationalWeights": vtk_weights}
    # point_data = None
    cell_data = {"HigherOrderDegrees": higher_order_degrees}
    # cell_data = None

    here = Path(__file__)
    path = here.parent / str(here.stem + ".vtu")

    meshio.write_points_cells(
        filename=path,
        points=vtk_points,
        cells=cells,
        point_data=point_data,
        cell_data=cell_data,
        binary=False,
    )


def VTK_BEZIER_WEDGE_cylinder():

    s33 = np.sqrt(3) / 3

    h = 3

    vertices = [
        # layer 0
        [0, -s33, 0, 1],
        [1 / 2, s33 / 2, 0, 1],
        [-1 / 2, s33 / 2, 0, 1],
        # layer 1
        [0, -s33, h, 1],
        [1 / 2, s33 / 2, h, 1],
        [-1 / 2, s33 / 2, h, 1],
    ]

    edges = [
        # layer 0
        [1, -s33, 0, 1 / 2],
        [0, 2 * s33, 0, 1 / 2],
        [-1, -s33, 0, 1 / 2],
        # layer 1
        [1, -s33, h, 1 / 2],
        [0, 2 * s33, h, 1 / 2],
        [-1, -s33, h, 1 / 2],
    ]

    faces = []

    volume = []

    vtk_points_weights = np.array(vertices + edges + faces + volume)

    vtk_points = vtk_points_weights[:, :3]
    vtk_weights = vtk_points_weights[:, 3]

    cells = [("VTK_BEZIER_WEDGE", np.arange(len(vtk_points))[None])]

    higher_order_degrees = [
        np.array([2, 2, 1])[None],
    ]

    point_data = {"RationalWeights": vtk_weights}
    cell_data = {"HigherOrderDegrees": higher_order_degrees}

    here = Path(__file__)
    path = here.parent / str(here.stem + ".vtu")

    meshio.write_points_cells(
        filename=path,
        points=vtk_points,
        cells=cells,
        point_data=point_data,
        cell_data=cell_data,
        binary=False,
    )


def VTK_BEZIER_HEXAHEDRON_cylinder(radius=1):

    s22 = np.sqrt(2) / 2

    vertices = [
        # layer 0
        [0, -1, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 0, 1],
        [-1, 0, 0, 1],
        # layer 1
        [0, -1, 3, 1],
        [1, 0, 3, 1],
        [0, 1, 3, 1],
        [-1, 0, 3, 1],
    ]

    edges = [
        # layer 0
        [1, -1, 0, s22],
        [1, 1, 0, s22],
        [-1, 1, 0, s22],
        [-1, -1, 0, s22],
        # layer 1
        [1, -1, 3, s22],
        [1, 1, 3, s22],
        [-1, 1, 3, s22],
        [-1, -1, 3, s22],
    ]

    faces = [
        [0, 0, 0, 1],
        [0, 0, 3, 1],
    ]

    volume = []

    vtk_points_weights = np.array(vertices + edges + faces + volume)
    vtk_points_weights[:, :2] *= radius

    vtk_points = vtk_points_weights[:, :3]
    vtk_weights = vtk_points_weights[:, 3]

    cells = [("VTK_BEZIER_HEXAHEDRON", np.arange(len(vtk_points))[None])]

    higher_order_degrees = [
        np.array([2, 2, 1])[None],
    ]

    # return points, cells, None, None
    point_data = {"RationalWeights": vtk_weights}
    # point_data = None
    cell_data = {"HigherOrderDegrees": higher_order_degrees}

    here = Path(__file__)
    path = here.parent / str(here.stem + ".vtu")

    meshio.write_points_cells(
        filename=path,
        points=vtk_points,
        cells=cells,
        point_data=point_data,
        cell_data=cell_data,
        binary=True,
    )


def VTK_BEZIER_HEXAHEDRON_quarter_torous():

    phis = np.linspace(0, 1, num=4) * np.pi / 2
    A_IKs = np.array([A_IK_basic(phi).z() for phi in phis])

    r_OPs = np.array(
        [
            np.array(
                [
                    np.cos(phi),
                    np.sin(phi),
                    0,
                ]
            )
            for phi in phis
        ]
    )

    s22 = np.sqrt(2) / 2

    vertices = [
        # layer 0
        [0, -1, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 0, 1],
        [-1, 0, 0, 1],
        # layer 1
        [0, -1, 3, 1],
        [1, 0, 3, 1],
        [0, 1, 3, 1],
        [-1, 0, 3, 1],
    ]

    edges = [
        # layer 0
        [1, -1, 0, s22],
        [1, 1, 0, s22],
        [-1, 1, 0, s22],
        [-1, -1, 0, s22],
        # layer 1
        [1, -1, 3, s22],
        [1, 1, 3, s22],
        [-1, 1, 3, s22],
        [-1, -1, 3, s22],
    ]

    faces = [
        [0, 0, 0, 1],
        [0, 0, 3, 1],
    ]

    volume = []

    vtk_points_weights = np.array(vertices + edges + faces + volume)

    vtk_points = vtk_points_weights[:, :3]
    vtk_weights = vtk_points_weights[:, 3]

    cells = [("VTK_BEZIER_HEXAHEDRON", np.arange(len(vtk_points))[None])]

    higher_order_degrees = [
        np.array([2, 2, 1])[None],
    ]

    # return points, cells, None, None
    point_data = {"RationalWeights": vtk_weights}
    # point_data = None
    cell_data = {"HigherOrderDegrees": higher_order_degrees}

    here = Path(__file__)
    path = here.parent / str(here.stem + ".vtu")

    meshio.write_points_cells(
        filename=path,
        points=vtk_points,
        cells=cells,
        point_data=point_data,
        cell_data=cell_data,
        binary=True,
    )


def VTK_BEZIER_HEXAHEDRON_quarter_circle():

    s33 = np.sqrt(3) / 3

    vertices = [
        [0, 0, 0, 1],
        [2, 2, 0, 1],
    ]

    edges = [
        [1, 0, 0, s33],
        [2, 1, 0, s33],
    ]

    faces = []

    volume = []

    vtk_points_weights = np.array(vertices + edges + faces + volume)

    vtk_points = vtk_points_weights[:, :3]
    vtk_weights = vtk_points_weights[:, 3]

    cells = [("VTK_BEZIER_CURVE", np.arange(len(vtk_points))[None])]

    higher_order_degrees = [
        np.array([3, 0, 0])[None],
    ]

    point_data = {"RationalWeights": vtk_weights}

    cell_data = {"HigherOrderDegrees": higher_order_degrees}

    here = Path(__file__)
    path = here.parent / str(here.stem + ".vtu")

    meshio.write_points_cells(
        filename=path,
        points=vtk_points,
        cells=cells,
        point_data=point_data,
        cell_data=cell_data,
        binary=False,
    )


if __name__ == "__main__":
    # VTK_LAGRANGE_WEDGE()
    VTK_BEZIER_WEDGE_cylinder()
    # VTK_BEZIER_HEXAHEDRON_cylinder()
    # VTK_BEZIER_HEXAHEDRON_quarter_torous()
    # VTK_BEZIER_HEXAHEDRON_quarter_circle()
    # test_volume()
    exit()

    p = 2
    q = 2
    r = 1
    # r = 0
    points_weights = np.zeros((p + 1, q + 1, r + 1, 4), dtype=float)

    # # layer 0
    # s22 = np.sqrt(2) / 2
    # points_weights[0, 0, 0] = np.array([-1, -1, 0, s22])
    # points_weights[1, 0, 0] = np.array([ 0, -1, 0,   1])
    # points_weights[2, 0, 0] = np.array([ 1, -1, 0, s22])

    # points_weights[0, 1, 0] = np.array([-1,  0, 0,   1])
    # points_weights[1, 1, 0] = np.array([ 0,  0, 0,   1])
    # points_weights[2, 1, 0] = np.array([ 1,  0, 0,   1])

    # points_weights[0, 2, 0] = np.array([-1,  1, 0, s22])
    # points_weights[1, 2, 0] = np.array([ 0,  1, 0,   1])
    # points_weights[2, 2, 0] = np.array([ 1,  1, 0, s22])

    # # layer 1
    # points_weights[0, 0, 1] = np.array([-1, -1, 3, s22])
    # points_weights[1, 0, 1] = np.array([ 0, -1, 3,   1])
    # points_weights[2, 0, 1] = np.array([ 1, -1, 3, s22])

    # points_weights[0, 1, 1] = np.array([-1,  0, 3,   1])
    # points_weights[1, 1, 1] = np.array([ 0,  0, 3,   1])
    # points_weights[2, 1, 1] = np.array([ 1,  0, 3,   1])

    # points_weights[0, 2, 1] = np.array([-1,  1, 3, s22])
    # points_weights[1, 2, 1] = np.array([ 0,  1, 3,   1])
    # points_weights[2, 2, 1] = np.array([ 1,  1, 3, s22])

    # layer 0
    s22 = np.sqrt(2) / 2
    points_weights[0, 0, 0] = np.array([-1, -1, 0, s22])
    points_weights[1, 0, 0] = np.array([0, -1, 0, 1])
    points_weights[2, 0, 0] = np.array([1, -1, 0, s22])

    points_weights[0, 1, 0] = np.array([-1, 0, 0, 1])
    points_weights[1, 1, 0] = np.array([0, 0, 0, 1])
    points_weights[2, 1, 0] = np.array([1, 0, 0, 1])

    points_weights[0, 2, 0] = np.array([-1, 1, 0, s22])
    points_weights[1, 2, 0] = np.array([0, 1, 0, 1])
    points_weights[2, 2, 0] = np.array([1, 1, 0, s22])

    # # layer 1
    # points_weights[0, 0, 1] = np.array([-1, -1, 3, s22])
    # points_weights[1, 0, 1] = np.array([ 0, -1, 3,   1])
    # points_weights[2, 0, 1] = np.array([ 1, -1, 3, s22])

    # points_weights[0, 1, 1] = np.array([-1,  0, 3,   1])
    # points_weights[1, 1, 1] = np.array([ 0,  0, 3,   1])
    # points_weights[2, 1, 1] = np.array([ 1,  0, 3,   1])

    # points_weights[0, 2, 1] = np.array([-1,  1, 3, s22])
    # points_weights[1, 2, 1] = np.array([ 0,  1, 3,   1])
    # points_weights[2, 2, 1] = np.array([ 1,  1, 3, s22])

    class BezierMesh3D:
        def __init__(self, p, q, r):
            self.p = p
            self.q = q
            self.r = r
            self.vtk_cell_type = "VTK_BEZIER_HEXAHEDRON"

        def split_vtk(self, P):
            """Rearranges either a Point Array with dimensions
            (p+1)x(q+1)x(r+1)x(dim) to vtk ordering, see vtk:

            References:
            -----------

            vtk: https://blog.kitware.com/modeling-arbitrary-order-lagrange-finite-elements-in-the-visualization-toolkit/
            """
            # # extract dimensions
            # p, q, r, dim = P.shape

            # #######################
            # # 1. vertices (corners)
            # #######################
            # # fmt: off
            # # points.extend(
            # vertices = [
            #     P[ 0,  0,  0],
            #     P[-1,  0,  0],
            #     P[-1, -1,  0],
            #     P[ 0, -1,  0],
            #     P[ 0,  0, -1],
            #     P[-1,  0, -1],
            #     P[-1, -1, -1],
            #     P[ 0, -1, -1],
            # ]
            # # fmt: on

            # ##########
            # # 2. edges
            # ##########
            # # fmt: off
            # edges = []
            # for iz in [0, -1]:
            #     edges.extend(P[1:-1,    0, iz])
            #     edges.extend(P[  -1, 1:-1, iz])
            #     edges.extend(P[1:-1,   -1, iz])
            #     edges.extend(P[   0, 1:-1, iz])
            # for ix in [0, -1]:
            #     edges.extend(P[ix, 0, 1:-1])
            # for ix in [0, -1]:
            #     edges.extend(P[ix, -1, 1:-1])
            # # fmt: on

            # ##########
            # # 3. faces
            # ##########
            # # yz
            # faces = []
            # for ix in [0, -1]:
            #     for iz in range(r - 2):
            #         faces.extend(P[ix, 1:-1, iz + 1])
            # # xz
            # for iy in [0, -1]:
            #     for iz in range(r - 2):
            #         faces.extend(P[1:-1, iy, iz + 1])
            # # xy
            # for iz in [0, -1]:
            #     for iy in range(q - 2):
            #         faces.extend(P[1:-1, iy + 1, iz])

            # ###########
            # # 4. volume
            # ###########
            # volume = []
            # for iz in range(r - 2):
            #     for iy in range(q - 2):
            #         for ix in range(p - 2):
            #             volume.extend([P[ix + 1, iy + 1, iz + 1]])

            # layer 0
            s22 = np.sqrt(2) / 2
            vertices = [
                # layer 0
                [0, -1, 0, 1],
                [1, 0, 0, 1],
                [0, 1, 0, 1],
                [-1, 0, 0, 1],
                # layer 1
                [0, -1, 3, 1],
                [1, 0, 3, 1],
                [0, 1, 3, 1],
                [-1, 0, 3, 1],
            ]

            edges = [
                # layer 0
                [1, -1, 0, s22],
                [1, 1, 0, s22],
                [-1, 1, 0, s22],
                [-1, -1, 0, s22],
                # layer 1
                [1, -1, 3, s22],
                [1, 1, 3, s22],
                [-1, 1, 3, s22],
                [-1, -1, 3, s22],
            ]

            faces = [
                [0, 0, 0, 1],
                [0, 0, 3, 1],
            ]

            volume = []

            return vertices, edges, faces, volume

        def points_vtk(self, P):
            vertices, edges, faces, volume = self.split_vtk(P)
            return np.array(vertices + edges + faces + volume)
            # return np.concatenate(
            #     (
            #         np.array(vertices),
            #         np.array(edges),
            #         np.array(faces),
            #         np.array(volume),
            #     )
            # )

        # TODO: This is only a single Bezier element!
        def export(self, points_weights, file):
            vtk_points_weights = self.points_vtk(points_weights)
            vtk_points = vtk_points_weights[:, :3]
            vtk_weights = vtk_points_weights[:, 3]

            cells = [(self.vtk_cell_type, np.arange(len(vtk_points))[None])]

            higher_order_degrees = [
                np.array([p, q, r])[None],
            ]

            # return points, cells, None, None
            point_data = {"RationalWeights": vtk_weights}
            # point_data = None
            cell_data = {"HigherOrderDegrees": higher_order_degrees}

            meshio.write_points_cells(
                filename=file,
                points=vtk_points,
                cells=cells,
                point_data=point_data,
                cell_data=cell_data,
                binary=False,
            )

    mesh = BezierMesh3D(p, q, r)

    # vertices, edges, faces, volume = mesh.split_vtk(points_weights[:, :, :, :3])
    # print(f"vertices:\n{vertices}")
    # print(f"edges:\n{edges}")
    # print(f"faces:\n{faces}")
    # print(f"volume:\n{volume}")

    # fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="3d"))
    # ax.scatter(*points_weights.T, color="black", label="points")
    # ax.scatter(*np.array(vertices).T, color="red", s=100, label="vertices")
    # if edges:
    #     ax.scatter(*np.array(edges).T, color="green", s=100, label="edges")
    # if faces:
    #     ax.scatter(*np.array(faces).T, color="blue", s=100, label="faces")
    # if volume:
    #     ax.scatter(*np.array(volume).T, color="black", s=100, label="volume")
    # ax.grid()
    # ax.legend()
    # plt.show()

    here = Path(__file__)
    path = here.parent / str(here.stem + ".vtu")
    mesh.export(points_weights, path)
