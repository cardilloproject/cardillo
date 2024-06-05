from numpy.polynomial import Polynomial
import numpy as np
from math import comb
import matplotlib.pyplot as plt
from vtk import VTK_BEZIER_HEXAHEDRON

from cardillo.visualization import Export
from cardillo.solver import Solution
from pathlib import Path

from scipy.sparse import lil_matrix


class BernsteinBasis:
    def __init__(self, degree, interval=[0, 1]):
        """Bernstein basis functions, see wiki:

        References:
        -----------
        wiki: https://en.wikipedia.org/wiki/Bernstein_polynomial#Definition.
        """
        self.degree = degree

        # compute equally spaced points on [0, 1]
        nus = np.linspace(0, 1, num=degree + 1)

        # polynomial x^1
        P = Polynomial([0, 1], domain=interval, window=[0, 1])

        # recursively construct Bernstein shape functions on [0, 1] and map
        # them on interval
        self.li = np.ones(degree + 1, dtype=object)
        for i in range(degree + 1):
            self.li[i] = comb(degree, i) * P**i * (1.0 - P) ** (degree - i)

    def __str__(self):
        tmp = f"{self.degree}th order Bernstein basis:\n"
        l = []
        for i, li in enumerate(self.li):
            l.append(f"  B{i}^{self.degree}: " + str(li))
            if i < self.degree:
                l.append("\n")
        return tmp + "".join(l)

    def __call__(self, xis):
        xis = np.atleast_1d(xis)
        values = np.zeros((len(xis), self.degree + 1), dtype=float)
        for i, xii in enumerate(xis):
            for j in range(self.degree + 1):
                values[i, j] = self.li[j](xii)
        return values.squeeze()

    def deriv(self, xis, n=1):
        xis = np.atleast_1d(xis)
        values = np.zeros((len(xis), self.degree + 1), dtype=float)
        for i, xii in enumerate(xis):
            for j in range(self.degree + 1):
                values[i, j] = self.li[j].deriv(n)(xii)
        return values.squeeze()


def test_line():
    # example of quadratic basis
    basis2 = BernsteinBasis(2)
    P2s = np.array([[0, 0], [0.5, 1.0], [1.0, 0.25]])

    # example of cubic basis
    basis3 = BernsteinBasis(3)
    P3s = np.array(
        [
            [0, 0],
            [0.25, 1.5],
            [0.75, 1.25],
            [1.0, 0.5],
        ]
    )

    num = 100
    xis = np.linspace(0, 1, num=num)
    r2 = np.zeros((num, 2))
    r3 = np.zeros((num, 2))
    for i, xi in enumerate(xis):
        r2[i] = basis2(xi) @ P2s
        r3[i] = basis3(xi) @ P3s

    fig, ax = plt.subplots(1, 2)

    ax[0].set_title("quadratic Bézier spline")
    ax[0].plot(*r2.T, "-k")
    ax[0].plot(*P2s.T, "-ob")
    ax[0].grid()

    ax[1].set_title("cubic Bézier spline")
    ax[1].plot(*r3.T, "-k")
    ax[1].plot(*P3s.T, "-ob")
    ax[1].grid()

    plt.show()


def test_surface():
    basis2 = BernsteinBasis(2)

    Ps = np.zeros((2, 3, 3), dtype=float)

    Ps[:, 0, 0] = np.array([0, 0])
    Ps[:, 0, 1] = np.array([1, 0])
    Ps[:, 0, 2] = np.array([2, 0])

    Ps[:, 1, 0] = np.array([0, 1])
    Ps[:, 1, 1] = np.array([1.5, 1])
    Ps[:, 1, 2] = np.array([3, 1])

    Ps[:, 2, 0] = np.array([0.5, 2])
    Ps[:, 2, 1] = np.array([1.5, 2.5])
    Ps[:, 2, 2] = np.array([3, 2])

    num = 25
    xis = np.linspace(0, 1, num=num)
    etas = np.linspace(0, 1, num=num)

    def eval(xi, eta):
        Bi = basis2(xi)
        Bj = basis2(eta)
        res = np.zeros(2)
        for i in range(3):
            for j in range(3):
                res += Bi[i] * Bj[j] * Ps[:, i, j]
        return res

    xxis, eetas = np.meshgrid(xis, etas, sparse=False)
    zzs = np.zeros((2, num, num))
    for i in range(num):
        for j in range(num):
            zzs[:, i, j] = eval(xxis[i, j], eetas[i, j])

    fig, ax = plt.subplots()
    ax.plot(*zzs, "-ok")
    ax.plot(*zzs.transpose(0, 2, 1), "-ok")
    ax.plot(*Ps, "-ob")
    ax.plot(*Ps.transpose(0, 2, 1), "-ob")
    ax.grid()
    plt.show()


def test_volume():
    # p = q = r = 1
    # Ps = np.zeros((p + 1, q + 1, r + 1, 3), dtype=float)

    # # layer 0
    # Ps[0, 0, 0] = np.array([0, 0, 0])
    # Ps[1, 0, 0] = np.array([1, 0, 0])

    # Ps[0, 1, 0] = np.array([0, 1, 0])
    # Ps[1, 1, 0] = np.array([1, 1, 0])

    # # layer 1
    # Ps[0, 0, 1] = np.array([0, 0, 1])
    # Ps[1, 0, 1] = np.array([1, 0, 1])

    # Ps[0, 1, 1] = np.array([0, 1, 1])
    # Ps[1, 1, 1] = np.array([1, 1, 1])

    # p = 2
    # q = r = 1
    # Ps = np.zeros((p + 1, q + 1, r + 1, 3), dtype=float)

    # # layer 0
    # Ps[0, 0, 0] = np.array([0, 0, 0])
    # Ps[1, 0, 0] = np.array([1, 0, 0])
    # Ps[2, 0, 0] = np.array([2, 0, 0])

    # Ps[0, 1, 0] = np.array([0, 1, 0])
    # Ps[1, 1, 0] = np.array([1, 1.5, 0])
    # Ps[2, 1, 0] = np.array([2, 0.75, 0])

    # # layer 1
    # Ps[0, 0, 1] = np.array([0, 0, 1])
    # Ps[1, 0, 1] = np.array([1, 0, 1])
    # Ps[2, 0, 1] = np.array([2, 0, 1])

    # Ps[0, 1, 1] = np.array([0, 1, 1])
    # Ps[1, 1, 1] = np.array([1, 1.5, 1])
    # Ps[2, 1, 1] = np.array([2, 0.75, 1])

    # p = 3
    # q = r = 1
    # Ps = np.zeros((p + 1, q + 1, r + 1, 3), dtype=float)

    # # layer 0
    # Ps[0, 0, 0] = np.array([0, 0, 0])
    # Ps[1, 0, 0] = np.array([1, 0, 0])
    # Ps[2, 0, 0] = np.array([2, 0, 0])
    # Ps[3, 0, 0] = np.array([3, 0, 0])

    # Ps[0, 1, 0] = np.array([0, 1, 0])
    # Ps[1, 1, 0] = np.array([1, 1.5, 0])
    # Ps[2, 1, 0] = np.array([2, 1.0, 0])
    # Ps[3, 1, 0] = np.array([3, 0.5, 0])

    # # layer 1
    # Ps[0, 0, 1] = np.array([0, 0, 1])
    # Ps[1, 0, 1] = np.array([1, 0, 1])
    # Ps[2, 0, 1] = np.array([2, 0, 1])
    # Ps[3, 0, 1] = np.array([3, 0, 1])

    # Ps[0, 1, 1] = np.array([0, 1, 1])
    # Ps[1, 1, 1] = np.array([1, 1.5, 1])
    # Ps[2, 1, 1] = np.array([2, 1.0, 1])
    # Ps[3, 1, 1] = np.array([3, 0.5, 1])

    p = 3
    q = 2
    r = 1
    Ps = np.zeros((p + 1, q + 1, r + 1, 3), dtype=float)

    # layer 0
    Ps[0, 0, 0] = np.array([0, 0, 0])
    Ps[1, 0, 0] = np.array([1, 0, 0])
    Ps[2, 0, 0] = np.array([2, 0, 0])
    Ps[3, 0, 0] = np.array([3, 0, 0])

    Ps[0, 1, 0] = np.array([0, 0.25, 0])
    Ps[1, 1, 0] = np.array([1, 0.25, 0])
    Ps[2, 1, 0] = np.array([2, 0.25, 0])
    Ps[3, 1, 0] = np.array([3, 0.25, 0])

    Ps[0, 2, 0] = np.array([0, 1, 0])
    Ps[1, 2, 0] = np.array([1, 1.5, 0])
    Ps[2, 2, 0] = np.array([2, 1.0, 0])
    Ps[3, 2, 0] = np.array([3, 0.5, 0])

    # layer 1
    Ps[0, 0, 1] = np.array([0, 0, 1])
    Ps[1, 0, 1] = np.array([1, 0, 1])
    Ps[2, 0, 1] = np.array([2, 0, 1])
    Ps[3, 0, 1] = np.array([3, 0, 1])

    Ps[0, 1, 1] = np.array([0, 0.25, 1])
    Ps[1, 1, 1] = np.array([1, 0.25, 1])
    Ps[2, 1, 1] = np.array([2, 0.25, 1])
    Ps[3, 1, 1] = np.array([3, 0.25, 1])

    Ps[0, 2, 1] = np.array([0, 1, 1])
    Ps[1, 2, 1] = np.array([1, 1.5, 1])
    Ps[2, 2, 1] = np.array([2, 1.0, 1])
    Ps[3, 2, 1] = np.array([3, 0.5, 1])

    # p = 4
    # q = 3
    # r = 2
    # Ps = np.zeros((p + 1, q + 1, r + 1, 3), dtype=float)

    # xis = np.linspace(0, 1, num=p+1)
    # etas = np.linspace(0, 1, num=q+1)
    # zetas = np.linspace(0, 1, num=r+1)

    # # xi_etas = np.outer(xis, etas)
    # # Ps = np.outer(xi_etas, zetas)
    # for i in range(p + 1):
    #     for j in range(q + 1):
    #         for k in range(r + 1):
    #             Ps[i, j, k] = np.array([xis[i], etas[j], zetas[k]])

    # # Ps = np.random.rand(p+1, q+1, r+1, 3)

    class BezierMesh:
        def __init__(self, p, q, r):
            self.p = p
            self.q = q
            self.r = r

        def split_vtk(self, P):
            """Rearranges either a Point Array with dimensions
            (p+1)x(q+1)x(r+1)x(dim) to vtk ordering, see vtk:

            References:
            -----------

            vtk: https://blog.kitware.com/modeling-arbitrary-order-lagrange-finite-elements-in-the-visualization-toolkit/
            """
            # extract dimensions
            p, q, r, dim = P.shape

            #######################
            # 1. vertices (corners)
            #######################
            # fmt: off
            # points.extend(
            vertices = [
                P[ 0,  0,  0],
                P[-1,  0,  0],
                P[-1, -1,  0],
                P[ 0, -1,  0],
                P[ 0,  0, -1],
                P[-1,  0, -1],
                P[-1, -1, -1],
                P[ 0, -1, -1],
            ]
            # fmt: on

            ##########
            # 2. edges
            ##########
            # fmt: off
            edges = []
            for iz in [0, -1]:
                edges.extend(P[1:-1,    0, iz])
                edges.extend(P[  -1, 1:-1, iz])
                edges.extend(P[1:-1,   -1, iz])
                edges.extend(P[   0, 1:-1, iz])
            for ix in [0, -1]:
                edges.extend(P[ix, 0, 1:-1])
            for ix in [0, -1]:
                edges.extend(P[ix, -1, 1:-1])
            # fmt: on

            ##########
            # 3. faces
            ##########
            # yz
            faces = []
            for ix in [0, -1]:
                for iz in range(r - 2):
                    faces.extend(P[ix, 1:-1, iz + 1])
            # xz
            for iy in [0, -1]:
                for iz in range(r - 2):
                    faces.extend(P[1:-1, iy, iz + 1])
            # xy
            for iz in [0, -1]:
                for iy in range(q - 2):
                    faces.extend(P[1:-1, iy + 1, iz])

            ###########
            # 4. volume
            ###########
            volume = []
            for iz in range(r - 2):
                for iy in range(q - 2):
                    for ix in range(p - 2):
                        volume.extend([P[ix + 1, iy + 1, iz + 1]])

            return vertices, edges, faces, volume

        def points_vtk(self, P):
            vertices, edges, faces, volume = self.split_vtk(P)
            return np.array(vertices + edges + faces + volume)

        # TODO: This is only a single Bezier element!
        # TODO: Is this function really used? Since it returns nothing,
        # adaptation for vtk is not complete.
        def export(self, P, file):
            points = self.points_vtk(P)

            cells = [(VTK_BEZIER_HEXAHEDRON, np.arange(len(points))[None])]

            higher_order_degrees = [
                np.array([p, q, r])[None],
            ]

            # return points, cells, None, None
            point_data = None
            cell_data = {"HigherOrderDegrees": higher_order_degrees}

            # # TODO: Replace this with vtk if still important
            # import meshio

            # meshio.write_points_cells(
            #     filename=file,
            #     points=points,
            #     cells=cells,
            #     point_data=point_data,
            #     cell_data=cell_data,
            #     binary=False,
            # )

    mesh = BezierMesh(p, q, r)

    vertices, edges, faces, volume = mesh.split_vtk(Ps)
    print(f"vertices:\n{vertices}")
    print(f"edges:\n{edges}")
    print(f"faces:\n{faces}")
    print(f"volume:\n{volume}")

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="3d"))
    ax.scatter(*Ps.T, color="black", label="points")
    ax.scatter(*np.array(vertices).T, color="red", s=100, label="vertices")
    if edges:
        ax.scatter(*np.array(edges).T, color="green", s=100, label="edges")
    if faces:
        ax.scatter(*np.array(faces).T, color="blue", s=100, label="faces")
    if volume:
        ax.scatter(*np.array(volume).T, color="black", s=100, label="volume")
    ax.grid()
    ax.legend()
    plt.show()

    here = Path(__file__)
    path = here.parent / str(here.stem + ".vtu")
    mesh.export(Ps, path)

    exit()

    basis_p = BernsteinBasis(p)
    basis_q = BernsteinBasis(q)
    basis_r = BernsteinBasis(r)

    def eval(xi, eta, zeta):
        Bi = basis_p(xi)
        Bj = basis_q(eta)
        Bk = basis_r(zeta)
        res = np.zeros(3)
        for i in range(p + 1):
            for j in range(q + 1):
                for k in range(r + 1):
                    res += Bi[i] * Bj[j] * Bk[k] * Ps[i, j, k]
        return res

    num = 5
    xis = np.linspace(0, 1, num=num)
    etas = np.linspace(0, 1, num=num)
    zetas = np.linspace(0, 1, num=num)
    z = np.zeros((num, num, num, 3))
    for i in range(num):
        for j in range(num):
            for k in range(num):
                z[i, j, k] += eval(xis[i], etas[j], zetas[k])

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="3d"))

    def plot_3D_mesh(ax, P, style="-ok"):
        n1, n2, n3, dim = P.shape
        for i in range(n1):
            for j in range(n2):
                ax.plot(*P[i, j].T, style)
            for k in range(n3):
                ax.plot(*P[i, :, k].T, style)
        for j in range(n2):
            for k in range(n3):
                ax.plot(*P[:, j, k].T, style)

    plot_3D_mesh(ax, z, "-k")
    plot_3D_mesh(ax, Ps, "-ob")

    ax.grid()
    plt.show()


def test_circle():
    """Quintic Bezier circle, see Piefl1997 Ex7.10."""
    radius = 1

    # basis = BernsteinBasis(6)

    # P0 = [ 0, -5, 0, 1]
    # P1 = [ 4, -1, 1, 1]
    # P2 = [ 2,  3, 0, 1]
    # P3 = [-2,  3, 0, 1]
    # P4 = [-4, -1, 0, 1]
    # points_weights = radius * np.array(
    #     [
    #         # [ 0, -5, 0, 4],
    #         # [ 4, -1, 0, 1],
    #         # [ 2,  3, 0, 1],
    #         # [-2,  3, 0, 1],
    #         # [-4, -1, 0, 1],
    #         # [ 0, -5, 0, 4],
    #         P0,
    #         P1,
    #         P2,
    #         P3,
    #         P4,
    #         P0,
    #         P1,
    #     ]
    # )
    # # points = radius * np.array([
    # #     [0, -5, 0],
    # #     [4, -1, 1],
    # #     [2, 3, 0],
    # #     [-2, 3, -1],
    # #     [-4, -1, 0],
    # #     [0, -5, 0],
    # # ])

    # # weights = np.array(
    # #     [
    # #         5,
    # #         1,
    # #         1,
    # #         1,
    # #         1,
    # #         5,
    # #     ]
    # # )

    # fourth-degree full circle, see Piegl1997 Ex.7.9
    basis = BernsteinBasis(4)

    # points_weights = radius * np.array([
    #     [3, 0, 0, 3],
    #     [0, 3, 0, 0],
    #     [-3, 0, 0, 1],
    #     [0, -3, 0, 0],
    #     [3, 0, 0, 3],
    # ])
    points_weights = radius * np.array(
        [
            [3, 0, 0, 3],
            [0, 3, 1, 0],
            [-3, 0, 0, 1],
            [0, -3, 0, 0],
            [3, 0, 0, 3],
        ]
    )

    points = points_weights[:, :-1]
    weights = points_weights[:, -1]

    num = 100
    xis = np.linspace(0, 1, num=num)
    rs = np.zeros((num, 3))
    for i, xi in enumerate(xis):
        N = basis(xi)
        w = N @ weights
        # rs[i] = (weights * N) @ points / w
        # rs[i] = (weights * N) @ points
        rs[i] = N @ points / w
        # rs[i] = basis(xi) @ points

    # fig, ax = plt.subplots()
    # ax.set_title("quadratic Bézier spline")
    # ax.plot(*rs.T, "-k")
    # ax.plot(*points.T, "-ob")
    # ax.axis("equal")
    # ax.grid()

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="3d"))
    ax.plot(*rs.T)
    ax.plot(*points.T, "-or")
    ax.set_xlim3d(left=-radius, right=radius)
    ax.set_ylim3d(bottom=-radius, top=radius)
    ax.set_zlim3d(bottom=-radius, top=radius)

    plt.show()


def test_quintic_nonrational_circle():
    radius = 1

    basis = BernsteinBasis(5)

    points_weights = radius * np.array(
        [
            [1, 0, 0, 1],
            [1 / 5, 4 / 5, 1, 1 / 5],
            [-3 / 5, 2 / 5, 0, 1 / 5],
            [-3 / 5, -2 / 5, 0, 1 / 5],
            [1 / 5, -4 / 5, 0, 1 / 5],
            [1, 0, 0, 1],
        ]
    )

    points = points_weights[:, :-1]
    weights = points_weights[:, -1]

    num = 100
    xis = np.linspace(0, 1, num=num)
    rs = np.zeros((num, 3))
    for i, xi in enumerate(xis):
        N = basis(xi)
        w = N @ weights
        # rs[i] = (N * weights) @ points / w
        rs[i] = N @ points / w

    # fig, ax = plt.subplots()
    # ax.set_title("quadratic Bézier spline")
    # ax.plot(*rs.T, "-k")
    # ax.plot(*points.T, "-ob")
    # ax.axis("equal")
    # ax.grid()

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="3d"))
    ax.plot(*rs.T)
    ax.plot(*points.T, "-or")
    ax.set_xlim3d(left=-radius, right=radius)
    ax.set_ylim3d(bottom=-radius, top=radius)
    ax.set_zlim3d(bottom=-radius, top=radius)

    plt.show()


def C0_continous_control_points(unique_points):
    # dimension of the control points
    dim = unique_points.shape[1]

    # get length of additional points of the last n - 1 segments
    # they should be a multiple of 3
    np_other = len(unique_points[4:])
    assert np_other % 3 == 0

    # number of curve segments
    n = np_other // 3 + 1

    # redundant points that enforce C1-continuity
    points = np.zeros((4 * n, dim), dtype=float)

    # add first four points
    points[:4] = unique_points[:4]

    for j in range(1, n):
        # C0-continuity
        p3 = points[(j - 1) * 4 + 3]
        points[j * 4] = p3.copy()

        # third and fourth point
        points[j * 4 + 1] = unique_points[4 + (j - 1) * 3]
        points[j * 4 + 2] = unique_points[4 + (j - 1) * 3 + 1]
        points[j * 4 + 3] = unique_points[4 + (j - 1) * 3 + 2]

    return points


def C1_continous_control_points(unique_points):
    # dimension of the control points
    dim = unique_points.shape[1]

    # get length of additional points of the last n - 1 segments
    # they should be a multiple of 2
    np_other = len(unique_points[4:])
    assert np_other % 2 == 0

    # number of curve segments
    n = np_other // 2 + 1

    # redundant points that enforce C1-continuity
    points = np.zeros((4 * n, dim), dtype=float)

    # add first four points
    points[:4] = unique_points[:4]

    for j in range(1, n):
        # C0-continuity
        p3 = points[(j - 1) * 4 + 3]
        points[j * 4] = p3.copy()

        # C1-continuity
        p2 = points[(j - 1) * 4 + 2]
        points[j * 4 + 1] = 2.0 * p3.copy() - p2.copy()

        # third and fourth point
        points[j * 4 + 2] = unique_points[4 + (j - 1) * 2]
        points[j * 4 + 3] = unique_points[4 + (j - 1) * 2 + 1]

    return points


def C2_continous_control_points(unique_points):
    # dimension of the control points
    dim = unique_points.shape[1]

    # get length of additional points of the last n - 1 segments
    np_other = len(unique_points[4:])

    # number of curve segments
    n = np_other + 1

    # redundant points that enforce C1-continuity
    points = np.zeros((4 * n, dim), dtype=float)

    # add first four points
    points[:4] = unique_points[:4]

    for j in range(1, n):
        # C0-continuity
        p3 = points[(j - 1) * 4 + 3]
        points[j * 4] = p3.copy()

        # C1-continuity
        p2 = points[(j - 1) * 4 + 2]
        points[j * 4 + 1] = 2.0 * p3.copy() - p2.copy()

        # C2-continuity
        p1 = points[(j - 1) * 4 + 1]
        points[j * 4 + 2] = p1 + 4 * (p3 - p2)

        # fourth point
        points[j * 4 + 3] = unique_points[4 + (j - 1)]

    return points


def reduced_matrix_C0_continous(A, dim):
    N_p = A.shape[0]
    n_first = 4 * dim
    n_other = (N_p - n_first) // (4 * dim)
    N_up = (4 + 3 * n_other) * dim

    n = n_other + 1

    A_red = np.zeros((N_p, N_up), dtype=float)
    A_red[:, :n_first] = A[:, :n_first]

    for j in range(1, n):
        for d in range(dim):
            # C0 continuity
            A_red[:, n_first + (j - 1) * 3 * dim + d - dim] = (
                A[:, j * n_first + d - dim] + A[:, j * n_first + d]
            )

            # Second, third and fourth point
            A_red[:, n_first + (j - 1) * 3 * dim + d] = A[:, j * n_first + d + dim]
            A_red[:, n_first + (j - 1) * 3 * dim + d + dim] = A[
                :, j * n_first + d + 2 * dim
            ]
            A_red[:, n_first + (j - 1) * 3 * dim + d + 2 * dim] = A[
                :, j * n_first + d + 3 * dim
            ]

    return A_red


def reduced_matrix_C1_continous(A, dim):
    N_p = A.shape[0]
    n_first = 4 * dim
    n_other = (N_p - n_first) // (4 * dim)
    N_up = (4 + 2 * n_other) * dim

    n = n_other + 1

    A_red = np.zeros((N_p, N_up), dtype=float)
    A_red[:, :n_first] = A[:, :n_first]

    for j in range(1, n):
        for d in range(dim):
            # C0 continuity
            A_red[:, n_first + (j - 1) * 2 * dim + d - dim] = (
                A[:, j * n_first + d - dim] + A[:, j * n_first + d]
            )

            # C1 continuity
            A_red[:, n_first + (j - 1) * 2 * dim + d - dim] += (
                2 * A[:, j * n_first + d + dim]
            )
            A_red[:, n_first + (j - 1) * 2 * dim + d - 2 * dim] = (
                A[:, j * n_first + d - 2 * dim] - A[:, j * n_first + d + dim]
            )

            # Third and fourth point
            A_red[:, n_first + (j - 1) * 2 * dim + d] = A[:, j * n_first + d + 2 * dim]
            A_red[:, n_first + (j - 1) * 2 * dim + d + dim] = A[
                :, j * n_first + d + 3 * dim
            ]

    return A_red


def eval_cubic(n, points, num_per_segment=50):
    # knot vector
    knot_vector = np.array([i / (n) for i in range(n + 1)])

    # build Bernstein basis polynomials on given knot vector intervals
    basis = []
    for j in range(n):
        interval = knot_vector[[j, j + 1]]
        basis.append(BernsteinBasis(3, interval))

    # evaluation points
    num = n * num_per_segment
    xis = np.linspace(0, 1, num=num)

    # reshape point such that every curve segment has four points
    dim = points.shape[1]
    points_segments = points.reshape(-1, 4, dim)

    # evaluate C1-continous composite cubic Bezier curve
    c = np.zeros((num, dim), dtype=float)
    for i, xi in enumerate(xis):
        for j in range(n):
            interval = knot_vector[[j, j + 1]]
            # Note: For simplicity we compite c[i] twice if it is exaclty on
            # the interval boundaries.
            if (interval[0] <= xi) and (xi <= interval[1]):
                c[i] = basis[j](xi) @ points_segments[j]

    return c


def unique_points_C0_continous(points):
    dim = points.shape[1]

    other_points = points[4:]
    np_other = len(other_points)
    n_other = np_other // 4
    n = n_other + 1

    unique_points = np.zeros((4 + 3 * n_other, dim), dtype=float)
    unique_points[:4] = points[:4]
    for j in range(1, n):
        unique_points[4 + (j - 1) * 3] = points[j * 4 + 1]
        unique_points[4 + (j - 1) * 3 + 1] = points[j * 4 + 2]
        unique_points[4 + (j - 1) * 3 + 2] = points[j * 4 + 3]

    return unique_points


def unique_points_C1_continous(points):
    dim = points.shape[1]

    other_points = points[4:]
    np_other = len(other_points)
    n_other = np_other // 4
    n = n_other + 1

    unique_points = np.zeros((4 + 2 * n_other, dim), dtype=float)
    unique_points[:4] = points[:4]
    for j in range(1, n):
        unique_points[4 + (j - 1) * 2] = points[j * 4 + 2]
        unique_points[4 + (j - 1) * 2 + 1] = points[j * 4 + 3]

    return unique_points


def split_vtk(points):
    """
    References:
    -----------

    VTKa: https://blog.kitware.com/modeling-arbitrary-order-lagrange-finite-elements-in-the-visualization-toolkit/ \\
    VTKb: https://coreform.com/papers/implementation-of-rational-bezier-cells-into-VTK-report.pdf
    """
    # extract dimensions
    dim = points.shape[1]

    # reshape to segents
    points_segents = points.reshape(-1, 4, dim)
    n = len(points_segents)

    vtk_points = np.zeros_like(points_segents)
    for i in range(n):
        # VTK ordering, see https://coreform.com/papers/implementation-of-rational-bezier-cells-into-VTK-report.pdf:
        # 1. vertices (corners)
        # 2. edges
        vtk_points[i] = np.array(
            [
                points_segents[i, 0],
                points_segents[i, 3],
                points_segents[i, 1],
                points_segents[i, 2],
            ],
            dtype=float,
        )

    return vtk_points


def test_C0_continous_composite_cubic_line():
    # number of Bezier segments
    n = 2

    # unique control points
    unique_points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.5],
            [2.0, 0.5],
            [2.5, 0.0],
            [4.0, 0.0],
            [4.0, 1.0],
            [3.0, 1.0],
        ]
    )

    # knot vector
    knot_vector = np.array([i / (n) for i in range(n + 1)])
    print(f"knot_vector: {knot_vector}")

    # create C0-continous control points
    points = C0_continous_control_points(unique_points)

    # evaluate cubic Bezier curves
    c = eval_cubic(n, points, 100)

    # visualize unique, redundant points and spline
    fig, ax = plt.subplots()
    ax.plot(*unique_points.T, "-ob", label="unique points")
    ax.plot(*points.T, "--xr", label="redundant points")
    ax.plot(*c.T, "-k", label="C0-continous composite cubic Bézier spline")
    ax.grid()
    ax.axis("equal")
    ax.legend()
    plt.show()


def test_C1_continous_composite_cubic_line():
    # # number of Bezier segments
    # n = 2

    # # unique control points
    # unique_points = np.array(
    #     [
    #         [0.0, 0.0],
    #         [1.0, 0.5],
    #         [2.0, 0.5],
    #         [2.5, 0.0],
    #         [4.0, 0.0],
    #         [4.0, 1.0],
    #     ]
    # )

    # number of Bezier segments
    n = 3

    # unique control points
    unique_points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.5],
            [2.0, 0.5],
            [2.5, 0.0],
            [4.0, 0.0],
            [4.0, 1.0],
            [3.0, 1.0],
            [2.0, 1.5],
        ]
    )

    # knot vector
    knot_vector = np.array([i / (n) for i in range(n + 1)])
    print(f"knot_vector: {knot_vector}")

    # create C1-continous control points
    points = C1_continous_control_points(unique_points)

    # evaluate cubic Bezier curves
    c = eval_cubic(n, points, 100)

    # visualize unique, redundant points and spline
    fig, ax = plt.subplots()
    ax.plot(*unique_points.T, "-ob", label="unique points")
    ax.plot(*points.T, "--xr", label="redundant points")
    ax.plot(*c.T, "-k", label="C1-continous composite cubic Bézier spline")
    ax.grid()
    ax.axis("equal")
    ax.legend()
    plt.show()


def test_C2_continous_composite_cubic_line():
    # number of Bezier segments
    n = 2

    # unique control points
    unique_points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.5],
            [2.0, 0.5],
            [2.5, 0.0],
            [4.0, 1.0],
        ]
    )

    # number of Bezier segments
    n = 3

    # unique control points
    unique_points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.5],
            [2.0, 0.5],
            [2.5, 0.0],
            [4.0, 1.0],
            [4.0, 2.0],
        ]
    )

    # knot vector
    knot_vector = np.array([i / (n) for i in range(n + 1)])
    print(f"knot_vector: {knot_vector}")

    # create C2-continous control points
    points = C2_continous_control_points(unique_points)

    # evaluate cubic Bezier curves
    c = eval_cubic(n, points, 100)

    # visualize unique, redundant points and spline
    fig, ax = plt.subplots()
    ax.plot(*unique_points.T, "-ob", label="unique points")
    ax.plot(*points.T, "--xr", label="redundant points")
    ax.plot(*c.T, "-k", label="C1-continous composite cubic Bézier spline")
    ax.grid()
    ax.axis("equal")
    ax.legend()
    plt.show()


def L2_projection_Bezier_curve(target_points, n, case="C1", cDOF=[0, -1]):
    # extract dimension of points
    dim = target_points.shape[1]

    # number of points of the target curve
    n_target_points = len(target_points)

    # number of unique points
    n_points = 4 * n

    # create a good initial guess by selecting the corresponding number of
    # points linearly spaced from the target points
    xis_target_points = np.linspace(0, 1, num=n_target_points)
    xis = np.linspace(0, 1, num=n_points)
    idx_target_points = (np.abs(xis[:, None] - xis_target_points)).argmin(axis=1)
    initial_guess = target_points[idx_target_points]

    # remove redundant points of Bezier patches depending on chosen continuity
    if case == "C-1":
        unique_initial_guess = initial_guess
    elif case == "C0":
        unique_initial_guess = unique_points_C0_continous(initial_guess)
    elif case == "C1":
        unique_initial_guess = unique_points_C1_continous(initial_guess)
    else:
        raise NotImplementedError

    # define constraint points
    zDOF = np.arange(unique_initial_guess.shape[0])
    # TODO: Is there a nicer solution to get index "-1" working?
    cDOF = np.asarray(cDOF, dtype=int)
    cDOF = zDOF[cDOF]
    fDOF = np.setdiff1d(zDOF, cDOF)

    # linearize unconstrainte points
    z0 = unique_initial_guess.reshape(-1)  # incl. constraint points
    x0 = unique_initial_guess[fDOF].reshape(-1)

    # knot vector
    knot_vector = np.array([i / (n) for i in range(n + 1)])

    # build Bernstein basis polynomials on given knot vector intervals
    basis = []
    for j in range(n):
        interval = knot_vector[[j, j + 1]]
        basis.append(BernsteinBasis(3, interval))

    from scipy.optimize import minimize, least_squares

    def residual(x):
        # reshape vector to list of points
        unknown_points = x.reshape(-1, dim)

        # set constraint points
        unique_points = unique_initial_guess.copy()
        unique_points[fDOF] = unknown_points

        # extend redundant points depending on chosen continuity
        if case == "C-1":
            points = unique_points
        elif case == "C0":
            points = C0_continous_control_points(unique_points)
        elif case == "C1":
            points = C1_continous_control_points(unique_points)

        # reshape point such that every curve segment has four points
        points_segments = points.reshape(-1, 4, dim)

        # evaluate C1-continous composite cubic Bezier curve
        R = np.zeros_like(target_points)
        for i, xi in enumerate(xis_target_points):
            for j in range(n):
                interval = knot_vector[[j, j + 1]]

                # TODO: rewrite problem such that first and last point are
                # interpolated, see Piegl1997 (9.6.3), p. 411
                pi = target_points[i]

                # Note: For simplicity we compute c[i] twice if it is exaclty on
                # the interval boundaries.
                if (interval[0] <= xi) and (xi <= interval[1]):
                    ci = basis[j](xi) @ points_segments[j]
                    R[i] = pi - ci

        return R.reshape(-1)

    def fun(x):
        R = residual(x).reshape(-1, dim)
        K = sum([Ri @ Ri for Ri in R])
        return K

    def solve_L2(z0):
        # compute unique points depending on required continuity
        unique_points = z0.reshape(-1, dim)
        if case == "C-1":
            points = unique_points
        elif case == "C0":
            points = C0_continous_control_points(unique_points)
        elif case == "C1":
            points = C1_continous_control_points(unique_points)

        # number of unknowns
        N = points.size
        N_up = unique_points.size

        # quadratic shape function matrix and rhs
        A = np.zeros((N, N), dtype=float)
        b = np.zeros(N, dtype=float)

        # elDOF matrix
        elDOF = np.zeros((n, 4 * dim), dtype=int)
        elDOF_el = np.arange(4 * dim)
        for el in range(n):
            elDOF[el] = elDOF_el + el * 4 * dim

        # nodalDOF = np.arange(N).reshape(n * 4, dim)
        nodalDOF = elDOF_el.reshape(-1, dim)

        for k, xi_k in enumerate(xis_target_points):
            for i in range(n):
                interval = knot_vector[[i, i + 1]]
                if (interval[0] <= xi_k) and (xi_k <= interval[1]):
                    elDOF_i = elDOF[i]
                    pi = target_points[k]
                    basis_k = basis[i](xi_k)

                    p1 = len(basis_k)
                    for p in range(p1):
                        elDOF_p = elDOF_i[nodalDOF[p]]
                        b[elDOF_p] += pi * basis_k[p]
                        for q in range(p1):
                            elDOF_q = elDOF_i[nodalDOF[q]]
                            A[elDOF_p[:, None], elDOF_q] += (
                                np.eye(dim, dtype=float) * basis_k[p] * basis_k[q]
                            )

        # compute rhs contributions of boundary terms and assemble constraint
        # degrees of freedom
        cDOF1 = np.arange(0, dim)
        b -= points[0].T @ A[cDOF1]
        cDOF2 = np.arange(-dim, 0)
        b -= points[-1].T @ A[cDOF2]

        if case == "C-1":
            A_red = A
        elif case == "C1":
            A_red = reduced_matrix_C1_continous(A, dim)
        else:
            A_red = reduced_matrix_C0_continous(A, dim)

        # remove boundary equations from the system
        cDOF = np.concatenate((cDOF1, cDOF2))
        qDOF = np.arange(N)
        qDOF_up = np.arange(N_up)
        fDOF = np.setdiff1d(qDOF, qDOF[cDOF])
        fDOF_up = np.setdiff1d(qDOF_up, qDOF_up[cDOF])

        # solve least square problem with eliminated first and last node
        # from scipy.sparse.linalg import spsolve
        from scipy.sparse.linalg import lsqr

        # unique_points = np.zeros(N_up, dtype=float)
        # unique_points[fDOF] = spsolve(A.tocsc()[fDOF[:, None], fDOF_up], b[fDOF])
        # unkown_points,*_ = lsqr(A_red.tocsc()[fDOF[:, None], fDOF_up], b[fDOF])
        unkown_points, *_ = lsqr(A_red[fDOF[:, None], fDOF_up], b[fDOF])

        # set first and last node to given values
        # unique_points[cDOF1] = points[0]
        # unique_points[cDOF2] = points[-1]

        # from scipy.linalg import lstsq
        # # unique_points = lstsq(A.toarray(), b)[0]
        # unique_points[fDOF] = lstsq(A.toarray()[fDOF[:, None], fDOF], b[fDOF])

        # # TODO: We have to skipp some DOF's since we want to enforce continuity.
        # # This can be done by removing the respective rows and columns from the the system
        # qDOF = np.arange(N)
        # if case == "C-1":
        #     cDOF = np.array([0, 1, 2, -3, -2, -1], dtype=int)
        # else:
        #     raise NotImplementedError

        # fDOF = np.setdiff1d(qDOF, cDOF)

        # unique_points = np.zeros(N, dtype=float)
        # unique_points[:dim] = points[0]
        # unique_points[-dim:] = points[-1]
        # unique_points[fDOF] = x

        from scipy.optimize import OptimizeResult

        return OptimizeResult(x=unkown_points, success=True)

    # # Note: Since the Jacobian is constant we only compute it once per projection step.
    # from cardillo.math import approx_fprime
    # J = approx_fprime(x0, residual, method="2-point", eps=1e-6)
    # jac = lambda x: J

    # solve optimization problem
    # sol = least_squares(residual, x0, method="lm")
    # sol = least_squares(residual, x0, jac=jac)
    # sol = minimize(fun, x0, method="SLSQP")
    sol = solve_L2(z0)
    success = sol.success
    assert success
    x = sol.x

    # reshape solution to list of points
    unknown_points = x.reshape(-1, dim)

    # set constraint points
    unique_points = unique_initial_guess.copy()
    unique_points[fDOF] = unknown_points

    if case == "C-1":
        points = unique_points
    elif case == "C0":
        points = C0_continous_control_points(unique_points)
    elif case == "C1":
        points = C1_continous_control_points(unique_points)

    # reshape point such that every curve segment has four points
    points_segments = points.reshape(-1, 4, dim)

    return unique_points, points, points_segments


def line2vtk(points_segments):
    points_segments = np.atleast_2d(points_segments)
    vtk_points = []
    for i, points_i in enumerate(points_segments):
        # VTK ordering, see https://coreform.com/papers/implementation-of-rational-bezier-cells-into-VTK-report.pdf:
        # 1. vertices (corners)
        # 2. edges
        vtk_points.append(points_i[0])
        vtk_points.append(points_i[-1])
        vtk_points.extend(points_i[1:-1])

    return vtk_points


# def fit_Bezier(case="C-1"):
# def fit_Bezier(case="C0"):
def fit_Bezier(case="C1"):
    num = 200
    xis = np.linspace(0, 1, num=num)

    def curve(xi):
        return np.array(
            [
                np.sin(xi * np.pi),
                np.cos(xi * np.pi),
            ]
        ).T

    # def curve(xi):
    #     return np.array(
    #         [
    #             xi,
    #             xi * np.sin(xi * 4 * np.pi),
    #         ]
    #     ).T

    # def curve(xi):
    #     return np.array([
    #         xi,
    #         xi * np.sin(1 / ((xi + 1.0e-3) / (1.0 - 1.0e-3) * np.pi)),
    #     ]).T

    target_points = curve(xis)

    # number of segments
    n = 3

    unique_points, points, points_segments = L2_projection_Bezier_curve(
        target_points, n, case
    )

    c = eval_cubic(n, points, 500)

    fig, ax = plt.subplots()
    ax.plot(*target_points.T, "-k", label="target curve")
    ax.plot(*unique_points.T, "-ob", label="unique points")
    ax.plot(*points.T, "--xr", label="redundant points")
    if case == "C-1":
        ax.plot(*c.T, "--g", label="C^{-1}-continous composite cubic Bézier spline")
    elif case == "C0":
        ax.plot(*c.T, "--g", label="C-0-continous composite cubic Bézier spline")
    elif case == "C1":
        ax.plot(*c.T, "--g", label="C1-continous composite cubic Bézier spline")
    ax.grid()
    ax.axis("equal")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    # test_line()
    # test_surface()
    # test_volume()
    # test_circle()
    # test_quintic_nonrational_circle()
    # test_C0_continous_composite_cubic_line()
    # test_C1_continous_composite_cubic_line()
    # test_C2_continous_composite_cubic_line()
    fit_Bezier()
