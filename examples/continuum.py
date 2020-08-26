from threading import main_thread
import numpy as np
import matplotlib.pyplot as plt

from cardillo.discretization.mesh import Mesh, cube
from cardillo.discretization.B_spline import Knot_vector, fit_B_spline_volume
from cardillo.discretization.indexing import flat3D
from cardillo.model.continuum import Ogden1997_compressible, First_gradient
from cardillo.solver import Newton
from cardillo.model import Model
from cardillo.math.algebra import A_IK_basic_z

if __name__ == "__main__":
    pass

def test_cylidner():    
    # build mesh
    degrees = (3, 3, 3)
    QP_shape = (3, 3, 3)
    element_shape = (3, 3, 5)

    Xi = Knot_vector(degrees[0], element_shape[0])
    Eta = Knot_vector(degrees[1], element_shape[1])
    Zeta = Knot_vector(degrees[2], element_shape[2])
    knot_vectors = (Xi, Eta, Zeta)
    
    mesh = Mesh(knot_vectors, QP_shape, derivative_order=1, basis='B-spline', nq_n=3)
    
    def cylinder(xi, eta, zeta, R=1, H=3):
        xi_ = 2 * xi - 1
        eta_ = 2 * eta - 1

        if np.abs(xi_) > np.abs(eta_):
            r = np.sqrt(1 + eta_**2)
        else:
            r = np.sqrt(1 + xi_**2)

        x = R / r * xi_
        y = R / r * eta_
        z = zeta * H
        return x, y, z

    nxi, neta, nzeta = 10, 10, 10
    xi = np.linspace(0, 1, num=nxi)
    eta = np.linspace(0, 1, num=neta)
    zeta = np.linspace(0, 1, num=nzeta)
    
    n3 = nxi * neta * nzeta
    knots = np.zeros((n3, 3))
    Pw = np.zeros((n3, 3))
    for i, xii in enumerate(xi):
        for j, etai in enumerate(eta):
            for k, zetai in enumerate(zeta):
                idx = flat3D(i, j, k, (nxi, neta, nzeta))
                knots[idx] = xii, etai, zetai
                Pw[idx] = cylinder(xii, etai, zetai)

    cDOF_ = np.array([], dtype=int)
    qc = np.array([], dtype=float).reshape((0, 3))
    X, Y, Z_ = fit_B_spline_volume(mesh, knots, Pw, qc, cDOF_)
    Z = np.concatenate((X, Y, Z_))

    # material model    
    mu1 = 0.3
    mu2 = 0.5
    mat = Ogden1997_compressible(mu1, mu2)

    # # 3D continuum
    # # cDOF = []
    # # b = lambda t: np.array([], dtype=float)
    # cDOF1 = mesh.surface_DOF[0].reshape(-1)
    # cDOF2 = mesh.surface_DOF[1][2]
    # cDOF = np.concatenate((cDOF1, cDOF2))
    # b1 = lambda t: Z[cDOF1]
    # b2 = lambda t: Z[cDOF2] + t * 0.5
    # b = lambda t: np.concatenate((b1(t), b2(t)))

    cDOF1 = mesh.surface_DOF[0].reshape(-1)
    # cDOF2 = mesh.surface_DOF[1][2]
    # cDOF2x, cDOF2y, cDOF2z = mesh.surface_DOF[1]
    # cDOF2 = mesh.surface_DOF[1].reshape(-1, order='F')
    cDOF2 = mesh.surface_DOF[1].reshape(-1)
    cDOF = np.concatenate((cDOF1, cDOF2))
    b1 = lambda t: Z[cDOF1]
    # b2 = lambda t: Z[cDOF2] + t * 0.5

    def b2(t, phi0=2*np.pi, h=0.5):
        cDOF2_xyz = cDOF2.reshape(3, -1).T
        out = np.zeros_like(Z)

        phi = t * phi0
        R = A_IK_basic_z(phi)

        th = t * np.array([0, 0, h])
        for DOF in cDOF2_xyz:
            out[DOF] = R @ Z[DOF] + th
        
        return out[cDOF2]

    b = lambda t: np.concatenate((b1(t), b2(t)))

    continuum = First_gradient(mat, mesh, Z, z0=Z, cDOF=cDOF, b=b)

    # build model
    model = Model()
    model.add(continuum)
    model.assemble()

    # static solver
    n_load_steps = 10
    tol = 1.0e-5
    max_iter = 10
    solver = Newton(model, n_load_steps=n_load_steps, tol=tol, max_iter=max_iter)
    
    import cProfile, pstats
    pr = cProfile.Profile()
    pr.enable()
    sol = solver.solve()
    pr.disable()

    sortby = 'cumulative'
    ps = pstats.Stats(pr).sort_stats(sortby)
    ps.print_stats(0.1) # print only first 5% of the list

    # vtk export
    for i, (ti, qi) in enumerate(zip(sol.t, sol.q)):
        continuum.post_processing(ti, qi, f'continuum{i}.vtu')

def test_cube():
    # build mesh
    degrees = (2, 2, 2)
    QP_shape = (2, 2, 2)
    # element_shape = (5, 5, 5)
    element_shape = (3, 3, 3)

    Xi = Knot_vector(degrees[0], element_shape[0])
    Eta = Knot_vector(degrees[1], element_shape[1])
    Zeta = Knot_vector(degrees[2], element_shape[2])
    knot_vectors = (Xi, Eta, Zeta)
    
    mesh = Mesh(knot_vectors, QP_shape, derivative_order=1, basis='B-spline', nq_n=3)

    # reference configuration is a cube
    L = 1
    B = 1
    H = 1
    cube_shape = (L, B, H)
    Z = cube(cube_shape, mesh, Greville=True)

    # material model    
    mu1 = 0.3
    mu2 = 0.5
    mat = Ogden1997_compressible(mu1, mu2)

    # 3D continuum
    # cDOF = []
    # b = lambda t: np.array([], dtype=float)
    cDOF1 = mesh.surface_DOF[0].reshape(-1)
    cDOF2 = mesh.surface_DOF[1][2]
    cDOF = np.concatenate((cDOF1, cDOF2))
    b1 = lambda t: Z[cDOF1]
    b2 = lambda t: Z[cDOF2] + t * 0.5
    b = lambda t: np.concatenate((b1(t), b2(t)))

    continuum = First_gradient(mat, mesh, Z, z0=Z, cDOF=cDOF, b=b)

    # build model
    model = Model()
    model.add(continuum)
    model.assemble()

    # static solver
    n_load_steps = 10
    tol = 1.0e-5
    max_iter = 10
    solver = Newton(model, n_load_steps=n_load_steps, tol=tol, max_iter=max_iter)
    
    import cProfile, pstats
    pr = cProfile.Profile()
    pr.enable()
    sol = solver.solve()
    pr.disable()

    sortby = 'cumulative'
    ps = pstats.Stats(pr).sort_stats(sortby)
    ps.print_stats(0.05) # print only first 5% of the list

    # vtk export
    for i, (ti, qi) in enumerate(zip(sol.t, sol.q)):
        continuum.post_processing(ti, qi, f'continuum{i}.vtu')

if __name__ == "__main__":
    # test_cube()
    test_cylidner()