import numpy as np
import matplotlib.pyplot as plt

from cardillo.discretization.mesh import Mesh, cube
from cardillo.discretization.B_spline import Knot_vector
from cardillo.model.continuum import Ogden1997_compressible, First_gradient
from cardillo.solver import Newton

if __name__ == "__main__":
    degrees = (2, 2, 2)
    QP_shape = (2, 2, 2)
    element_shape = (5, 5, 5)

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

    from cardillo.model import Model
    model = Model()
    model.add(continuum)
    model.assemble()
    
    # evaluate internal forces in reference configuration
    Q = Z[continuum.fDOF]

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

    # scatter q's
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    max_val = np.max(np.abs(sol.q[0]))
    ax.set_xlim3d(left=-max_val, right=max_val)
    ax.set_ylim3d(bottom=-max_val, top=max_val)
    ax.set_zlim3d(bottom=-max_val, top=max_val)

    t = sol.t
    q = sol.q
    # z0 = continuum.z(t[0], q[0])
    # z1 = continuum.z(t[-1], q[-1])

    # ax.scatter(*z0.reshape(3,-1), marker='o', color='blue')
    # ax.scatter(*z1.reshape(3,-1), marker='x', color='red')
    
    # plt.show()

    for i, (ti, qi) in enumerate(zip(t, q)):
        # z = continuum.z(ti, qi)
        continuum.post_processing(ti, qi, f'continuum{i}.vtu')