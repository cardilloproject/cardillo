from cardillo.math.numerical_derivative import Numerical_derivative
import numpy as np
from cardillo.discretization.indexing import flat2D, flat3D
from cardillo.discretization.B_spline import B_spline_basis3D
from cardillo.math.algebra import inverse3D, determinant3D

# TODO: import global meshio
import meshio as meshio

class First_gradient(object):
    def __init__(self, material, mesh, Q, q0=None):
        self.mat = material

        # store generalized coordinates
        self.Q = Q
        self.q0 = q0 if q0 is not None else Q.copy()
        self.nq = len(Q)
        self.nu = self.nq

        # store mesh and extrac data
        self.mesh = mesh

        self.elDOF = mesh.elDOF
        self.nodalDOF = mesh.nodalDOF
        self.nel = mesh.nel
        self.nn = mesh.nn
        self.nn_el = mesh.nn_el # number of nodes of an element
        self.nq_el = mesh.nq_el
        self.nqp = mesh.nqp

        self.dim = int(len(Q) / self.nn)
        if self.dim == 2:
            self.flat = flat2D
        else:
            self.flat = flat3D

        # for each Gauss point, compute kappa0^-1, N_X and w_J0 = w * det(kappa0^-1)
        self.kappa0_xi_inv, self.N_X, self.w_J0 = self.mesh.reference_mappings(Q)
        
    def post_processing(self, q, filename, binary=False):

        # generalized coordinates, connectivity and polynomial degree
        cells, points, HigherOrderDegrees = self.mesh.vtk_mesh(q)

        # dictionary storing point data
        point_data = {}
        
        # evaluate deformation gradient at quadrature points
        F = np.zeros((self.mesh.nel, self.mesh.nqp, self.mesh.nq_n, self.mesh.nq_n))
        for el in range(self.mesh.nel):
            qe = q[self.mesh.elDOF[el]]
            for i in range(self.mesh.nqp):
                for a in range(self.mesh.nn_el):
                    F[el, i] += np.outer(qe[self.mesh.nodalDOF[a]], self.N_X[el, i, a]) # Bonet 1997 (7.6b)

        F_vtk = self.mesh.field_to_vtk(F)
        point_data.update({"F": F_vtk})

        # field data vtk export
        point_data_fields = {
            "C": lambda F: F.T @ F,
            "J": lambda F: np.array([determinant3D(F)]),
            "P": lambda F: self.mat.P(F),
            "S": lambda F: self.mat.S(F),
        }

        for name, fun in point_data_fields.items():
            tmp = fun(F_vtk[0].reshape(self.dim, self.dim)).reshape(-1)
            field = np.zeros((len(F_vtk), len(tmp)))
            for i, Fi in enumerate(F_vtk):
                field[i] = fun(Fi.reshape(self.dim, self.dim)).reshape(-1)
            point_data.update({name: field})
     
        # write vtk mesh using meshio
        meshio.write_points_cells(
            filename,
            points,
            cells,
            point_data=point_data,
            cell_data={"HigherOrderDegrees": HigherOrderDegrees},
            binary=binary
        )

    def F(self, knots, q):
        n = len(knots)
        F = np.zeros((n, self.dim, self.dim))

        for i, (xi, eta, zeta) in enumerate(knots):
            el_xi = self.mesh.Xi.element_number(xi)[0]
            el_eta = self.mesh.Eta.element_number(eta)[0]
            el_zeta = self.mesh.Zeta.element_number(zeta)[0]
            el = flat3D(el_xi, el_eta, el_zeta, self.mesh.nel_per_dim)
            elDOF = self.elDOF[el]
            qe = q[elDOF]
            Qe = self.Q[elDOF]

            # evaluate shape functions
            NN = B_spline_basis3D(self.mesh.degrees, 1, self.mesh.knot_vectors, (xi, eta, zeta))
            N_xi = NN[0, :, 1:4]

            # reference mapping gradient w.r.t. parameter space
            kappa0_xi = np.zeros((self.mesh.nq_n, self.mesh.nq_n))
            for a in range(self.mesh.nn_el):
                kappa0_xi += np.outer(Qe[self.mesh.nodalDOF[a]], N_xi[a]) # Bonet 1997 (7.6b)
            
            for a in range(self.mesh.nn_el):
                self.N_X[el, i, a] = N_xi[a] @ inverse3D(kappa0_xi) # Bonet 1997 (7.6a) modified

            for a in range(self.mesh.nn_el):
                F[i] += np.outer(qe[self.mesh.nodalDOF[a]], self.N_X[el, i, a]) # Bonet 1997 (7.5)

        return F

    def f_pot_el(self, qe, el):
        f = np.zeros(self.nq_el)

        for i in range(self.nqp):
            N_X = self.N_X[el, i]
            w_J0 = self.w_J0[el, i]

            # deformation gradient
            F = np.zeros((self.dim, self.dim))
            for a in range(self.nn_el):
                F += np.outer(qe[self.nodalDOF[a]], N_X[a]) # Bonet 1997 (7.5)

            # first Piola-Kirchhoff deformation tensor
            P = self.mat.P(F)

            # internal forces
            for a in range(self.nn_el):
                # TODO: reference to Bonet1997?
                # Bonet1997 (2.52b)
                f[self.nodalDOF[a]] += -P @ N_X[a] * w_J0

        return f

    def f_pot(self, t, q):
        f_pot = np.zeros(self.nq)
        for el in range(self.nel):
            f_pot[self.elDOF[el]] += self.f_pot_el(q[self.elDOF[el]], el)
        return f_pot

    def f_pot_q(self, t, q, coo):
        for el in range(self.nEl):
            elDOF = self.elDOF[el]
            Ke = Numerical_derivative(lambda t, q: self.f_pot_el(q, el))._x(t, q[elDOF])

            # sparse assemble element internal stiffness matrix
            coo.extend(Ke, (self.uDOF[elDOF], self.qDOF[elDOF]))

def test_gradient():
    from cardillo.discretization.mesh import Mesh, cube
    from cardillo.discretization.B_spline import Knot_vector, fit_B_spline_volume, B_spline_volume2vtk
    from cardillo.discretization.indexing import flat3D

    QP_shape = (1, 1, 1)
    degrees = (3, 3, 1)
    element_shape = (10, 10, 1)

    Xi = Knot_vector(degrees[0], element_shape[0])
    Eta = Knot_vector(degrees[1], element_shape[1])
    Zeta = Knot_vector(degrees[2], element_shape[2])
    knot_vectors = (Xi, Eta, Zeta)
    
    mesh = Mesh(knot_vectors, QP_shape, derivative_order=1, basis='B-spline', nq_n=3)

    def bending(xi, eta, zeta, phi0=np.pi/2, R=1, B=1, H=1):
        phi = (1 - xi) * phi0
        x = (R + B * eta) * np.cos(phi)
        y = (R + B * eta) * np.sin(phi)
        z = zeta * H
        return x, y, z

    nxi, neta, nzeta = 15, 15, 5
    xi = np.linspace(0, 1, num=nxi)
    eta = np.linspace(0, 1, num=neta)
    zeta = np.linspace(0, 1, num=nzeta)

    phi0, R, B, H = np.pi / 2, 1, 1, 1
    
    n3 = nxi * neta * nzeta
    knots = np.zeros((n3, 3))
    Pw = np.zeros((n3, 3))
    for i, xii in enumerate(xi):
        for j, etai in enumerate(eta):
            for k, zetai in enumerate(zeta):
                idx = flat3D(i, j, k, (nxi, neta, nzeta))
                knots[idx] = xii, etai, zetai
                Pw[idx] = bending(xii, etai, zetai, phi0=phi0, R=R, B=B, H=H)

    cDOF = np.array([], dtype=int)
    qc = np.array([], dtype=float).reshape((0, 3))
    x, y, z = fit_B_spline_volume(mesh, knots, Pw, qc, cDOF)

    L = 1
    cube_shape = (L, B, H)
    Q = cube(cube_shape, mesh, Greville=True)
    q0 = np.concatenate((x, y, z))
    continuum = First_gradient(None, mesh, Q, q0=q0)
    
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # ax.scatter(*Pw.T, color='black')
    # ax.scatter(*Q.reshape(3, -1), color='blue')
    # ax.scatter(*q0.reshape(3, -1), color='red')
    # plt.show()

    # knots = np.array([[0.5, 0.5, 0.5]])
    # knots = np.array([[0.125, 0.35, 0.85]])
    knots = np.random.rand(3).reshape(1, 3)
    F_num = continuum.F(knots, q0)
    print(f'F_num({knots[0]}):\n{F_num[0]}')

    def F(xi, eta, zeta, phi0, R, B, L):
        r = phi0 * (R + B * eta) / L
        phi = (1 - xi) * phi0
        F = np.array([
            [r * np.sin(phi),  np.cos(phi), 0],
            [-r * np.cos(phi), np.sin(phi), 0],
            [0,                          0, 1],
        ])
        return F

    F_an = F(*knots[0], phi0, R, B, L)
    print(f'F({knots[0]}):\n{F_an}')

    error = np.linalg.norm(F_num[0] - F_an)
    print(f'error: {error}')
    
def test_gradient_vtk_export():
    from cardillo.discretization.mesh import Mesh, cube
    from cardillo.discretization.B_spline import Knot_vector, fit_B_spline_volume
    from cardillo.discretization.indexing import flat3D

    QP_shape = (5, 5, 5)
    degrees = (3, 3, 1)
    element_shape = (5, 5, 1)

    Xi = Knot_vector(degrees[0], element_shape[0])
    Eta = Knot_vector(degrees[1], element_shape[1])
    Zeta = Knot_vector(degrees[2], element_shape[2])
    knot_vectors = (Xi, Eta, Zeta)
    
    mesh = Mesh(knot_vectors, QP_shape, derivative_order=1, basis='B-spline', nq_n=3)

    # reference configuration is a cube
    phi0 = np.pi / 2
    R = 1
    B = 1
    H = 1
    # L = (R + B / 2) * phi0
    L = (R) * phi0
    cube_shape = (L, B, H)
    Q = cube(cube_shape, mesh, Greville=True)

    # 3D continuum
    continuum = First_gradient(None, mesh, Q, q0=Q)

    # fit quater circle configuration
    def bending(xi, eta, zeta, phi0, R, B, H):
        phi = (1 - xi) * phi0
        x = (R + B * eta) * np.cos(phi)
        y = (R + B * eta) * np.sin(phi)
        z = zeta * H
        return x, y, z

    nxi, neta, nzeta = 15, 15, 5
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
                Pw[idx] = bending(xii, etai, zetai, phi0=phi0, R=R, B=B, H=H)
    
    cDOF = np.array([], dtype=int)
    qc = np.array([], dtype=float).reshape((0, 3))
    x, y, z = fit_B_spline_volume(mesh, knots, Pw, qc, cDOF)
    q = np.concatenate((x, y, z))

    # export current configuration and deformation gradient on quadrature points to paraview
    continuum.post_processing(q, 'test.vtu')

def test_internal_forces():
    from cardillo.discretization.mesh import Mesh, cube
    from cardillo.discretization.B_spline import Knot_vector, fit_B_spline_volume
    from cardillo.discretization.indexing import flat3D
    from cardillo.model.continuum import Ogden1997_compressible

    QP_shape = (5, 5, 5)
    degrees = (3, 3, 1)
    element_shape = (5, 5, 1)

    Xi = Knot_vector(degrees[0], element_shape[0])
    Eta = Knot_vector(degrees[1], element_shape[1])
    Zeta = Knot_vector(degrees[2], element_shape[2])
    knot_vectors = (Xi, Eta, Zeta)
    
    mesh = Mesh(knot_vectors, QP_shape, derivative_order=1, basis='B-spline', nq_n=3)

    # reference configuration is a cube
    phi0 = np.pi / 2
    R = 1
    B = 1
    H = 1
    # L = (R + B / 2) * phi0
    L = (R) * phi0
    cube_shape = (L, B, H)
    Q = cube(cube_shape, mesh, Greville=True)

    # material model    
    mu1 = 0.3
    mu2 = 0.5
    mat = Ogden1997_compressible(mu1, mu2)

    # 3D continuum
    continuum = First_gradient(mat, mesh, Q, q0=Q)

    # fit quater circle configuration
    def bending(xi, eta, zeta, phi0, R, B, H):
        phi = (1 - xi) * phi0
        x = (R + B * eta) * np.cos(phi)
        y = (R + B * eta) * np.sin(phi)
        z = zeta * H
        return x, y, z

    nxi, neta, nzeta = 15, 15, 5
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
                Pw[idx] = bending(xii, etai, zetai, phi0=phi0, R=R, B=B, H=H)
    
    cDOF = np.array([], dtype=int)
    qc = np.array([], dtype=float).reshape((0, 3))
    x, y, z = fit_B_spline_volume(mesh, knots, Pw, qc, cDOF)
    q = np.concatenate((x, y, z))

    # evaluate internal forces in deformed configuration
    f_pot = continuum.f_pot(0, q)
    print(f'f_pot:\n{f_pot}')

    # export current configuration and deformation gradient on quadrature points to paraview
    continuum.post_processing(q, 'test.vtu')

if __name__ == "__main__":
    # test_gradient()
    # test_gradient_vtk_export()
    test_internal_forces()