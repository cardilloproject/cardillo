import numpy as np
from numpy.core.multiarray import concatenate
from scipy.sparse.linalg import spsolve
from cardillo.discretization.indexing import flat2D, flat3D, split2D, split3D
from cardillo.discretization.B_spline import B_spline_basis3D, decompose_B_spline_volume, q_to_Pw_3D, flat3D_vtk
from cardillo.math.algebra import inverse3D, determinant3D, quat2mat, quat2mat_p
from cardillo.utility.coo import Coo

# TODO: import global meshio
import meshio as meshio

class First_gradient(object):
    def __init__(self, mesh, Q, q0=None):
    # def __init__(self, material, mesh):
    #     self.mat = material

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
        self.nn = mesh.nn # number of nodes of an element

        self.dim = int(len(Q) / self.nn)
        if self.dim == 2:
            self.flat = flat2D
        else:
            self.flat = flat3D

        # for each Gauss point, compute kappa0^-1, N_X and w_J0 = w * det(kappa0^-1)
        self.kappa0_xi_inv, self.N_X, self.w_J0 = self.mesh.reference_mappings(Q)
        
    def post_processing(self, q, filename, binary=False):
        # evaluate deformation gradient at quadrature points
        F = np.zeros((self.mesh.nel, self.mesh.nqp, self.mesh.nq_n, self.mesh.nq_n))
        for el in range(self.mesh.nel):
            qe = q[self.mesh.elDOF[el]]
            for i in range(self.mesh.nqp):
                for a in range(self.mesh.nn_el):
                    F[el, i] += np.outer(qe[self.mesh.nodalDOF[a]], self.N_X[el, i, a]) # Bonet 1997 (7.6b)

        # L2 projection of deformation gradient on nodes
        A = Coo((self.mesh.nn, self.mesh.nn))
        # B = np.zeros((self.mesh.nn, self.mesh.nel, self.mesh.nqp))
        b_F = np.zeros((self.mesh.nn, self.dim * self.dim))
        for el in range(self.mesh.nel):
            elDOF_el = self.elDOF[el, :self.mesh.nn_el]
            be_F = np.zeros((self.mesh.nn_el, self.dim * self.dim))
            Ae = np.zeros((self.mesh.nn_el, self.mesh.nn_el))
            Nel = self.mesh.N[el]
            for a in range(self.mesh.nn_el):
                for i in range(self.mesh.nqp):
                    be_F[a] += Nel[i, a] * F[el, i].reshape(-1)
                # B[elDOF_el[a], el] =  Nel[:, a]

                for b in range(self.mesh.nn_el):
                    for i in range(self.mesh.nqp):
                        Ae[a, b] += Nel[i, a] * Nel[i, b]
            b_F[elDOF_el] += be_F
            A.extend(Ae, (elDOF_el, elDOF_el))

        # b_F = np.einsum('ijk,jkl->il', B, F.reshape(self.mesh.nel, self.mesh.nqp, -1))

        A = A.tocsc()
        
        q_F = np.zeros((self.mesh.nn, self.dim * self.dim))
        for i, b in enumerate(b_F.T):
            q_F[:, i] = spsolve(A, b)

        # rearrange q's from solver to Piegl's 3D ordering
        knot_vectors = self.mesh.knot_vector_objs
        Pw = q_to_Pw_3D(knot_vectors, q)

        p = knot_vectors[0].degree
        q = knot_vectors[1].degree
        r = knot_vectors[2].degree
        degrees = (p, q, r)
        
        Qw = decompose_B_spline_volume(knot_vectors, Pw)
        nbezier_xi, nbezier_eta, nbezier_zeta, p1, q1, r1, dim = Qw.shape

        # build cells
        n_patches = nbezier_xi * nbezier_eta * nbezier_zeta
        patch_size = p1 * q1 * r1
        points = np.zeros((n_patches * patch_size, dim))
        cells = []
        HigherOrderDegree_patches = []
        for i in range(nbezier_xi):
            for j in range(nbezier_eta):
                for k in range(nbezier_zeta):
                    idx = flat3D(i, j, k, (nbezier_xi, nbezier_eta))
                    point_range = np.arange(idx * patch_size, (idx + 1) * patch_size)
                    points[point_range] = flat3D_vtk(Qw[i, j, k])
                    
                    cells.append( ("VTK_BEZIER_HEXAHEDRON", point_range[None]) )
                    HigherOrderDegree_patches.append( np.array(degrees)[None] )

        point_data_fields = {
            "F": (lambda F: F, self.mesh.nq_n * self.mesh.nq_n),
            "C": (lambda F: F.T @ F, self.mesh.nq_n * self.mesh.nq_n),
            "J": (lambda F: determinant3D(F), 1),
        }

        point_data = {}
        for name, (fun, dim) in point_data_fields.items():               
            # build rhs vector for L2 projection
            field_b = np.zeros((self.mesh.nn, dim))
            for el in range(self.mesh.nel):
                elDOF_el = self.elDOF[el, :self.mesh.nn_el]
                
                Nel = self.mesh.N[el]
                for i in range(self.mesh.nqp):
                    field_bei = fun(F[el, i]).reshape(-1)
                    for a in range(self.mesh.nn_el):
                        field_b[elDOF_el[a]] += Nel[i, a] * field_bei

            # solve L2 projection
            field_q = np.zeros((self.mesh.nn, dim))
            for i, b in enumerate(field_b.T):
                field_q[:, i] = spsolve(A, b)

            # Bezier decomposition of point data fields
            field_Pw = q_to_Pw_3D(knot_vectors, field_q.T.reshape(-1), dim=field_q.shape[1])
            field_Qw = decompose_B_spline_volume(knot_vectors, field_Pw)
            
            # rearrange Bezier cell coordinate in vtk ordering
            field_points = np.zeros((n_patches * patch_size, dim))
            for i in range(nbezier_xi):
                for j in range(nbezier_eta):
                    for k in range(nbezier_zeta):
                        idx = flat3D(i, j, k, (nbezier_xi, nbezier_eta))
                        point_range = np.arange(idx * patch_size, (idx + 1) * patch_size)
                        field_points[point_range] = flat3D_vtk(field_Qw[i, j, k])

            # update dictionary of point data fields
            point_data.update({name: field_points})
                        
        meshio.write_points_cells(
            filename,
            points,
            cells,
            point_data=point_data,
            cell_data={"HigherOrderDegrees": HigherOrderDegree_patches},
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

    # def internal_forces_el(self, qel, el):
    #     f = np.zeros(self.mesh.nodal_DOF * self.mesh.n)

    #     for i in range(self.mesh.nQP):

    #         # compute deformation gradient of the deformed placement w.r.t. reference placement
    #         F_gp = np.zeros((self.mesh.nodal_DOF, self.mesh.nodal_DOF))
    #         for a in range(self.nn):
    #             # TODO: reference
    #             qa = qel[self.mesh.ndDOF[a]]
    #             F_gp += np.outer(qa, self.nablaX_N[el,i,a])

    #         P_gp = self.mat.P(F_gp)

    #         for a in range(self.nn):
    #             # TODO: reference to Bonet1997
    #             f[self.ndDOF[a]] += P_gp @ self.nablaX_N[el,i,a] * self.w_J0[el, i]

    #     return -f

    # def internal_forces(self, q):
    #     f_int = np.zeros(self.nq)

    #     for el in range(self.nel):
    #         qel = q[self.elDOF[el]]
    #         f_int[self.elDOF[el]] += self.internal_forces_el(qel, el)

    #     return f_int


    # # def internal_forces_q(self, q):
    # #     eps = 1.0e-6
    # #     K_int = np.zeros((self.nq, self.nq))

    # #     for el in range(self.nel):
    # #         qel = q[self.elDOF[el]]
    # #         residual = lambda t, q: self.internal_forces_el(q, el)

    # #         # C^e^T C_a^T K C_a C^e += K^e
    # #         K_int[np.ix_(self.elDOF[el], self.elDOF[el])] += NumericalDerivative(residual, order=1).dR_dq(0, qel, eps=eps)

    # #     return K_int


    # def internal_forces_el_q(self, qel, el):
    #     K_el = np.zeros((self.mesh.nodal_DOF * self.mesh.n, self.mesh.nodal_DOF * self.mesh.n))

    #     for i in range(self.mesh.nQP):

    #         # compute deformation gradient of the deformed placement w.r.t. reference placement
    #         F_gp = np.zeros((self.mesh.nodal_DOF, self.mesh.nodal_DOF))
    #         dF_dq = np.zeros((self.ndim,self.ndim,self.mesh.nodal_DOF * self.mesh.n))
    #         for a in range(self.nn):
    #             # TODO: reference
    #             qa = qel[self.mesh.ndDOF[a]]
    #             F_gp += np.outer(qa, self.nablaX_N[el,i,a])
    #             dF_dq[:,:,self.mesh.ndDOF[a]] += np.einsum('ik,j->ijk', np.eye(self.ndim), self.nablaX_N[el,i,a])

    #         #dS_dC_gp = self.mat.dS_dC(F_gp.T @ F_gp)
    #         S_gp = self.mat.S(F_gp.T @ F_gp)
    #         #dC_dF = np.einsum('kp,ol->opkl', F_gp, np.eye(2)) + np.einsum('ko,pl->opkl', F_gp, np.eye(2))
    #         #dP_dF = np.einsum('ik,lj->ijkl', np.eye(2), S_gp)  + np.einsum('in,njop,opkl->ijkl',F_gp, dS_dC_gp, dC_dF)
            
    #         ######## numerical derivatives
    #         #dP_dF_num = NumericalDerivative(lambda t,F: self.mat.P(F), order=1).dR_dQ(0, F_gp, eps=eps)
    #         #dC_dF_num = NumericalDerivative(lambda t,F:  F.T @ F, order=1).dR_dQ(0, F_gp, eps=eps)
    #         #dS_dF_analytical = np.einsum('njop,opkl->njkl',dS_dC_gp, dC_dF)  
    #         eps = 1e-6
    #         dS_dF_num = NumericalDerivative(lambda t,F: self.mat.S(F.T @ F), order=1).dR_dQ(0, F_gp, eps=eps)

    #         dP_dF  = np.einsum('ik,lj->ijkl', np.eye(self.mat.ndim), S_gp)  + np.einsum('in,njkl->ijkl',F_gp, dS_dF_num)


    #         #######

    #         dP_dq_gp = np.einsum('klmn,mnj->klj',dP_dF, dF_dq)

    #         for a in range(self.nn):
    #             # TODO: reference to Bonet1997
    #             K_el[self.ndDOF[a]] += np.einsum('ijk,j->ik', dP_dq_gp ,self.nablaX_N[el,i,a]) * self.w_J0[el, i]

    #     return -K_el


    # def internal_forces_q(self, q):
    #     # analytical derivatives
    #     K_int = np.zeros((self.nq, self.nq))

    #     for el in range(self.nel):
    #         qel = q[self.elDOF[el]]

    #         # C^e^T C_a^T K C_a C^e += K^e
    #         K_int[np.ix_(self.elDOF[el], self.elDOF[el])] += self.internal_forces_el_q(qel, el)

    #     return K_int


    # def eval_all_gp(self, q):
    #     # required for vtk visualization
    #     P_all_gp = np.zeros((self.nel * self.mesh.nQP, self.mesh.nodal_DOF, self.mesh.nodal_DOF))
    #     F_all_gp = np.zeros((self.nel * self.mesh.nQP, self.mesh.nodal_DOF, self.mesh.nodal_DOF))
    #     W_all_gp = np.zeros((self.nel * self.mesh.nQP))
    #     n_all_eta = self.mesh.nel_eta * self.mesh.nQP_eta
    #     for el in range(self.nel):
    #         qel = q[self.elDOF[el]]
    #         el_xi, el_eta, el_nu = self.mesh.split_el(el)
    #         for i in range(self.mesh.nQP):
    #             i_xi, i_eta, i_nu = self.mesh.split_gp(i)
    #             i_all_xi = i_xi + el_xi*self.mesh.nQP_xi
    #             i_all_eta = i_eta + el_eta*self.mesh.nQP_eta
    #             i_all_nu = i_nu *el_nu*self.mesh.nQP_nu
    #             # compute deformation gradient of the deformed placement w.r.t. reference placement
    #             F_gp = np.zeros((self.mesh.nodal_DOF, self.mesh.nodal_DOF))
    #             for a in range(self.nn):
    #                 # TODO: reference
    #                 qa = qel[self.mesh.ndDOF[a]]
    #                 F_gp += np.outer(qa, self.nablaX_N[el,i,a])
                
    #             F_all_gp[i_all_eta + n_all_eta * i_all_xi] = F_gp
    #             P_all_gp[i_all_eta + n_all_eta * i_all_xi] = self.mat.P(F_gp)
    #             W_all_gp[i_all_eta + n_all_eta * i_all_xi] =  self.mat.W(F_gp.T @ F_gp)
    #     return F_all_gp, P_all_gp, W_all_gp

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
    continuum = First_gradient(mesh, Q, q0=q0)
    
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
    from cardillo.discretization.B_spline import Knot_vector, B_spline_volume2vtk, fit_B_spline_volume
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
    L = (R + B / 2) * phi0
    cube_shape = (L, B, H)
    Q = cube(cube_shape, mesh, Greville=True)

    # 3D continuum
    continuum = First_gradient(mesh, Q, q0=Q)

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
    q = concatenate((x, y, z))

    # export current configuration and deformation gradient on quadrature points to paraview
    continuum.post_processing(q, 'test.vtu')

if __name__ == "__main__":
    # test_gradient()
    test_gradient_vtk_export()