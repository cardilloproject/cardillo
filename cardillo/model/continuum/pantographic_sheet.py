import numpy as np
import meshio
import os
from math import asin

from cardillo.math.numerical_derivative import Numerical_derivative
from cardillo.discretization.indexing import flat2D, flat3D, split2D, split3D
from cardillo.discretization.B_spline import B_spline_basis3D
from cardillo.math.algebra import determinant2D, inverse3D, determinant3D, A_IK_basic_z, norm2

class Pantographic_sheet():
    def __init__(self, density, material, mesh, Z, z0=None, v0=None, cDOF=[], b=None, fiber_angle=np.pi/4):
        self.density = density
        self.mat = material

        # store generalized coordinates
        self.Z = Z
        z0 = z0 if z0 is not None else Z.copy()
        v0 = v0 if v0 is not None else np.zeros_like(Z)
        self.nz = len(Z)
        self.nc = len(cDOF)
        self.nq = self.nu = self.nz - self.nc
        self.zDOF = np.arange(self.nz)
        self.cDOF = cDOF
        self.fDOF = np.setdiff1d(self.zDOF, cDOF)
        self.q0 = z0[self.fDOF]
        self.u0 = v0[self.fDOF]

        if b is None:
            self.b = lambda t: np.array([], dtype=float)
        else:
            if callable(b):
                self.b = b
            else:
                self.b = lambda t: b
        assert len(cDOF) == len(self.b(0))

        # store mesh and extrac data
        self.mesh = mesh
        self.nel = mesh.nel
        self.nn = mesh.nn
        self.nn_el = mesh.nn_el # number of nodes of an element
        self.nq_el = mesh.nq_el
        self.nqp = mesh.nqp
        self.elDOF = mesh.elDOF
        self.nodalDOF = mesh.nodalDOF
        self.N = self.mesh.N

        self.dim = int(len(Z) / self.nn)
        assert self.dim == 2
        self.flat = flat2D
        self.determinant = determinant2D

        self.fiber_angle = fiber_angle
        self.R = A_IK_basic_z(fiber_angle)[:2, :2]

        # for each Gauss point, compute kappa0^-1, N_X and w_J0 = w * det(kappa0^-1)
        kappa0_xi_inv, N_X, self.w_J0 = self.mesh.reference_mappings(Z)

        # second gradient
        N_XX = self.mesh.N_XX(Z, kappa0_xi_inv)
        nel, nqp, nn_el, _, _ = N_XX.shape
        
        self.N_Theta = np.zeros_like(N_X)
        self.N_ThetaTheta = np.zeros_like(N_XX)
        for el in range(nel):
            for i in range(nqp):
                for a in range(nn_el):
                    self.N_Theta[el, i, a] = N_X[el, i, a] @ self.R.T
                    self.N_ThetaTheta[el, i, a] = self.R @ N_XX[el, i, a] @ self.R.T

    def assembler_callback(self):
        self.elfDOF = []
        self.elqDOF = []
        self.eluDOF = []
        for elDOF in self.elDOF:
            elfDOF = np.setdiff1d(elDOF, self.cDOF)
            self.elfDOF.append(np.searchsorted(elDOF, elfDOF))
            idx = np.searchsorted(self.fDOF, elfDOF)
            self.elqDOF.append(self.qDOF[idx])
            self.eluDOF.append(self.uDOF[idx])

    def z(self, t, q):
        z = np.zeros(self.nz)
        z[self.fDOF] = q
        z[self.cDOF] = self.b(t)
        return z
        
    def post_processing_single_configuration(self, t, q, filename, binary=True):
            # compute redundant generalized coordinates
            z = self.z(t, q)

            # generalized coordinates, connectivity and polynomial degree
            cells, points, HigherOrderDegrees = self.mesh.vtk_mesh(z)

            # dictionary storing point data
            point_data = {}
            
            # evaluate deformation gradient at quadrature points
            F = np.zeros((self.mesh.nel, self.mesh.nqp, self.mesh.nq_n, self.mesh.nq_n))
            for el in range(self.mesh.nel):
                ze = z[self.mesh.elDOF[el]]
                for i in range(self.mesh.nqp):
                    for a in range(self.mesh.nn_el):
                        F[el, i] += np.outer(ze[self.mesh.nodalDOF[a]], self.N_Theta[el, i, a]) # Bonet 1997 (7.6b)

            F_vtk = self.mesh.field_to_vtk(F)
            point_data.update({"F": F_vtk})

            # field data vtk export
            point_data_fields = {
                "C": lambda F: F.T @ F,
                "J": lambda F: np.array([self.determinant(F)]),
                # "P": lambda F: self.mat.P(F),
                # "S": lambda F: self.mat.S(F),
                # "W": lambda F: self.mat.W(F),
            }

            for name, fun in point_data_fields.items():
                tmp = fun(F_vtk[0].reshape(self.dim, self.dim)).reshape(-1)
                field = np.zeros((len(F_vtk), len(tmp)))
                for i, Fi in enumerate(F_vtk):
                    field[i] = fun(Fi.reshape(self.dim, self.dim)).reshape(-1)
                point_data.update({name: field})
        
            # write vtk mesh using meshio
            meshio.write_points_cells(
                os.path.splitext(os.path.basename(filename))[0] + '.vtu',
                points,
                cells,
                point_data=point_data,
                cell_data={"HigherOrderDegrees": HigherOrderDegrees},
                binary=binary
            )
        
    def post_processing(self, t, q, filename, binary=True):
        # write paraview PVD file collecting time and all vtk files, see https://www.paraview.org/Wiki/ParaView/Data_formats#PVD_File_Format
        from xml.dom import minidom
        
        root = minidom.Document()
        
        vkt_file = root.createElement('VTKFile')
        vkt_file.setAttribute('type', 'Collection')
        root.appendChild(vkt_file)
        
        collection = root.createElement('Collection')
        vkt_file.appendChild(collection)

        for i, (ti, qi) in enumerate(zip(t, q)):
            filei = filename + f'{i}.vtu'

            # write time step and file name in pvd file
            dataset = root.createElement('DataSet')
            dataset.setAttribute('timestep', f'{ti:0.6f}')
            dataset.setAttribute('file', filei)
            collection.appendChild(dataset)

            self.post_processing_single_configuration(ti, qi, filei, binary=binary)

        # write pvd file        
        xml_str = root.toprettyxml(indent ="\t")          
        with open(filename + '.pvd', "w") as f:
            f.write(xml_str)

    #########################################
    # kinematic equation
    #########################################
    def q_dot(self, t, q, u):
        return u

    def B(self, t, q, coo):
        coo.extend_diag(np.ones(self.nq), (self.qDOF, self.uDOF))

    def q_ddot(self, t, q, u, u_dot):
        return u_dot

    #########################################
    # equations of motion
    #########################################
    def M_el(self, el):
        M_el = np.zeros((self.nq_el, self.nq_el))

        I_nq_n = np.eye(self.dim)

        for a in range(self.nn_el):
            for b in range(self.nn_el):
                idx = np.ix_(self.nodalDOF[a], self.nodalDOF[b])
                for i in range(self.nqp):
                    N = self.N[el, i]
                    w_J0 = self.w_J0[el, i]
                    M_el[idx] += N[a] * N[b] * self.density * w_J0 * I_nq_n

        return M_el

    def M(self, t, q, coo):
        for el in range(self.nel):
            M_el = self.M_el(el)

            # sparse assemble element internal stiffness matrix
            elfDOF = self.elfDOF[el]
            eluDOF = self.eluDOF[el]
            coo.extend(M_el[elfDOF[:, None], elfDOF], (eluDOF, eluDOF))

    def f_pot_el(self, ze, el):
        f = np.zeros(self.nq_el)

        for i in range(self.nqp):
            N_Theta = self.N_Theta[el, i]
            N_ThetaTheta = self.N_ThetaTheta[el, i]
            w_J0 = self.w_J0[el, i]

            # first deformation gradient
            F = np.zeros((self.dim, self.dim))
            for a in range(self.nn_el):
                F += np.outer(ze[self.nodalDOF[a]], N_Theta[a]) # Bonet 1997 (7.5)

            # second deformation gradient
            G = np.zeros((self.dim, self.dim, self.dim))
            for a in range(self.nn_el):
                G += np.einsum('i,jk->ijk', ze[self.nodalDOF[a]], N_ThetaTheta[a]) # TODO: reference to Evan's thesis

            # strain measures of pantographic sheet
            d1 = F[:, 0]
            d2 = F[:, 1]
            rho1 = norm2(d1)
            rho2 = norm2(d2)
            rho = np.array([rho1, rho2])

            e1 = d1 / rho1
            e2 = d2 / rho2

            d1_1 = G[:, 0, 0]
            d1_2 = G[:, 0, 1]
            d2_1 = G[:, 1, 0]
            d2_2 = G[:, 1, 1]
            rho1_1 = d1_1 @ e1
            rho1_2 = d1_2 @ e1
            rho2_1 = d2_1 @ e2
            rho2_2 = d2_2 @ e2
            rho_s = np.array([[rho1_1, rho1_2],
                              [rho2_1, rho2_2]])

            d1_perp = np.array([-d1[1], d1[0]])
            d2_perp = np.array([-d2[1], d2[0]])
            theta1_1 = d1_1 @ d1_perp / rho1**2
            theta1_2 = d1_2 @ d1_perp / rho1**2
            theta2_1 = d2_1 @ d2_perp / rho2**2
            theta2_2 = d2_2 @ d2_perp / rho2**2
            theta_s = np.array([[theta1_1, theta1_2],
                                [theta2_1, theta2_2]])

            Gamma = asin(e2 @ e1)

            # evaluate material model (-> derivatives w.r.t. strain measures)
            W_rho = self.mat.W_rho(rho, rho_s, Gamma, theta_s)
            W_rho_s = self.mat.W_rho_s(rho, rho_s, Gamma, theta_s)
            W_Gamma = self.mat.W_Gamma(rho, rho_s, Gamma, theta_s)
            W_theta_s = self.mat.W_theta_s(rho, rho_s, Gamma, theta_s)
            
            # internal forces
            for a in range(self.nn_el):
                # delta rho_alpha
                rho1_q = N_Theta[a, 0] * e1
                rho2_q = N_Theta[a, 1] * e2

                # TODO: delta rho_s zero for Maurin2019 material

                # delta Gamma 
                d1_q = np.array([N_Theta[a, 0], N_Theta[a, 0]])   
                d2_q = np.array([N_Theta[a, 1], N_Theta[a, 1]])   

                Gamma_q = (d1 @ d2_q + d2 @ d1_q) / (rho1 * rho2) \
                          - (d1 @ d2) / (rho1 * rho2)**2 * (rho2 * rho1_q + rho1 * rho2_q)

                # delta theta_s
                d1_1_q = np.array([N_ThetaTheta[a, 0, 0], N_ThetaTheta[a, 0, 0]])
                d1_2_q = np.array([N_ThetaTheta[a, 0, 1], N_ThetaTheta[a, 0, 1]])
                d2_1_q = np.array([N_ThetaTheta[a, 1, 0], N_ThetaTheta[a, 1, 0]])
                d2_2_q = np.array([N_ThetaTheta[a, 1, 1], N_ThetaTheta[a, 1, 1]])

                d1_perp_q = np.array([-N_Theta[a, 0], N_Theta[a, 0]])   
                d2_perp_q = np.array([-N_Theta[a, 1], N_Theta[a, 1]]) 

                theta1_1_q = (d1_1 @ d1_perp_q + d1_perp @ d1_1_q) / rho1**2 + (d1_1 @ d1_perp) / rho1**3 * rho1_q
                theta1_2_q = (d1_2 @ d1_perp_q + d1_perp @ d1_2_q) / rho1**2 + (d1_2 @ d1_perp) / rho1**3 * rho1_q
                
                theta2_1_q = (d2_1 @ d2_perp_q + d2_perp @ d2_1_q) / rho2**2 + (d2_1 @ d2_perp) / rho2**3 * rho2_q
                theta2_2_q = (d2_2 @ d2_perp_q + d2_perp @ d2_2_q) / rho2**2 + (d2_2 @ d2_perp) / rho2**3 * rho2_q

                f[self.nodalDOF[a]] -= (W_rho[0] * rho1_q + W_rho[1] * rho2_q
                                        + W_Gamma * Gamma_q
                                        + W_theta_s[0, 0] * theta1_1_q + W_theta_s[0, 1] * theta1_2_q + W_theta_s[1, 0] * theta2_1_q + W_theta_s[1, 1] * theta2_2_q) * w_J0
                # f[self.nodalDOF[a]] -= (W_rho[0] * rho1_q + W_rho[1] * rho2_q) * w_J0

        return f

    def f_pot(self, t, q):
        z = self.z(t, q)
        f_pot = np.zeros(self.nz)
        for el in range(self.nel):
            f_pot[self.elDOF[el]] += self.f_pot_el(z[self.elDOF[el]], el)
        return f_pot[self.fDOF]

    def f_pot_q_el(self, ze, el):
        Ke = np.zeros((self.nq_el, self.nq_el))
        I3 = np.eye(self.dim)

        for i in range(self.nqp):
            N_X = self.N_X[el, i]
            w_J0 = self.w_J0[el, i]

            # deformation gradient
            F = np.zeros((self.dim, self.dim))
            F_q = np.zeros((self.dim, self.dim, self.nq_el))
            for a in range(self.nn_el):
                F += np.outer(ze[self.nodalDOF[a]], N_X[a]) # Bonet 1997 (7.5)
                F_q[:, :, self.nodalDOF[a]] += np.einsum('ik,j->ijk', I3, N_X[a])

            # differentiate first Piola-Kirchhoff deformation tensor w.r.t. generalized coordinates
            S = self.mat.S(F)
            S_F = self.mat.S_F(F)
            P_F  = np.einsum('ik,lj->ijkl', I3, S)  + np.einsum('in,njkl->ijkl', F, S_F)
            P_q = np.einsum('klmn,mnj->klj', P_F, F_q)

            # internal element stiffness matrix
            for a in range(self.nn_el):
                Ke[self.nodalDOF[a]] += np.einsum('ijk,j->ik', P_q, -N_X[a] * w_J0)

        return Ke

    def f_pot_q(self, t, q, coo):
        z = self.z(t, q)
        for el in range(self.nel):
            # Ke = self.f_pot_q_el(z[self.elDOF[el]], el)
            # Ke_num = Numerical_derivative(lambda t, z: self.f_pot_el(z, el), order=2)._x(t, z[self.elDOF[el]])
            # error = np.linalg.norm(Ke - Ke_num)
            # print(f'error: {error}')
            Ke = Numerical_derivative(lambda t, z: self.f_pot_el(z, el), order=2)._x(t, z[self.elDOF[el]])

            # sparse assemble element internal stiffness matrix
            elfDOF = self.elfDOF[el]
            eluDOF = self.eluDOF[el]
            elqDOF = self.elqDOF[el]
            coo.extend(Ke[elfDOF[:, None], elfDOF], (eluDOF, elqDOF))

    ####################################################
    # surface forces
    ####################################################
    def force_distr2D_el(self, force, t, el, srf_mesh):
        fe = np.zeros(srf_mesh.nq_el)

        el_xi, el_eta = split2D(el, (srf_mesh.nel_xi,))

        for i in range(srf_mesh.nqp):
            N = srf_mesh.N[el, i]
            w_J0 = self.srf_w_J0[srf_mesh.idx][el, i]
            
            i_xi, i_eta = split2D(i, (srf_mesh.nqp_xi,))
            xi = srf_mesh.qp_xi[el_xi, i_xi]
            eta = srf_mesh.qp_eta[el_eta, i_eta]

            # internal forces
            for a in range(srf_mesh.nn_el):
                fe[srf_mesh.nodalDOF[a]] += force(t, xi, eta) * N[a] * w_J0

        return fe

    def force_distr2D(self, t, q, force, srf_idx):
        z = self.z(t, q)
        f = np.zeros(self.nz)

        srf_mesh = self.mesh.surface_mesh[srf_idx]
        srf_zDOF = self.mesh.surface_qDOF[srf_idx].ravel()
        
        for el in range(srf_mesh.nel):
            f[srf_zDOF[srf_mesh.elDOF[el]]] += self.force_distr2D_el(force, t, el, srf_mesh)
        return f[self.fDOF]

    def force_distr2D_q(self, t, q, coo, force, srf_idx):
        pass

    ####################################################
    # volume forces
    ####################################################
    def force_distr3D_el(self, force, t, el):
        fe = np.zeros(self.nq_el)

        el_xi, el_eta, el_zeta = split3D(el, (self.mesh.nel_xi, self.mesh.nel_eta))

        for i in range(self.nqp):
            N = self.mesh.N[el, i]
            w_J0 = self.w_J0[el, i]
            
            i_xi, i_eta, i_zeta = split3D(i, (self.mesh.nqp_xi, self.mesh.nqp_eta))
            xi = self.mesh.qp_xi[el_xi, i_xi]
            eta = self.mesh.qp_eta[el_eta, i_eta]
            zeta = self.mesh.qp_zeta[el_zeta, i_zeta]

            # internal forces
            for a in range(self.nn_el):
                fe[self.nodalDOF[a]] += force(t, xi, eta, zeta) * N[a] * w_J0

        return fe

    def force_distr3D(self, t, q, force):
        z = self.z(t, q)
        f = np.zeros(self.nz)
        
        for el in range(self.nel):
            f[self.elDOF[el]] += self.force_distr3D_el(force, t, el)
        return f[self.fDOF]

    def force_distr3D_q(self, t, q, coo, force):
        pass
