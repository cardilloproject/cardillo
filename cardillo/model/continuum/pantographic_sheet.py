import numpy as np
import meshio
import os
from math import asin
import matplotlib.pyplot as plt

from cardillo.math.numerical_derivative import Numerical_derivative
from cardillo.discretization.indexing import flat2D, flat3D, split2D, split3D
from cardillo.discretization.B_spline import B_spline_basis3D
from cardillo.math.algebra import determinant2D, inverse3D, determinant3D, A_IK_basic_z, norm2
from cardillo.discretization.mesh2D import Mesh2D

def strain_measures(F, G):
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

    return rho, rho_s, Gamma, theta_s, e1, e2

def strain_single_point(continuum, t, q, vxi, displ, comp_data_rho=None, comp_data_gamma=None):
    # evaluate individual points
    mesh = continuum.mesh
    knot_vector_objs = mesh.knot_vector_objs
    el_xi, el_eta = tuple(knot_vector_objs[i].element_number(vxi[i]) for i in range(2))
    el = flat2D(el_xi, el_eta, mesh.nel_per_dim)
    mesh_eval = Mesh2D(knot_vector_objs, mesh.nqp_per_dim, derivative_order=2, basis='B-spline', nq_n=2, vxi=vxi, elDOF=mesh.elDOF[el])
    continuum_eval = Pantographic_sheet(None, continuum.mat, mesh_eval, continuum.Z, cDOF=continuum.cDOF, b=continuum.b)

    rho = np.zeros([len(t), 2])
    rho_s = np.zeros([len(t), 2, 2])
    Gamma = np.zeros([len(t)])
    theta_s = np.zeros([len(t), 2, 2])

    for i, (ti, qi) in enumerate(zip(t, q)):
        rho[i], rho_s[i], Gamma[i], theta_s[i] = continuum_eval.post_processing_single_configuration(ti, qi, None, return_strain=True)
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(t * displ, (rho-1) * 100)
    ax[0, 1].plot(t * displ, - Gamma * 180 / np.pi + 90)
    ax[1, 0].plot(t * displ, rho_s.reshape(-1, 4))
    ax[1, 1].plot(t * displ, theta_s.reshape(-1, 4))

    ax[0, 0].set_title("rho")
    ax[0, 1].set_title("phi")
    ax[1, 0].set_title("rho_s")
    ax[1, 1].set_title("theta_s")
    ax[1, 0].set_xlabel("time")
    ax[1, 1].set_xlabel("time")

    if comp_data_rho is not None:
        ax[0, 0].plot(comp_data_rho[0], comp_data_rho[1], color='gray', linestyle='dotted')
    if comp_data_gamma is not None:
        ax[0, 1].plot(comp_data_gamma[0], comp_data_gamma[1], color='gray', linestyle='dotted')

    plt.show()

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
        # self.nn = mesh.nn
        self.nn_el = mesh.nn_el # number of nodes of an element
        self.nq_el = mesh.nq_el
        self.nqp = mesh.nqp
        self.elDOF = mesh.elDOF
        self.nodalDOF = mesh.nodalDOF
        self.N = self.mesh.N

        # self.dim = int(len(Z) / self.nn)
        # assert self.dim == 2  #TODO: check is currently not compatible with evaluation mesh
        self.dim = 2
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
        
    def post_processing_single_configuration(self, t, q, filename, binary=True, return_strain=False):
            # compute redundant generalized coordinates
            z = self.z(t, q)

            # generalized coordinates, connectivity and polynomial degree
            cells, points, HigherOrderDegrees = self.mesh.vtk_mesh(z)

            # dictionary storing point data
            point_data = {}
            
            # evaluate deformation gradient at quadrature points
            F = np.zeros((self.mesh.nel, self.mesh.nqp, self.dim, self.dim))
            G = np.zeros((self.mesh.nel, self.mesh.nqp, self.dim, self.dim, self.dim))
            for el in range(self.mesh.nel):
                ze = z[self.mesh.elDOF[el]]
                for i in range(self.mesh.nqp):
                    for a in range(self.mesh.nn_el):
                        # first deformation gradient
                        F[el, i] += np.outer(ze[self.mesh.nodalDOF[a]], self.N_Theta[el, i, a]) # Bonet 1997 (7.6b)
                        G[el, i] += np.einsum('i,jk->ijk', ze[self.nodalDOF[a]], self.N_ThetaTheta[el, i, a]) 

            if return_strain == False:             
                F_vtk = self.mesh.field_to_vtk(F)
                G_vtk = self.mesh.field_to_vtk(G)
                point_data.update({"F": F_vtk, "G": G_vtk})

                # field data vtk export
                point_data_fields = {
                    #TODO: make strain_measures function calls less redundant
                    "C": lambda F, G: F.T @ F,
                    "J": lambda F, G: np.array([self.determinant(F)]),
                    "W": lambda F, G: self.mat.W(*strain_measures(F, G)[:4]),
                    "rho": lambda F, G: strain_measures(F, G)[0],
                    "rho_s": lambda F, G: strain_measures(F, G)[1].ravel(),
                    "Gamma": lambda F, G:  np.array([strain_measures(F, G)[2]]),
                    "theta_s": lambda F, G:  strain_measures(F, G)[3].ravel(), 
                    "e1": lambda F, G: np.append(strain_measures(F, G)[4], 0),
                    "e2": lambda F, G: np.append(strain_measures(F, G)[5], 0),
                    # "W_axial": lambda F, G: self.mat.W_axial(*strain_measures(F, G)[:4]),
                    # "W_bending": lambda F, G: self.mat.W_bending(*strain_measures(F, G)[:4]),
                    # "W_shear": lambda F, G: np.array([self.mat.W_shear(*strain_measures(F, G)[:4])]),
                }

                for name, fun in point_data_fields.items():
                    try:
                        tmp = fun(F_vtk[0].reshape(self.dim, self.dim), G_vtk[0].reshape(self.dim, self.dim, self.dim)).ravel()
                        field = np.zeros((len(F_vtk), len(tmp)))
                        for i, Fi in enumerate(F_vtk):
                            Gi = G_vtk[i]
                            field[i] = fun(Fi.reshape(self.dim, self.dim), Gi.reshape(self.dim, self.dim, self.dim)).ravel()
                        point_data.update({name: field})
                    except ValueError:
                        print(f"A math domain error occured in evaluating {name}. No field data returned for this metric.")
            
                # write vtk mesh using meshio
                meshio.write_points_cells(
                    os.path.splitext(os.path.basename(filename))[0] + '.vtu',
                    points,
                    cells,
                    point_data=point_data,
                    cell_data={"HigherOrderDegrees": HigherOrderDegrees},
                    binary=binary
                )

            else:
                return strain_measures(F[0, 0], G[0, 0])[:4]
        
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

            self.post_processing_single_configuration(ti, qi, filei, binary=True)

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
            d1_1_perp = np.array([d1_1[1], -d1_1[0]])
            d1_2_perp = np.array([d1_2[1], -d1_2[0]])
            d2_1_perp = np.array([d2_1[1], -d2_1[0]])
            d2_2_perp = np.array([d2_2[1], -d2_2[0]])
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

            # precompute matrices
            G_rho1 = (np.eye(2) - np.outer(e1, e1)) / rho1
            G_rho2 = (np.eye(2) - np.outer(e2, e2)) / rho2
            
            # internal forces
            for a in range(self.nn_el):

                # delta rho_s
                rho1_q = N_Theta[a, 0] * e1
                rho2_q = N_Theta[a, 1] * e2   

                # delta rho_s_s (currently unused)
                rho1_1_q = N_ThetaTheta[a, 0, 0] * e1 + N_Theta[a, 0] * G_rho1 @ d1_1
                rho1_2_q = N_ThetaTheta[a, 0, 1] * e1 + N_Theta[a, 0] * G_rho1 @ d1_2
                rho2_1_q = N_ThetaTheta[a, 1, 0] * e2 + N_Theta[a, 1] * G_rho2 @ d2_1
                rho2_2_q = N_ThetaTheta[a, 1, 1] * e2 + N_Theta[a, 1] * G_rho2 @ d2_2

                # delta Gamma 
                Gamma_q = 1/(1-(e2 @ e1)**2)**0.5 * (
                     (d1 * N_Theta[a, 1] + d2 * N_Theta[a, 0]) / (rho1 * rho2) 
                    - (d1 @ d2) / (rho1 * rho2)**2 * (rho2 * rho1_q + rho1 * rho2_q)
                    )

                # delta theta_s_s
                d1_1_q = N_ThetaTheta[a, 0, 0]
                d1_2_q = N_ThetaTheta[a, 0, 1]
                d2_1_q = N_ThetaTheta[a, 1, 0]
                d2_2_q = N_ThetaTheta[a, 1, 1]

                d1_perp_q = np.array([[0,-1],[1,0]]) * N_Theta[a, 0]  
                d2_perp_q = np.array([[0,-1],[1,0]]) * N_Theta[a, 1]
                
                theta1_1_q = (d1_1_perp * N_Theta[a, 0] + d1_perp * d1_1_q) / rho1**2 - (d1_1 @ d1_perp) / rho1**3 * 2 * rho1_q
                theta1_2_q = (d1_2_perp * N_Theta[a, 0] + d1_perp * d1_2_q) / rho1**2 - (d1_2 @ d1_perp) / rho1**3 * 2 * rho1_q
                theta2_1_q = (d2_1_perp * N_Theta[a, 1] + d2_perp * d2_1_q) / rho2**2 - (d2_1 @ d2_perp) / rho2**3 * 2 * rho2_q
                theta2_2_q = (d2_2_perp * N_Theta[a, 1] + d2_perp * d2_2_q) / rho2**2 - (d2_2 @ d2_perp) / rho2**3 * 2 * rho2_q

                f[self.nodalDOF[a]] -= (  
                                        W_rho[0] * rho1_q + W_rho[1] * rho2_q
                                        + W_rho_s[0, 0] * rho1_1_q + W_rho_s[0, 1] * rho1_2_q + W_rho_s[1, 0] * rho2_1_q + W_rho_s[1, 1] * rho2_2_q
                                        + W_Gamma * Gamma_q
                                        + W_theta_s[0, 0] * theta1_1_q + W_theta_s[0, 1] * theta1_2_q + W_theta_s[1, 0] * theta2_1_q + W_theta_s[1, 1] * theta2_2_q
                                        ) * w_J0

        return f

    def f_pot(self, t, q):
        z = self.z(t, q)
        f_pot = np.zeros(self.nz)
        for el in range(self.nel):
            f_pot[self.elDOF[el]] += self.f_pot_el(z[self.elDOF[el]], el)
        return f_pot[self.fDOF]

    def f_pot_q_el(self, ze, el):
        Ke = np.zeros((self.nq_el, self.nq_el))

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
            d1_1_perp = np.array([d1_1[1], -d1_1[0]])
            d1_2_perp = np.array([d1_2[1], -d1_2[0]])
            d2_1_perp = np.array([d2_1[1], -d2_1[0]])
            d2_2_perp = np.array([d2_2[1], -d2_2[0]])
            theta1_1 = d1_1 @ d1_perp / rho1**2
            theta1_2 = d1_2 @ d1_perp / rho1**2
            theta2_1 = d2_1 @ d2_perp / rho2**2
            theta2_2 = d2_2 @ d2_perp / rho2**2
            theta_s = np.array([[theta1_1, theta1_2],
                                [theta2_1, theta2_2]])

            Gamma = asin(e2 @ e1)

            # precompute matrices
            G_rho1 = (np.eye(2) - np.outer(e1, e1)) / rho1
            G_rho2 = (np.eye(2) - np.outer(e2, e2)) / rho2
            G_rho1_1 = (-G_rho1 @ np.outer(d1_1, e1) - np.dot(e1, d1_1) * G_rho1 - np.outer(e1, d1_1) @ G_rho1) / rho1
            G_rho1_2 = (-G_rho1 @ np.outer(d1_2, e1) - np.dot(e1, d1_2) * G_rho1 - np.outer(e1, d1_2) @ G_rho1) / rho1
            G_rho2_1 = (-G_rho2 @ np.outer(d2_1, e2) - np.dot(e2, d2_1) * G_rho2 - np.outer(e2, d2_1) @ G_rho2) / rho2
            G_rho2_2 = (-G_rho2 @ np.outer(d2_2, e2) - np.dot(e2, d2_2) * G_rho2 - np.outer(e2, d2_2) @ G_rho2) / rho2

            tmp = (1 - (e1 @ e2)**2)**0.5
            G_Gamma11 = (e1 @ e2 * G_rho1 @ np.outer(e2, e2) @ G_rho1 / (1 - (e1 @ e2)**2) - (np.outer(e1, e2) + np.outer(e2, e1) + (np.eye(2) - 3 * np.outer(e1, e1)) * (e1 @ e2)) / rho1**2 ) / tmp
            G_Gamma12 = (e1 @ e2 * G_rho1 @ np.outer(e2, e1) @ G_rho2 / (1 - (e1 @ e2)**2) + G_rho1 @ G_rho2) / tmp
            G_Gamma21 = (e1 @ e2 * G_rho2 @ np.outer(e1, e2) @ G_rho1 / (1 - (e1 @ e2)**2) + G_rho2 @ G_rho1) / tmp
            G_Gamma22 = (e1 @ e2 * G_rho2 @ np.outer(e1, e1) @ G_rho2 / (1 - (e1 @ e2)**2) - (np.outer(e1, e2) + np.outer(e2, e1) + (np.eye(2) - 3 * np.outer(e2, e2)) * (e1 @ e2)) / rho2**2 ) / tmp
            G_theta11_1 = -2 * np.array([[0,-1],[1,0]]) @ (np.outer(d1, d1_1) + np.outer(d1_1, d1) + (np.eye(2) - 4 * np.outer(e1, e1)) * (d1_1 @ d1)) / rho1**4
            G_theta22_2 = -2 * np.array([[0,-1],[1,0]]) @ (np.outer(d2, d2_2) + np.outer(d2_2, d2) + (np.eye(2) - 4 * np.outer(e2, e2)) * (d2_2 @ d2)) / rho2**4
            G_theta11 = np.array([[0,-1],[1,0]]) @ (np.eye(2) - np.outer(e1, e1)) / rho1**2
            G_theta22 = np.array([[0,-1],[1,0]]) @ (np.eye(2) - np.outer(e2, e2)) / rho2**2
            G_theta12_1 = -2 * np.array([[0,-1],[1,0]]) @ (np.outer(d1, d1_2) + np.outer(d1_2, d1) + (np.eye(2) - 4 * np.outer(e1, e1)) * (d1_2 @ d1)) / rho1**4
            G_theta21_2 = -2 * np.array([[0,-1],[1,0]]) @ (np.outer(d2, d2_1) + np.outer(d2_1, d2) + (np.eye(2) - 4 * np.outer(e2, e2)) * (d2_1 @ d2)) / rho2**4

            # evaluate material model (-> derivatives w.r.t. strain measures)
            W_rho = self.mat.W_rho(rho, rho_s, Gamma, theta_s)
            W_rho_s = self.mat.W_rho_s(rho, rho_s, Gamma, theta_s)
            W_Gamma = self.mat.W_Gamma(rho, rho_s, Gamma, theta_s)
            W_theta_s = self.mat.W_theta_s(rho, rho_s, Gamma, theta_s)
            W_rho_rho = self.mat.W_rho_rho(rho, rho_s, Gamma, theta_s)
            W_Gamma_Gamma = self.mat.W_Gamma_Gamma(rho, rho_s, Gamma, theta_s)
            W_theta_s_theta_s = self.mat.W_theta_s_theta_s(rho, rho_s, Gamma, theta_s)

            # for Barchiesi
            W_rho_rho_s = self.mat.W_rho_rho_s(rho, rho_s, Gamma, theta_s)
            W_rho_theta_s = self.mat.W_rho_theta_s(rho, rho_s, Gamma, theta_s)
            W_rho_s_rho = self.mat.W_rho_s_rho(rho, rho_s, Gamma, theta_s)
            W_rho_s_rho_s = self.mat.W_rho_s_rho_s(rho, rho_s, Gamma, theta_s)
            W_theta_s_rho = self.mat.W_theta_s_rho(rho, rho_s, Gamma, theta_s)
            
            rho1_q = np.zeros((self.nq_el))
            rho2_q = np.zeros((self.nq_el))
            rho1_1_q = np.zeros((self.nq_el))
            rho1_2_q = np.zeros((self.nq_el))
            rho2_1_q = np.zeros((self.nq_el))
            rho2_2_q = np.zeros((self.nq_el))
            Gamma_q = np.zeros((self.nq_el))
            theta1_1_q = np.zeros((self.nq_el))
            theta1_2_q = np.zeros((self.nq_el))
            theta2_1_q = np.zeros((self.nq_el))
            theta2_2_q = np.zeros((self.nq_el))

            for a in range(self.nn_el):
                ndDOFa = self.mesh.nodalDOF[a]

                # delta rho_s
                rho1_q[ndDOFa] = N_Theta[a, 0] * e1
                rho2_q[ndDOFa] = N_Theta[a, 1] * e2    

                # delta rho_s_s
                rho1_1_q[ndDOFa] = N_ThetaTheta[a, 0, 0] * e1  +  N_Theta[a, 0] * G_rho1 @ d1_1
                # rho1_2_q[ndDOFa] = N_ThetaTheta[a, 0, 1] * e1  +  N_Theta[a, 0] * G_rho1 @ d1_2
                # rho2_1_q[ndDOFa] = N_ThetaTheta[a, 1, 0] * e2  +  N_Theta[a, 1] * G_rho2 @ d2_1
                rho2_2_q[ndDOFa] = N_ThetaTheta[a, 1, 1] * e2  +  N_Theta[a, 1] * G_rho2 @ d2_2

                # delta Gamma 
                Gamma_q[ndDOFa] = 1/(1-(e2 @ e1)**2)**0.5 * (
                     (d1 * N_Theta[a, 1] + d2 * N_Theta[a, 0]) / (rho1 * rho2) 
                    - (d1 @ d2) / (rho1 * rho2)**2 * (rho2 * rho1_q[ndDOFa] + rho1 * rho2_q[ndDOFa])
                    )
                
                # delta theta_s_s
                theta1_1_q[ndDOFa] = (d1_1_perp * N_Theta[a, 0] + d1_perp * N_ThetaTheta[a, 0, 0]) / rho1**2 - (d1_1 @ d1_perp) / rho1**3 * 2 * rho1_q[ndDOFa]
                # theta1_2_q[ndDOFa] = (d1_2_perp * N_Theta[a, 0] + d1_perp * N_ThetaTheta[a, 0, 1]) / rho1**2 - (d1_2 @ d1_perp) / rho1**3 * 2 * rho1_q[ndDOFa]
                # theta2_1_q[ndDOFa] = (d2_1_perp * N_Theta[a, 1] + d2_perp * N_ThetaTheta[a, 1, 0]) / rho2**2 - (d2_1 @ d2_perp) / rho2**3 * 2 * rho2_q[ndDOFa]
                theta2_2_q[ndDOFa] = (d2_2_perp * N_Theta[a, 1] + d2_perp * N_ThetaTheta[a, 1, 1]) / rho2**2 - (d2_2 @ d2_perp) / rho2**3 * 2 * rho2_q[ndDOFa]  
                
            for a in range(self.nn_el):
                ndDOFa = self.mesh.nodalDOF[a]
                for b in range(self.nn_el):
                    ndDOFb = self.mesh.nodalDOF[b]

                    rho1_qq = N_Theta[a, 0] * G_rho1 * N_Theta[b, 0]
                    rho2_qq = N_Theta[a, 1] * G_rho2 * N_Theta[b, 1]

                    rho1_1_qq = N_ThetaTheta[a, 0, 0] * G_rho1 * N_Theta[b, 0] + N_Theta[a, 0] * G_rho1 * N_ThetaTheta[b, 0, 0] + N_Theta[a, 0] * G_rho1_1  * N_Theta[b, 0]
                    # rho1_2_qq = N_ThetaTheta[a, 0, 1] * G_rho1 * N_Theta[b, 0] + N_Theta[a, 0] * G_rho1 * N_ThetaTheta[b, 0, 1] + N_Theta[a, 0] * G_rho1_2  * N_Theta[b, 0]
                    # rho2_1_qq = N_ThetaTheta[a, 1, 0] * G_rho2 * N_Theta[b, 1] + N_Theta[a, 1] * G_rho2 * N_ThetaTheta[b, 1, 0] + N_Theta[a, 1] * G_rho2_1  * N_Theta[b, 1]
                    rho2_2_qq = N_ThetaTheta[a, 1, 1] * G_rho2 * N_Theta[b, 1] + N_Theta[a, 1] * G_rho2 * N_ThetaTheta[b, 1, 1] + N_Theta[a, 1] * G_rho2_2  * N_Theta[b, 1]

                    Gamma_qq = N_Theta[a, 0] * G_Gamma11 * N_Theta[b, 0] + N_Theta[a, 0] * G_Gamma12 * N_Theta[b, 1] + N_Theta[a, 1] * G_Gamma21 * N_Theta[b, 0] + N_Theta[a, 1] * G_Gamma22 * N_Theta[b, 1]
                    
                    theta1_1_qq = N_Theta[a, 0] * G_theta11_1 * N_Theta[b, 0] + N_ThetaTheta[a, 0, 0] * G_theta11 * N_Theta[b, 0] + N_Theta[a, 0] * G_theta11 * N_ThetaTheta[b, 0, 0]
                    # theta1_2_qq = N_Theta[a, 0] * G_theta12_1 * N_Theta[b, 0] + N_ThetaTheta[a, 0, 1] * G_theta11 * N_Theta[b, 0] + N_Theta[a, 0] * G_theta11 * N_ThetaTheta[b, 0, 1]
                    # theta2_1_qq = N_Theta[a, 1] * G_theta21_2 * N_Theta[b, 1] + N_ThetaTheta[a, 1, 0] * G_theta22 * N_Theta[b, 1] + N_Theta[a, 1] * G_theta22 * N_ThetaTheta[b, 1, 0]
                    theta2_2_qq = N_Theta[a, 1] * G_theta22_2 * N_Theta[b, 1] + N_ThetaTheta[a, 1, 1] * G_theta22 * N_Theta[b, 1] + N_Theta[a, 1] * G_theta22 * N_ThetaTheta[b, 1, 1]

                    Ke[np.ix_(ndDOFa, ndDOFb)] -= (
                          W_rho[0] * rho1_qq + W_rho[1] * rho2_qq
                        + W_rho_s[0, 0] * rho1_1_qq + W_rho_s[1, 1] * rho2_2_qq
                        + W_Gamma * Gamma_qq
                        + W_theta_s[0, 0] * theta1_1_qq + W_theta_s[1, 1] * theta2_2_qq #+ W_theta_s[0, 1] * theta1_2_qq + W_theta_s[1, 0] * theta2_1_qq
                        + W_rho_rho[0, 0] * np.outer(rho1_q[ndDOFa], rho1_q[ndDOFb]) + W_rho_rho[1, 1] * np.outer(rho2_q[ndDOFa], rho2_q[ndDOFb]) #+ W_rho_rho[0, 1] * np.outer(rho1_q[ndDOFa], rho2_q[ndDOFb]) + W_rho_rho[1, 0] * np.outer(rho2_q[ndDOFa], rho1_q[ndDOFb])
                        + W_theta_s_theta_s[0, 0, 0, 0] * np.outer(theta1_1_q[ndDOFa], theta1_1_q[ndDOFb]) + W_theta_s_theta_s[1, 1, 1, 1] * np.outer(theta2_2_q[ndDOFa], theta2_2_q[ndDOFb]) 
                        #Maurin
                        + W_Gamma_Gamma * np.outer(Gamma_q[ndDOFa], Gamma_q[ndDOFb])
                        #Barchiesi
                        + W_rho_rho_s[0, 0, 0] * np.outer(rho1_q[ndDOFa], rho1_1_q[ndDOFb]) + W_rho_rho_s[1, 1, 1] * np.outer(rho2_q[ndDOFa], rho2_2_q[ndDOFb])
                        + W_rho_theta_s[0, 0, 0] * np.outer(rho1_q[ndDOFa], theta1_1_q[ndDOFb]) + W_rho_theta_s[1, 1, 1] * np.outer(rho2_q[ndDOFa], theta2_2_q[ndDOFb]) 
                        + W_rho_s_rho[0, 0, 0] * np.outer(rho1_1_q[ndDOFa], rho1_q[ndDOFb]) + W_rho_s_rho[1, 1, 1] * np.outer(rho2_2_q[ndDOFa], rho2_q[ndDOFb])
                        + W_rho_s_rho_s[0, 0, 0, 0] * np.outer(rho1_1_q[ndDOFa], rho1_1_q[ndDOFb]) + W_rho_s_rho_s[1, 1, 1, 1] * np.outer(rho2_2_q[ndDOFa], rho2_2_q[ndDOFb])
                        + W_theta_s_rho[0, 0, 0] * np.outer(theta1_1_q[ndDOFa], rho1_q[ndDOFb]) + W_theta_s_rho[1, 1, 1] * np.outer(theta2_2_q[ndDOFa], rho2_q[ndDOFb])
                        ) * w_J0

        return Ke

    def f_pot_q(self, t, q, coo):
        z = self.z(t, q)
        for el in range(self.nel):
            Ke = self.f_pot_q_el(z[self.elDOF[el]], el)
            # Ke_num = Numerical_derivative(lambda t, z: self.f_pot_el(z, el), order=2)._x(t, z[self.elDOF[el]])
            # error = np.linalg.norm(Ke - Ke_num)
            # print(f'error K: {error}')
            # Ke = Numerical_derivative(lambda t, z: self.f_pot_el(z, el), order=2)._x(t, z[self.elDOF[el]])

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



def test_gradients():
    # this test compares the deformation gradients F and G for the analytical and the discretized formulation of a deformed body

    from cardillo.discretization.mesh2D import Mesh2D, rectangle
    from cardillo.discretization.B_spline import Knot_vector, fit_B_spline_volume
    from cardillo.discretization.indexing import flat2D
    from cardillo.model.continuum import Maurin2019
    from cardillo.math.numerical_derivative import Numerical_derivative
    from cardillo.discretization.indexing import split2D 
    from cardillo.math.algebra import A_IK_basic_z


    QP_shape = (2, 2)
    degrees = (3, 3)
    element_shape = (5, 3)

    Xi = Knot_vector(degrees[0], element_shape[0])
    Eta = Knot_vector(degrees[1], element_shape[1])
    knot_vectors = (Xi, Eta)
    
    mesh = Mesh2D(knot_vectors, QP_shape, derivative_order=2, basis='B-spline', nq_n=2)

    # reference configuration is a cube
    alpha0 = np.pi / 2
    R = 0.1
    B = 1
    # L = (R) * alpha0
    L = 2
    rectangle_shape = (L, B)
    Z = rectangle(rectangle_shape, mesh, Greville=True)

    # material model
    K_rho = 1.34e5
    K_Gamma = 1.59e2
    K_Theta_s = 1.92e-2
    gamma = 1.36
    mat = Maurin2019(K_rho, K_Gamma, K_Theta_s, gamma)

    # 3D continuum
    continuum = Pantographic_sheet(None, mat, mesh, Z, z0=Z, fiber_angle=np.pi/4)

    # fit quater circle configuration
    def kappa(vxi):
        xi, eta = vxi
        alpha = (1 - xi) * alpha0
        x = (R + B * eta) * np.cos(alpha)
        y = (R + B * eta) * np.sin(alpha)
        return np.array([x, y])

    A = np.array([[L, 0], [0, B]])
    rotation = A_IK_basic_z(continuum.fiber_angle)[:2, :2]
    kappa0 = lambda vxi: rotation @ A @ vxi
    inv_kappa0 = lambda vX: np.linalg.inv(rotation @ A) @ vX
    phi = lambda vX: kappa(inv_kappa0(vX))
    phi_vX_num = lambda vX: Numerical_derivative(phi)._X(vX)

    nxi, neta = 20, 20
    xi = np.linspace(0, 1, num=nxi)
    eta = np.linspace(0, 1, num=neta)
    
    n2 = nxi * neta
    knots = np.zeros((n2, 2))
    Pw = np.zeros((n2, 2))
    for i, xii in enumerate(xi):
        for j, etai in enumerate(eta):
            idx = flat2D(i, j, (nxi, neta))
            knots[idx] = xii, etai
            Pw[idx] = kappa(np.array([xii, etai]))

    cDOF = np.array([], dtype=int)
    qc = np.array([], dtype=float).reshape((0, 3))
    x, y = fit_B_spline_volume(mesh, knots, Pw, qc, cDOF)
    z = np.concatenate((x, y))

    # export current configuration and deformation gradient on quadrature points to paraview
    continuum.post_processing([0], [z], 'test')

    # evaluate deformation gradient at quadrature points
    F = np.zeros((mesh.nel, mesh.nqp, continuum.dim, continuum.dim))
    G = np.zeros((mesh.nel, mesh.nqp, continuum.dim, continuum.dim, continuum.dim))
    F_num = np.zeros((mesh.nel, mesh.nqp, continuum.dim, continuum.dim))
    G_num = np.zeros((mesh.nel, mesh.nqp, continuum.dim, continuum.dim, continuum.dim))
    for el in range(mesh.nel):
        ze = z[mesh.elDOF[el]]
        el_xi, el_eta = split2D(el, mesh.element_shape)
        for i in range(mesh.nqp):
            i_xi, i_eta = split2D(i, mesh.nqp_per_dim)
            for a in range(mesh.nn_el):
                # deformation gradients
                F[el, i] += np.outer(ze[mesh.nodalDOF[a]], continuum.N_Theta[el, i, a]) # Bonet 1997 (7.6b)
                G[el, i] += np.einsum('i,jk->ijk', ze[mesh.nodalDOF[a]], continuum.N_ThetaTheta[el, i, a]) 

            vxi = np.array([mesh.qp_xi[el_xi, i_xi], mesh.qp_eta[el_eta, i_eta]])
            vX = kappa0(vxi)
            F_num[el, i] = phi_vX_num(vX)
            G_num[el, i] = Numerical_derivative(phi_vX_num)._X(vX)

    print(f"error F: {np.linalg.norm(F-F_num)}")
    print(f"error G: {np.linalg.norm(G-G_num)}")

def test_variations():
    # compare analytical and numerical expressions for rho_q, Gamma_q and theta_s_q
    from cardillo.math.numerical_derivative import Numerical_derivative
    from cardillo.discretization.mesh2D import Mesh2D, rectangle
    from cardillo.model.continuum.pantographic_sheet import Pantographic_sheet
    from cardillo.discretization.B_spline import Knot_vector, fit_B_spline_volume

    QP_shape = (2, 2)
    degrees = (3, 3)
    element_shape = (3, 3)

    Xi = Knot_vector(degrees[0], element_shape[0])
    Eta = Knot_vector(degrees[1], element_shape[1])
    knot_vectors = (Xi, Eta)
    mesh = Mesh2D(knot_vectors, QP_shape, derivative_order=2, basis='B-spline', nq_n=2)

    Z = rectangle((1, 1), mesh, Greville=True)
    continuum = Pantographic_sheet(None, None, mesh, Z, z0=Z, fiber_angle=np.pi/4)

    el = 0 # select arbitrary element
    i = 3 # select arbitrary quadrature point

    # fit quater circle configuration
    B = 1
    L = 2
    alpha0 = np.pi / 2
    R = 0.1

    def kappa(vxi):
        xi, eta = vxi
        alpha = (1 - xi) * alpha0
        x = (R + B * eta) * np.cos(alpha)
        y = (R + B * eta) * np.sin(alpha)
        return np.array([x, y])

    A = np.array([[L, 0], [0, B]])
    rotation = A_IK_basic_z(continuum.fiber_angle)[:2, :2]
    kappa0 = lambda vxi: rotation @ A @ vxi
    inv_kappa0 = lambda vX: np.linalg.inv(rotation @ A) @ vX
    phi = lambda vX: kappa(inv_kappa0(vX))
    phi_vX_num = lambda vX: Numerical_derivative(phi)._X(vX)

    nxi, neta = 20, 20
    xi = np.linspace(0, 1, num=nxi)
    eta = np.linspace(0, 1, num=neta)
    
    n2 = nxi * neta
    knots = np.zeros((n2, 2))
    Pw = np.zeros((n2, 2))
    for ii, xii in enumerate(xi):
        for j, etai in enumerate(eta):
            idx = flat2D(ii, j, (nxi, neta))
            knots[idx] = xii, etai
            Pw[idx] = kappa(np.array([xii, etai]))

    cDOF = np.array([], dtype=int)
    qc = np.array([], dtype=float).reshape((0, 3))
    x, y = fit_B_spline_volume(mesh, knots, Pw, qc, cDOF)
    z = np.concatenate((x, y))

    ze = z[mesh.elDOF[el]] 

    N_Theta = continuum.N_Theta[el, i]
    N_ThetaTheta = continuum.N_ThetaTheta[el, i]

    def strain_measure_single_point(continuum, ze):

        # first deformation gradient
        F = np.zeros((continuum.dim, continuum.dim))
        for a in range(continuum.nn_el):
            F += np.outer(ze[continuum.nodalDOF[a]], N_Theta[a]) # Bonet 1997 (7.5)

        # second deformation gradient
        G = np.zeros((continuum.dim, continuum.dim, continuum.dim))
        for a in range(continuum.nn_el):
            G += np.einsum('i,jk->ijk', ze[continuum.nodalDOF[a]], N_ThetaTheta[a]) # TODO: reference to Evan's thesis

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

        return rho, rho_s, Gamma, theta_s

    rho_q_num_fun = lambda ze: Numerical_derivative(lambda t, ze: strain_measure_single_point(continuum, ze)[0])._x(0, ze)
    rho_s_q_num_fun = lambda ze: Numerical_derivative(lambda t, ze: strain_measure_single_point(continuum, ze)[1])._x(0, ze)
    Gamma_q_num_fun = lambda ze: Numerical_derivative(lambda t, ze: np.array([strain_measure_single_point(continuum, ze)[2]]))._x(0, ze)
    theta_s_q_num_fun = lambda ze: Numerical_derivative(lambda t, ze: strain_measure_single_point(continuum, ze)[3])._x(0, ze)

    rho_q_num = rho_q_num_fun(ze)
    rho_s_q_num = rho_s_q_num_fun(ze)
    Gamma_q_num = Gamma_q_num_fun(ze)
    theta_s_q_num = theta_s_q_num_fun(ze)

    rho_qq_num = Numerical_derivative(rho_q_num_fun)._X(ze)
    rho_s_qq_num = Numerical_derivative(rho_s_q_num_fun)._X(ze)
    Gamma_qq_num = Numerical_derivative(Gamma_q_num_fun)._X(ze)
    theta_s_qq_num = Numerical_derivative(theta_s_q_num_fun)._X(ze)

    # Analytical derivatives
    # first deformation gradient
    F = np.zeros((continuum.dim, continuum.dim))
    for a in range(continuum.nn_el):
        F += np.outer(ze[continuum.nodalDOF[a]], N_Theta[a]) # Bonet 1997 (7.5)

    # second deformation gradient
    G = np.zeros((continuum.dim, continuum.dim, continuum.dim))
    for a in range(continuum.nn_el):
        G += np.einsum('i,jk->ijk', ze[continuum.nodalDOF[a]], N_ThetaTheta[a]) # TODO: reference to Evan's thesis

    # strain measures of pantographic sheet
    d1 = F[:, 0]
    d2 = F[:, 1]
    rho1 = norm2(d1)
    rho2 = norm2(d2)

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

    d1_perp = np.array([-d1[1], d1[0]])
    d2_perp = np.array([-d2[1], d2[0]])
    d1_1_perp = np.array([d1_1[1], -d1_1[0]])
    d1_2_perp = np.array([d1_2[1], -d1_2[0]])
    d2_1_perp = np.array([d2_1[1], -d2_1[0]])
    d2_2_perp = np.array([d2_2[1], -d2_2[0]])
    theta1_1 = d1_1 @ d1_perp / rho1**2
    theta1_2 = d1_2 @ d1_perp / rho1**2
    theta2_1 = d2_1 @ d2_perp / rho2**2
    theta2_2 = d2_2 @ d2_perp / rho2**2
    theta_s = np.array([[theta1_1, theta1_2],
                        [theta2_1, theta2_2]])

    # contributions to stiffness matrix
    G_rho1 = (np.eye(2) - np.outer(e1, e1)) / rho1
    G_rho2 = (np.eye(2) - np.outer(e2, e2)) / rho2
    G_rho1_1 = (-G_rho1 @ np.outer(d1_1, e1) - np.dot(e1, d1_1) * G_rho1 - np.outer(e1, d1_1) @ G_rho1) / rho1
    G_rho1_2 = (-G_rho1 @ np.outer(d1_2, e1) - np.dot(e1, d1_2) * G_rho1 - np.outer(e1, d1_2) @ G_rho1) / rho1
    G_rho2_1 = (-G_rho2 @ np.outer(d2_1, e2) - np.dot(e2, d2_1) * G_rho2 - np.outer(e2, d2_1) @ G_rho2) / rho2
    G_rho2_2 = (-G_rho2 @ np.outer(d2_2, e2) - np.dot(e2, d2_2) * G_rho2 - np.outer(e2, d2_2) @ G_rho2) / rho2

    tmp = (1 - (e1 @ e2)**2)**0.5
    G_Gamma11 = (e1 @ e2 * G_rho1 @ np.outer(e2, e2) @ G_rho1 / (1 - (e1 @ e2)**2) - (np.outer(e1, e2) + np.outer(e2, e1) + (np.eye(2) - 3 * np.outer(e1, e1)) * (e1 @ e2)) / rho1**2 ) / tmp
    G_Gamma12 = (e1 @ e2 * G_rho1 @ np.outer(e2, e1) @ G_rho2 / (1 - (e1 @ e2)**2) + G_rho1 @ G_rho2) / tmp
    G_Gamma21 = (e1 @ e2 * G_rho2 @ np.outer(e1, e2) @ G_rho1 / (1 - (e1 @ e2)**2) + G_rho2 @ G_rho1) / tmp
    G_Gamma22 = (e1 @ e2 * G_rho2 @ np.outer(e1, e1) @ G_rho2 / (1 - (e1 @ e2)**2) - (np.outer(e1, e2) + np.outer(e2, e1) + (np.eye(2) - 3 * np.outer(e2, e2)) * (e1 @ e2)) / rho2**2 ) / tmp

    G_theta11_1 = -2 * np.array([[0,-1],[1,0]]) @ (np.outer(d1, d1_1) + np.outer(d1_1, d1) + (np.eye(2) - 4 * np.outer(e1, e1)) * (d1_1 @ d1)) / rho1**4
    G_theta22_2 = -2 * np.array([[0,-1],[1,0]]) @ (np.outer(d2, d2_2) + np.outer(d2_2, d2) + (np.eye(2) - 4 * np.outer(e2, e2)) * (d2_2 @ d2)) / rho2**4
    G_theta11 = np.array([[0,-1],[1,0]]) @ (np.eye(2) - 2 * np.outer(e1, e1)) / rho1**2
    G_theta22 = np.array([[0,-1],[1,0]]) @ (np.eye(2) - 2 * np.outer(e2, e2)) / rho2**2

    G_theta12_1 = -2 * np.array([[0,-1],[1,0]]) @ (np.outer(d1, d1_2) + np.outer(d1_2, d1) + (np.eye(2) - 4 * np.outer(e1, e1)) * (d1_2 @ d1)) / rho1**4
    G_theta21_2 = -2 * np.array([[0,-1],[1,0]]) @ (np.outer(d2, d2_1) + np.outer(d2_1, d2) + (np.eye(2) - 4 * np.outer(e2, e2)) * (d2_1 @ d2)) / rho2**4

    rho_q_an = np.zeros((2, mesh.nq_el))
    rho_s_q_an = np.zeros((2, 2, mesh.nq_el))
    Gamma_q_an = np.zeros((1, mesh.nq_el))
    theta_s_q_an = np.zeros((2, 2, mesh.nq_el))
    G_rho1_q_an = np.zeros((2, mesh.nq_el))

    rho_qq_an = np.zeros((2, mesh.nq_el, mesh.nq_el))
    rho_s_qq_an = np.zeros((2, 2, mesh.nq_el, mesh.nq_el))
    Gamma_qq_an = np.zeros((1, mesh.nq_el, mesh.nq_el))
    theta_s_qq_an = np.zeros((2, 2, mesh.nq_el, mesh.nq_el))

    for a in range(continuum.nn_el):
        # delta rho_alpha
        rho1_q = N_Theta[a, 0] * e1
        rho2_q = N_Theta[a, 1] * e2

        # delta Gamma 
        Gamma_q = 1/(1-(e2 @ e1)**2)**0.5 * (
                     (d1 * N_Theta[a, 1] + d2 * N_Theta[a, 0]) / (rho1 * rho2) 
                    - (d1 @ d2) / (rho1 * rho2)**2 * (rho2 * rho1_q + rho1 * rho2_q)
                    )

        # delta theta_s
        d1_1_q = N_ThetaTheta[a, 0, 0]
        d1_2_q = N_ThetaTheta[a, 0, 1]
        d2_1_q = N_ThetaTheta[a, 1, 0]
        d2_2_q = N_ThetaTheta[a, 1, 1]

        d1_perp_q = np.array([[0,-1],[1,0]]) * N_Theta[a, 0]  
        d2_perp_q = np.array([[0,-1],[1,0]]) * N_Theta[a, 1]

        rho1_1_q = N_ThetaTheta[a, 0, 0] * e1  +  N_Theta[a, 0] * G_rho1 @ d1_1
        rho1_2_q = N_ThetaTheta[a, 0, 1] * e1  +  N_Theta[a, 0] * G_rho1 @ d1_2
        rho2_1_q = N_ThetaTheta[a, 1, 0] * e2  +  N_Theta[a, 1] * G_rho2 @ d2_1
        rho2_2_q = N_ThetaTheta[a, 1, 1] * e2  +  N_Theta[a, 1] * G_rho2 @ d2_2
        
        theta1_1_q = (d1_1_perp * N_Theta[a, 0] + d1_perp * d1_1_q) / rho1**2 - (d1_1 @ d1_perp) / rho1**3 * 2 * rho1_q
        theta1_2_q = (d1_2_perp * N_Theta[a, 0] + d1_perp * d1_2_q) / rho1**2 - (d1_2 @ d1_perp) / rho1**3 * 2 * rho1_q
        theta2_1_q = (d2_1_perp * N_Theta[a, 1] + d2_perp * d2_1_q) / rho2**2 - (d2_1 @ d2_perp) / rho2**3 * 2 * rho2_q
        theta2_2_q = (d2_2_perp * N_Theta[a, 1] + d2_perp * d2_2_q) / rho2**2 - (d2_2 @ d2_perp) / rho2**3 * 2 * rho2_q
        
        rho_q_an[:, mesh.nodalDOF[a]] += np.array([rho1_q, rho2_q])
        rho_s_q_an[:, :, mesh.nodalDOF[a]] += np.array([[rho1_1_q, rho1_2_q],
                                                     [rho2_1_q, rho2_2_q]])
        Gamma_q_an[:, mesh.nodalDOF[a]] += Gamma_q
        theta_s_q_an[:, :, mesh.nodalDOF[a]] += np.array([[theta1_1_q, theta1_2_q],
                                                          [theta2_1_q, theta2_2_q]])

        for b in range(continuum.nn_el):

            rho1_qq = N_Theta[a, 0] * G_rho1 * N_Theta[b, 0]
            rho2_qq = N_Theta[a, 1] * G_rho2 * N_Theta[b, 1]

            rho1_1_qq = N_ThetaTheta[a, 0, 0] * G_rho1 * N_Theta[b, 0] + N_Theta[a, 0] * G_rho1 * N_ThetaTheta[b, 0, 0] + N_Theta[a, 0] * G_rho1_1  * N_Theta[b, 0]
            rho1_2_qq = N_ThetaTheta[a, 0, 1] * G_rho1 * N_Theta[b, 0] + N_Theta[a, 0] * G_rho1 * N_ThetaTheta[b, 0, 1] + N_Theta[a, 0] * G_rho1_2  * N_Theta[b, 0]
            rho2_1_qq = N_ThetaTheta[a, 1, 0] * G_rho2 * N_Theta[b, 1] + N_Theta[a, 1] * G_rho2 * N_ThetaTheta[b, 1, 0] + N_Theta[a, 1] * G_rho2_1  * N_Theta[b, 1]
            rho2_2_qq = N_ThetaTheta[a, 1, 1] * G_rho2 * N_Theta[b, 1] + N_Theta[a, 1] * G_rho2 * N_ThetaTheta[b, 1, 1] + N_Theta[a, 1] * G_rho2_2  * N_Theta[b, 1]

            Gamma_qq = N_Theta[a, 0] * G_Gamma11 * N_Theta[b, 0] + N_Theta[a, 0] * G_Gamma12 * N_Theta[b, 1] + N_Theta[a, 1] * G_Gamma21 * N_Theta[b, 0] + N_Theta[a, 1] * G_Gamma22 * N_Theta[b, 1]
            theta1_1_qq = N_Theta[a, 0] * G_theta11_1 * N_Theta[b, 0] + N_ThetaTheta[a, 0, 0] * G_theta11 * N_Theta[b, 0] + N_Theta[a, 0] * G_theta11 * N_ThetaTheta[b, 0, 0]
            theta2_2_qq = N_Theta[a, 1] * G_theta22_2 * N_Theta[b, 1] + N_ThetaTheta[a, 1, 1] * G_theta22 * N_Theta[b, 1] + N_Theta[a, 1] * G_theta22 * N_ThetaTheta[b, 1, 1]
            theta1_2_qq = N_Theta[a, 0] * G_theta12_1 * N_Theta[b, 0] + N_ThetaTheta[a, 0, 1] * G_theta11 * N_Theta[b, 0] + N_Theta[a, 0] * G_theta11 * N_ThetaTheta[b, 0, 1]
            theta2_1_qq = N_Theta[a, 1] * G_theta21_2 * N_Theta[b, 1] + N_ThetaTheta[a, 1, 0] * G_theta22 * N_Theta[b, 1] + N_Theta[a, 1] * G_theta22 * N_ThetaTheta[b, 1, 0]

            rho_qq_an[np.ix_([0, 1], mesh.nodalDOF[a], mesh.nodalDOF[b])] += np.array([rho1_qq, rho2_qq])
            rho_s_qq_an[np.ix_([0, 1], [0, 1], mesh.nodalDOF[a], mesh.nodalDOF[b])] += np.array([[rho1_1_qq, rho1_2_qq], [rho2_1_qq, rho2_2_qq]])
            Gamma_qq_an[np.ix_([0],mesh.nodalDOF[a], mesh.nodalDOF[b])] += Gamma_qq
            theta_s_qq_an[np.ix_([0, 1], [0, 1], mesh.nodalDOF[a], mesh.nodalDOF[b])] += np.array([[theta1_1_qq, theta1_2_qq], [theta2_1_qq, theta2_2_qq]])

    print(f"error rho_q: {np.linalg.norm(rho_q_an - rho_q_num)}")
    print(f"error rho_s_q: {np.linalg.norm(rho_s_q_an - rho_s_q_num)}")
    print(f"error Gamma_q: {np.linalg.norm(Gamma_q_an - Gamma_q_num)}")
    print(f"error theta_a_q: {np.linalg.norm(theta_s_q_an - theta_s_q_num)}")

    print(f"error rho_qq: {np.linalg.norm(rho_qq_an - rho_qq_num)}")
    print(f"error rho_s_qq: {np.linalg.norm(rho_s_qq_an - rho_s_qq_num)}")
    print(f"error Gamma_qq: {np.linalg.norm(Gamma_qq_an - Gamma_qq_num)}")
    print(f"error theta_s_qq: {np.linalg.norm(theta_s_qq_an - theta_s_qq_num)}")

if __name__ == "__main__":
    # test_gradients()
    test_variations()