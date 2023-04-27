import numpy as np
import meshio
from cardillo.discretization.indexing import flat2D, flat3D, split2D, split3D
from cardillo.math import det2D, det3D
from cardillo.utility.coo_matrix import CooMatrix


class SecondGradient:
    def __init__(self, density, material, mesh, Z, z0=None, v0=None, cDOF=[], b=None):
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

        # store mesh and extract data
        self.mesh = mesh
        self.nel = mesh.nel
        self.nn = mesh.nn
        self.nn_el = mesh.nn_el  # number of nodes of an element
        self.nq_el = mesh.nq_el
        self.nqp = mesh.nqp
        self.elDOF = mesh.elDOF
        self.nodalDOF = mesh.nodalDOF
        self.N = self.mesh.N
        self.nq_n = mesh.nq_n

        self.dim = int(len(Z) / self.nn)
        if self.dim == 2:
            self.flat = flat2D
            self.determinant = det2D
        else:
            self.flat = flat3D
            self.determinant = det3D

        # for each Gauss point, compute kappa0^-1, N_X and w_J0 = w * det(kappa0^-1)
        if self.nq_n > 1:
            self.kappa0_xi_inv, self.N_X, self.w_J0 = self.mesh.reference_mappings(Z)
            # if self.dim == 3:
            #     self.srf_w_J0 = []
            #     for i in range(6):
            #         self.srf_w_J0.append(self.mesh.surface_mesh[i].reference_mappings(Z[self.mesh.surface_qDOF[i].ravel()]))

            self.N_XX = self.mesh.N_XX(Z, self.kappa0_xi_inv)
        # # For 1D Continua
        # else:
        #     self.w_J0 = self.mesh.reference_mappings(Z)
        #     for el in range(self.nel):
        #         for i in range(self.nqp):
        #             for a in range(self.nn_el):
        #                 kappa0_xixi =
        #             self.N_X[el, i] = mesh.N_xi[el, i] / self.w_J0[el, i]
        #             self.N_XX[el, i] = mesh.n_xixi[el ,i] - self.N_X[el ,i] *

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

    def F_qp(self, q):
        """Compute deformation gradient at quadrature points"""
        F = np.zeros((self.nel, self.nqp, self.dim, self.dim))
        for el in range(self.nel):
            qel = q[self.elDOF[el]]
            for i in range(self.nqp):
                for a in range(self.nn_el):
                    F[el, i] += np.outer(qel[self.nodalDOF[a]], self.N_X[el, i, a])
        self.F = F

    def G_qp(self, q):
        """Compute gradient of deformation gradient at quadrature points"""
        G = np.zeros((self.nel, self.nqp, self.dim, self.dim, self.dim))
        for el in range(self.nel):
            qel = q[self.elDOF[el]]
            for i in range(self.nqp):
                for a in range(self.nn_el):
                    G[el, i] += np.einsum(
                        "i,jk->ijk", qel[self.nodalDOF[a]], self.N_XX[el, i, a]
                    )
        self.G = G

    def pre_iteration_update(self, t, q, u):
        self.F_qp(self.z(t, q))
        self.G_qp(self.z(t, q))

    #########################################
    # kinematic equation
    #########################################
    def q_dot(self, t, q, u):
        return u

    def B(self, t, q):
        return np.ones(self.nq)

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

    def M(self, t, q):
        coo = CooMatrix((self.nu, self.nu))
        for el in range(self.nel):
            M_el = self.M_el(el)

            # sparse assemble element internal stiffness matrix
            elfDOF = self.elfDOF[el]
            eluDOF = self.eluDOF[el]
            coo[eluDOF, eluDOF] = M_el[elfDOF[:, None], elfDOF]

        return coo

    def f_pot_el(self, ze, el):
        f = np.zeros(self.nq_el)

        for i in range(self.nqp):
            N_X = self.N_X[el, i]
            N_XX = self.N_XX[el, i]
            w_J0 = self.w_J0[el, i]

            # F_eli = 0
            # G_eli = 0
            # for a in range(self.nn_el):
            #     G_eli += np.einsum('i,jk->ijk', ze[self.nodalDOF[a]], self.N_XX[el, i, a])
            #     F_eli += np.outer(ze[self.nodalDOF[a]], self.N_X[el, i, a])

            # P = self.mat.P(F_eli, G_eli)
            # bbP = self.mat.bbP(F_eli, G_eli)

            #  Piola-Lagrange stress tensor
            P = self.mat.P(self.F[el, i], self.G[el, i])
            # Piola-Lagrange double-stress tensor
            bbP = self.mat.bbP(self.F[el, i], self.G[el, i])

            # internal forces
            for a in range(self.nn_el):
                # TODO: reference
                f[self.nodalDOF[a]] -= (
                    P @ N_X[a] + np.einsum("ijk,jk->i", bbP, N_XX[a])
                ) * w_J0

        return f

    def h(self, t, q, u):
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
            N_XX = self.N_XX[el, i]
            w_J0 = self.w_J0[el, i]

            F_eli = self.F[el, i]
            G_eli = self.G[el, i]
            F_q = np.zeros((self.dim, self.dim, self.nq_el))
            G_q = np.zeros((self.dim, self.dim, self.dim, self.nq_el))
            for a in range(self.nn_el):
                # F += np.outer(ze[self.nodalDOF[a]], N_X[a]) # Bonet 1997 (7.5)
                F_q[:, :, self.nodalDOF[a]] += np.einsum("ik,j->ijk", I3, N_X[a])
                G_q[:, :, :, self.nodalDOF[a]] += np.einsum("il,jk->ijkl", I3, N_XX[a])

            # derivative of P and bbP w.r.t. q
            P_F = self.mat.P_F(F_eli, G_eli)
            P_G = self.mat.P_G(F_eli, G_eli)
            bbP_F = self.mat.bbP_F(F_eli, G_eli)
            bbP_G = self.mat.bbP_G(F_eli, G_eli)
            P_q = np.einsum("klmn,mnj->klj", P_F, F_q) + np.einsum(
                "klmnj,mnjo->klo", P_G, G_q
            )
            bbP_q = np.einsum("klomn,mnj->kloj", bbP_F, F_q) + np.einsum(
                "klomnp,mnpj->kloj", bbP_G, G_q
            )

            # internal element stiffness matrix
            for a in range(self.nn_el):
                Ke[self.nodalDOF[a]] += np.einsum(
                    "ijk,j->ik", P_q, -N_X[a] * w_J0
                ) + np.einsum("ijkl,jk->il", bbP_q, -N_XX[a] * w_J0)

        return Ke

    def h_q(self, t, q, u):
        # import time
        z = self.z(t, q)
        coo = CooMatrix((self.nu, self.nq))
        for el in range(self.nel):
            # t0 = time.time()
            Ke = self.f_pot_q_el(z[self.elDOF[el]], el)

            # sparse assemble element internal stiffness matrix
            elfDOF = self.elfDOF[el]
            eluDOF = self.eluDOF[el]
            elqDOF = self.elqDOF[el]
            coo[eluDOF, elqDOF] = Ke[elfDOF[:, None], elfDOF]

        return coo

    ####################################################
    # Line forces
    ####################################################
    def force_distr1D_el(self, force, t, el, edge_mesh, edge_w_J0el):
        fe = np.zeros(edge_mesh.nq_el)

        for i in range(edge_mesh.nqp):
            N = edge_mesh.N[el, i]
            # w_J0 = self.edge_w_J0[edge_mesh.idx][el, i] # Where to create?

            xi = edge_mesh.qp[el, i]

            # internal forces
            for a in range(edge_mesh.nn_el):
                fe[edge_mesh.nodalDOF[a]] += force(t, xi) * N[a] * edge_w_J0el[i]

        return fe

    def force_distr1D(self, t, q, force, edge_idx, edge_qDOF, edge_w_J0):
        z = self.z(t, q)
        f = np.zeros(self.nz)

        srf_idx = edge_idx // 4
        srf_edge_idx = edge_idx % 4

        edge_mesh = self.mesh.surface_mesh[srf_idx].edge_mesh[srf_edge_idx]

        for el in range(edge_mesh.nel):
            f[edge_qDOF[edge_mesh.elDOF[el]]] += self.force_distr1D_el(
                force, t, el, edge_mesh, edge_w_J0[el]
            )
        return f[self.fDOF]

    def force_distr1D_q(self, t, q, force, edge_idx, edge_w_J0):
        return None

    ####################################################
    # surface forces
    ####################################################
    def force_distr2D_el(self, force, t, el, srf_mesh, srf_w_J0el):
        fe = np.zeros(srf_mesh.nq_el)

        el_xi, el_eta = split2D(el, (srf_mesh.nel_xi,))

        for i in range(srf_mesh.nqp):
            Neli = srf_mesh.N[el, i]
            srf_w_J0eli = srf_w_J0el[i]

            i_xi, i_eta = split2D(i, (srf_mesh.nqp_xi,))
            xi = srf_mesh.qp_xi[el_xi, i_xi]
            eta = srf_mesh.qp_eta[el_eta, i_eta]

            # internal forces
            for a in range(srf_mesh.nn_el):
                fe[srf_mesh.nodalDOF[a]] += force(t, xi, eta) * Neli[a] * srf_w_J0eli

        return fe

    def force_distr2D(self, t, q, force, srf_idx, srf_w_J0):
        z = self.z(t, q)
        f = np.zeros(self.nz)

        srf_mesh = self.mesh.surface_mesh[srf_idx]
        srf_zDOF = self.mesh.surface_qDOF[srf_idx].ravel()

        for el in range(srf_mesh.nel):
            f[srf_zDOF[srf_mesh.elDOF[el]]] += self.force_distr2D_el(
                force, t, el, srf_mesh, srf_w_J0[el]
            )
        return f[self.fDOF]

    def force_distr2D_q(self, t, q, force, srf_idx):
        return None

    ####################################################
    # surface double forces
    ####################################################
    def doubleforce_distr2D_el(
        self, dforce, t, el, srf_mesh, srf_w_J0el, srf_Norm, Nb_Xel
    ):
        fe = np.zeros(self.mesh.nq_el)

        el_xi, el_eta = split2D(el, (srf_mesh.nel_xi,))

        for i in range(srf_mesh.nqp):
            Nb_Xeli = Nb_Xel[i]

            i_xi, i_eta = split2D(i, (srf_mesh.nqp_xi,))
            xi = srf_mesh.qp_xi[el_xi, i_xi]
            eta = srf_mesh.qp_eta[el_eta, i_eta]

            for a in range(self.mesh.nn_el):
                fe[self.mesh.nodalDOF[a]] += (
                    dforce(t, xi, eta) * Nb_Xeli[a, srf_Norm] * srf_w_J0el[i]
                )

            # internal forces
            # for a_tilde in range(srf_mesh.nn_el):
            #     fe[srf_mesh.nodalDOF[a_tilde]] += dfi * srf_w_J0el[i]

        return fe

    def doubleforce_distr2D(self, t, q, dforce, srf_idx, srf_w_J0, Nb_X):
        z = self.z(t, q)
        f = np.zeros(self.nz)

        srf_Norm = srf_idx // 2

        srf_mesh = self.mesh.surface_mesh[srf_idx]
        srf_zDOF = self.mesh.surface_qDOF[srf_idx].ravel()

        for el in range(srf_mesh.nel):
            elDOF_el = self.mesh.elDOF[self.mesh.bc_el[srf_idx][el]]
            f[elDOF_el] += self.doubleforce_distr2D_el(
                dforce, t, el, srf_mesh, srf_w_J0[el], srf_Norm, Nb_X[el]
            )
        return f[self.fDOF]

    def doubleforce_distr2D_q(self, t, q, doubleforce_distr2D, srf_idx, srf_w_J0):
        return None

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

    def force_distr3D_q(self, t, q, force):
        return None

    # TODO: Include global data in vtk file
    def post_processing_single_configuration(
        self, t, q, filename, binary=True, return_strain=False
    ):
        # compute redundant generalized coordinates
        z = self.z(t, q)

        # generalized coordinates, connectivity and polynomial degree
        cells, points, HigherOrderDegrees = self.mesh.vtk_mesh(z)

        # dictionary storing point data
        point_data = {}

        # evaluate deformation gradient at quadrature points
        F = np.zeros((self.mesh.nel, self.mesh.nqp, self.mesh.nq_n, self.mesh.nq_n))
        G = np.zeros(
            (
                self.mesh.nel,
                self.mesh.nqp,
                self.mesh.nq_n,
                self.mesh.nq_n,
                self.mesh.nq_n,
            )
        )
        We = np.zeros((self.mesh.nel, self.nqp))
        for el in range(self.mesh.nel):
            ze = z[self.mesh.elDOF[el]]
            for i in range(self.mesh.nqp):
                for a in range(self.mesh.nn_el):
                    F[el, i] += np.outer(
                        ze[self.mesh.nodalDOF[a]], self.N_X[el, i, a]
                    )  # Bonet 1997 (7.6b)
                    G[el, i] += np.einsum(
                        "i,jk->ijk", ze[self.nodalDOF[a]], self.N_XX[el, i, a]
                    )
                We[el, i] = self.mat.We(F[el, i], G[el, i])

        if return_strain == False:
            F_vtk = self.mesh.field_to_vtk(F)
            G_vtk = self.mesh.field_to_vtk(G)
            point_data.update({"F": F_vtk, "G": G_vtk})

            # field data vtk export
            # TODO: get export fields from material
            point_data_fields = {
                # "C": lambda F: F.T @ F,
                "J": lambda F, G: np.array([self.determinant(F)]),
                "P": lambda F, G: self.mat.P(F, G),
                "bbP": lambda F, G: self.mat.bbP(F, G),
                # "S": lambda F: self.mat.S(F),
                "W": lambda F, G: self.mat.W(F, G),
                "We": lambda F, G: self.mat.We(F, G),
                "Ws": lambda F, G: self.mat.Ws(F, G),
                # "Wc": lambda F, G: self.mat.Wc(F, G),
                # "Wg": lambda F, G: self.mat.Wg(F, G),
                # "Wn": lambda F, G: self.mat.Wn(F, G),
                # "Wt": lambda F, G: self.mat.Wt(F, G),
                # "rho": lambda F, G: self.mat.rho(F, G),
                # "ga": lambda F, G: self.mat.ga(F, G),
                # "tau": lambda F, G: self.mat.tau(F, G),
                # "kappa_n": lambda F, G: self.mat.kappa_n(F, G),
                # "kappa_g": lambda F, G: self.mat.kappa_g(F, G),
            }

            for name, fun in point_data_fields.items():
                tmp = fun(
                    F_vtk[0].reshape(self.dim, self.dim),
                    G_vtk[0].reshape(self.dim, self.dim, self.dim),
                ).ravel()
                field = np.zeros((len(F_vtk), len(tmp)))
                for i, Fi in enumerate(F_vtk):
                    field[i] = fun(
                        Fi.reshape(self.dim, self.dim),
                        G_vtk[i].reshape(self.dim, self.dim, self.dim),
                    ).ravel()
                point_data.update({name: field})

            # Global Values as Field Data
            field_data = {
                "W_tot": self.integrate(self.mat.W, F, G),
                "We_tot": self.integrate(self.mat.We, F, G),
                "Ws_tot": self.integrate(self.mat.Ws, F, G),
                # "Wn_tot": self.integrate(self.mat.Wn, F, G),
                # "Wg_tot": self.integrate(self.mat.Wg, F, G),
                # "Wt_tot": self.integrate(self.mat.Wt, F, G),
                # "Wc_tot": self.integrate(self.mat.Wc, F, G),
            }
            # write field data
            import pickle

            with open(
                filename.parent
                / str("field_data_" + str.split(filename.stem, "_")[-1]),
                mode="wb",
            ) as f:
                pickle.dump(field_data, f)
            print(field_data)
            # write vtk mesh using meshio
            meshio.write_points_cells(
                filename.parent / (filename.stem + ".vtu"),
                points,
                cells,
                point_data=point_data,
                cell_data={"HigherOrderDegrees": HigherOrderDegrees},
                field_data=field_data,
                binary=binary,
            )

        else:
            return F[0, 0]

    def post_processing(self, t, q, filename, binary=True, project_to_reference=False):
        # write paraview PVD file collecting time and all vtk files, see https://www.paraview.org/Wiki/ParaView/Data_formats#PVD_File_Format
        from xml.dom import minidom

        root = minidom.Document()

        vkt_file = root.createElement("VTKFile")
        vkt_file.setAttribute("type", "Collection")
        root.appendChild(vkt_file)

        collection = root.createElement("Collection")
        vkt_file.appendChild(collection)

        for i, (ti, qi) in enumerate(zip(t, q)):
            filei = filename.parent / (filename.stem + f"_{i}.vtu")

            # write time step and file name in pvd file
            dataset = root.createElement("DataSet")
            dataset.setAttribute("timestep", f"{ti:0.6f}")
            dataset.setAttribute("file", filei.name)
            collection.appendChild(dataset)

            if project_to_reference == True:
                raise NotImplementedError
            self.post_processing_single_configuration(ti, qi, filei, binary=binary)

        # write pvd file
        xml_str = root.toprettyxml(indent="\t")
        with (filename.parent / (filename.stem + ".pvd")).open("w") as f:
            f.write(xml_str)

    # Export for Lagrange multipliers
    def post_processing_la_single_configuration(
        self, t, la_mesh, la, q, z_range, filename, binary=True
    ):
        la_q = self.z(t, q)[z_range]

        # generalized coordinates, connectivity and polynomial degree
        cells, points, HigherOrderDegrees = la_mesh.vtk_mesh(la_q, dim=self.nq_n)

        # dictionary storing point data
        point_data = {}

        _, la_vtk, _ = la_mesh.vtk_mesh(la, dim=la_mesh.nq_n)
        point_data.update({"la": la_vtk})

        # write vtk mesh using meshio
        meshio.write_points_cells(
            filename.parent / (filename.stem + ".vtu"),
            points,
            cells,
            point_data=point_data,
            cell_data={"HigherOrderDegrees": HigherOrderDegrees},
            binary=binary,
        )

    def post_processing_la(
        self,
        t,
        la_mesh,
        la,
        q_la,
        z_range,
        filename,
        binary=True,
        project_to_reference=False,
    ):
        # write paraview PVD file collecting time and all vtk files, see https://www.paraview.org/Wiki/ParaView/Data_formats#PVD_File_Format
        from xml.dom import minidom

        root = minidom.Document()

        vkt_file = root.createElement("VTKFile")
        vkt_file.setAttribute("type", "Collection")
        root.appendChild(vkt_file)

        collection = root.createElement("Collection")
        vkt_file.appendChild(collection)

        for i, (ti, lai, q_lai) in enumerate(zip(t, la, q_la)):
            filei = filename.parent / (filename.stem + f"la_{i}.vtu")

            # write time step and file name in pvd file
            dataset = root.createElement("DataSet")
            dataset.setAttribute("timestep", f"{ti:0.6f}")
            dataset.setAttribute("file", filei.name)
            collection.appendChild(dataset)

            if project_to_reference == True:
                raise NotImplementedError
            self.post_processing_la_single_configuration(
                ti, la_mesh, lai, q_lai, z_range, filei, binary=binary
            )

        # write pvd file
        xml_str = root.toprettyxml(indent="\t")
        with (filename.parent / (filename.stem + "_la.pvd")).open("w") as f:
            f.write(xml_str)

    # Integration to obtain global values for export
    def integrate(self, var, F, G):
        # integrates variable over domain
        Var = 0.0
        for el in range(self.nel):
            for i in range(self.nqp):
                vari = var(F[el, i], G[el, i])
                # Neli = self.N[el, i]
                w_J0eli = self.w_J0[el, i]
                # for a in range(self.nq_n):
                Var += vari * w_J0eli

        return np.array([[Var]])

    # TODO: test functions
    # def test_gradient():
    #     from cardillo.discretization.mesh3D import Mesh3D, cube
    #     from cardillo.discretization.B_spline import Knot_vector, fit_B_spline_volume
    #     from cardillo.discretization.indexing import flat3D

    #     QP_shape = (1, 1, 1)
    #     degrees = (3, 3, 1)
    #     element_shape = (10, 10, 1)

    #     Xi = Knot_vector(degrees[0], element_shape[0])
    #     Eta = Knot_vector(degrees[1], element_shape[1])
    #     Zeta = Knot_vector(degrees[2], element_shape[2])
    #     knot_vectors = (Xi, Eta, Zeta)

    #     mesh = Mesh3D(knot_vectors, QP_shape, derivative_order=1, basis="B-spline", nq_n=3)

    #     def bending(xi, eta, zeta, phi0=np.pi / 2, R=1, B=1, H=1):
    #         phi = (1 - xi) * phi0
    #         x = (R + B * eta) * np.cos(phi)
    #         y = (R + B * eta) * np.sin(phi)
    #         z = zeta * H
    #         return x, y, z

    #     nxi, neta, nzeta = 15, 15, 5
    #     xi = np.linspace(0, 1, num=nxi)
    #     eta = np.linspace(0, 1, num=neta)
    #     zeta = np.linspace(0, 1, num=nzeta)

    #     phi0, R, B, H = np.pi / 2, 1, 1, 1

    #     n3 = nxi * neta * nzeta
    #     knots = np.zeros((n3, 3))
    #     Pw = np.zeros((n3, 3))
    #     for i, xii in enumerate(xi):
    #         for j, etai in enumerate(eta):
    #             for k, zetai in enumerate(zeta):
    #                 idx = flat3D(i, j, k, (nxi, neta, nzeta))
    #                 knots[idx] = xii, etai, zetai
    #                 Pw[idx] = bending(xii, etai, zetai, phi0=phi0, R=R, B=B, H=H)

    #     cDOF = np.array([], dtype=int)
    #     qc = np.array([], dtype=float).reshape((0, 3))
    #     x, y, z = fit_B_spline_volume(mesh, knots, Pw, qc, cDOF)

    #     L = 1
    #     cube_shape = (L, B, H)
    #     Q = cube(cube_shape, mesh, Greville=True)
    #     q0 = np.concatenate((x, y, z))
    #     continuum = First_gradient(None, mesh, Q, z0=q0)

    #     # import matplotlib.pyplot as plt
    #     # fig = plt.figure()
    #     # ax = fig.add_subplot(111, projection='3d')
    #     # # ax.scatter(*Pw.T, color='black')
    #     # ax.scatter(*Q.reshape(3, -1), color='blue')
    #     # ax.scatter(*q0.reshape(3, -1), color='red')
    #     # plt.show()

    #     # knots = np.array([[0.5, 0.5, 0.5]])
    #     # knots = np.array([[0.125, 0.35, 0.85]])
    #     knots = np.random.rand(3).reshape(1, 3)
    #     F_num = continuum.F(knots, q0)
    #     print(f"F_num({knots[0]}):\n{F_num[0]}")

    #     def F(xi, eta, zeta, phi0, R, B, L):
    #         r = phi0 * (R + B * eta) / L
    #         phi = (1 - xi) * phi0
    #         F = np.array(
    #             [
    #                 [r * np.sin(phi), np.cos(phi), 0],
    #                 [-r * np.cos(phi), np.sin(phi), 0],
    #                 [0, 0, 1],
    #             ]
    #         )
    #         return F

    #     F_an = F(*knots[0], phi0, R, B, L)
    #     print(f"F({knots[0]}):\n{F_an}")

    #     error = np.linalg.norm(F_num[0] - F_an)
    #     print(f"error: {error}")

    # def test_gradient_vtk_export():
    #     from cardillo.discretization.mesh3D import Mesh3D, cube
    #     from cardillo.discretization.B_spline import Knot_vector, fit_B_spline_volume
    #     from cardillo.discretization.indexing import flat3D

    #     QP_shape = (5, 5, 5)
    #     degrees = (3, 3, 1)
    #     element_shape = (5, 5, 1)

    #     Xi = Knot_vector(degrees[0], element_shape[0])
    #     Eta = Knot_vector(degrees[1], element_shape[1])
    #     Zeta = Knot_vector(degrees[2], element_shape[2])
    #     knot_vectors = (Xi, Eta, Zeta)

    #     mesh = Mesh3D(knot_vectors, QP_shape, derivative_order=1, basis="B-spline", nq_n=3)

    #     # reference configuration is a cube
    #     phi0 = np.pi / 2
    #     R = 1
    #     B = 1
    #     H = 1
    #     # L = (R + B / 2) * phi0
    #     L = (R) * phi0
    #     cube_shape = (L, B, H)
    #     Q = cube(cube_shape, mesh, Greville=True)

    #     # 3D continuum
    #     continuum = First_gradient(None, mesh, Q, z0=Q)

    #     # fit quater circle configuration
    #     def bending(xi, eta, zeta, phi0, R, B, H):
    #         phi = (1 - xi) * phi0
    #         x = (R + B * eta) * np.cos(phi)
    #         y = (R + B * eta) * np.sin(phi)
    #         z = zeta * H
    #         return x, y, z

    #     nxi, neta, nzeta = 15, 15, 5
    #     xi = np.linspace(0, 1, num=nxi)
    #     eta = np.linspace(0, 1, num=neta)
    #     zeta = np.linspace(0, 1, num=nzeta)

    #     n3 = nxi * neta * nzeta
    #     knots = np.zeros((n3, 3))
    #     Pw = np.zeros((n3, 3))
    #     for i, xii in enumerate(xi):
    #         for j, etai in enumerate(eta):
    #             for k, zetai in enumerate(zeta):
    #                 idx = flat3D(i, j, k, (nxi, neta, nzeta))
    #                 knots[idx] = xii, etai, zetai
    #                 Pw[idx] = bending(xii, etai, zetai, phi0=phi0, R=R, B=B, H=H)

    #     cDOF = np.array([], dtype=int)
    #     qc = np.array([], dtype=float).reshape((0, 3))
    #     x, y, z = fit_B_spline_volume(mesh, knots, Pw, qc, cDOF)
    #     q = np.concatenate((x, y, z))

    #     # export current configuration and deformation gradient on quadrature points to paraview
    #     continuum.post_processing(q, "test.vtu")

    # def test_internal_forces():
    # from cardillo.discretization.mesh3D import Mesh3D, cube
    # from cardillo.discretization.B_spline import Knot_vector, fit_B_spline_volume
    # from cardillo.discretization.indexing import flat3D
    # from cardillo.model.continuum import Ogden1997_compressible

    # QP_shape = (2, 2, 2)
    # degrees = (2, 2, 2)
    # element_shape = (4, 4, 2)

    # Xi = Knot_vector(degrees[0], element_shape[0])
    # Eta = Knot_vector(degrees[1], element_shape[1])
    # Zeta = Knot_vector(degrees[2], element_shape[2])
    # knot_vectors = (Xi, Eta, Zeta)

    # mesh = Mesh3D(knot_vectors, QP_shape, derivative_order=1, basis="B-spline", nq_n=3)

    # # reference configuration is a cube
    # phi0 = np.pi / 2
    # R = 1
    # B = 1
    # H = 1
    # # L = (R + B / 2) * phi0
    # L = (R) * phi0
    # cube_shape = (L, B, H)
    # Z = cube(cube_shape, mesh, Greville=True)

    # # material model
    # mu1 = 0.3
    # mu2 = 0.5
    # mat = Ogden1997_compressible(mu1, mu2)

    # # 3D continuum
    # # cDOF = []
    # # b = lambda t: np.array([], dtype=float)
    # cDOF1 = mesh.surface_DOF[0].reshape(-1)
    # cDOF2 = mesh.surface_DOF[1][2]
    # cDOF = np.concatenate((cDOF1, cDOF2))
    # b1 = lambda t: Z[cDOF1]
    # b2 = lambda t: Z[cDOF2] + t * 0.25
    # b = lambda t: np.concatenate((b1(t), b2(t)))

    # continuum = First_gradient(mat, mesh, Z, z0=Z, cDOF=cDOF, b=b)

    # from cardillo.model import Model

    # model = Model()
    # model.add(continuum)
    # model.assemble()

    # # evaluate internal forces in reference configuration
    # Q = Z[continuum.fDOF]
    # # f_pot = continuum.f_pot(0, Q)
    # f_pot = model.f_pot(0, Q)
    # # print(f'f_pot:\n{f_pot}')
    # print(f"f_pot.shape: {f_pot.shape}")
    # print(f"len(cDOF): {len(cDOF)}")
    # print(f"len(b(0)): {len(b(0))}")

    # f_pot_q = model.f_pot_q(0, Q)
    # # print(f'f_pot_q:\n{f_pot_q.toarray()}')
    # print(f"f_pot_q.shape:\n{f_pot_q.toarray().shape}")

    # # export current configuration and deformation gradient on quadrature points to paraview
    # continuum.post_processing(0.137, Q, "test.vtu")


if __name__ == "__main__":
    # test_gradient()
    # test_gradient_vtk_export()
    test_internal_forces()
