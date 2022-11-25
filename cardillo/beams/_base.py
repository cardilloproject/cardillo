import numpy as np
from abc import ABC, abstractmethod

from cardillo.utility.coo import Coo
from cardillo.discretization.bezier import L2_projection_Bezier_curve

from cardillo.math import (
    cross3,
    ax2skew,
)

# # Timoshenko + Petrov-Galerkin:
# # - *args for rod formulation
# def make_rod(material_model, cross_section, RodFormulation):
#     class Rod:
#         pass


class TimoshenkoPetrovGalerkinBase(ABC):
    def __init__(self):
        self.radius = 0.125
        self.a = 0.1
        self.b = 0.2

        # self.cross_section = "circle"
        # self.cross_section = "circle_wedge"
        self.cross_section = "rectangle"

    @abstractmethod
    def r_OP(self, t, q, frame_ID, K_r_SP):
        ...

    @abstractmethod
    def A_IK(self, t, q, frame_ID):
        ...

    def assembler_callback(self):
        if self.constant_mass_matrix:
            self._M_coo()

    #########################################
    # equations of motion
    #########################################
    def assembler_callback(self):
        if self.constant_mass_matrix:
            self._M_coo()

    def M_el_constant(self, el):
        M_el = np.zeros((self.nq_element, self.nq_element), dtype=float)

        for i in range(self.nquadrature):
            # extract reference state variables
            qwi = self.qw[el, i]
            Ji = self.J[el, i]

            # delta_r A_rho0 r_ddot part
            M_el_r_r = np.eye(3) * self.A_rho0 * Ji * qwi
            for node_a in range(self.nnodes_element_r):
                nodalDOF_a = self.nodalDOF_element_r[node_a]
                for node_b in range(self.nnodes_element_r):
                    nodalDOF_b = self.nodalDOF_element_r[node_b]
                    M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_r_r * (
                        self.N_r[el, i, node_a] * self.N_r[el, i, node_b]
                    )

            # first part delta_phi (I_rho0 omega_dot + omega_tilde I_rho0 omega)
            M_el_psi_psi = self.K_I_rho0 * Ji * qwi
            for node_a in range(self.nnodes_element_psi):
                nodalDOF_a = self.nodalDOF_element_psi[node_a]
                for node_b in range(self.nnodes_element_psi):
                    nodalDOF_b = self.nodalDOF_element_psi[node_b]
                    M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_psi_psi * (
                        self.N_psi[el, i, node_a] * self.N_psi[el, i, node_b]
                    )

        return M_el

    def M_el(self, qe, el):
        M_el = np.zeros((self.nq_element, self.nq_element), dtype=float)

        for i in range(self.nquadrature):
            # extract reference state variables
            qwi = self.qw[el, i]
            Ji = self.J[el, i]

            # delta_r A_rho0 r_ddot part
            M_el_r_r = np.eye(3) * self.A_rho0 * Ji * qwi
            for node_a in range(self.nnodes_element_r):
                nodalDOF_a = self.nodalDOF_element_r[node_a]
                for node_b in range(self.nnodes_element_r):
                    nodalDOF_b = self.nodalDOF_element_r[node_b]
                    M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_r_r * (
                        self.N_r[el, i, node_a] * self.N_r[el, i, node_b]
                    )

            # first part delta_phi (I_rho0 omega_dot + omega_tilde I_rho0 omega)
            M_el_psi_psi = self.K_I_rho0 * Ji * qwi
            for node_a in range(self.nnodes_element_psi):
                nodalDOF_a = self.nodalDOF_element_psi[node_a]
                for node_b in range(self.nnodes_element_psi):
                    nodalDOF_b = self.nodalDOF_element_psi[node_b]
                    M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_psi_psi * (
                        self.N_psi[el, i, node_a] * self.N_psi[el, i, node_b]
                    )

            # For non symmetric cross sections there are also other parts
            # involved in the mass matrix. These parts are configuration
            # dependent and lead to configuration dependent mass matrix.
            _, A_IK, _, _ = self.eval(qe, self.qp[el, i])
            M_el_r_psi = A_IK @ self.K_S_rho0 * Ji * qwi
            M_el_psi_r = A_IK @ self.K_S_rho0 * Ji * qwi

            for node_a in range(self.nnodes_element_r):
                nodalDOF_a = self.nodalDOF_element_r[node_a]
                for node_b in range(self.nnodes_element_psi):
                    nodalDOF_b = self.nodalDOF_element_psi[node_b]
                    M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_r_psi * (
                        self.N_r[el, i, node_a] * self.N_psi[el, i, node_b]
                    )
            for node_a in range(self.nnodes_element_psi):
                nodalDOF_a = self.nodalDOF_element_psi[node_a]
                for node_b in range(self.nnodes_element_r):
                    nodalDOF_b = self.nodalDOF_element_r[node_b]
                    M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_psi_r * (
                        self.N_psi[el, i, node_a] * self.N_r[el, i, node_b]
                    )

        return M_el

    def _M_coo(self):
        self.__M = Coo((self.nu, self.nu))
        for el in range(self.nelement):
            # extract element degrees of freedom
            elDOF = self.elDOF[el]

            # sparse assemble element mass matrix
            self.__M.extend(
                self.M_el_constant(el), (self.uDOF[elDOF], self.uDOF[elDOF])
            )

    def M(self, t, q, coo):
        if self.constant_mass_matrix:
            coo.extend_sparse(self.__M)
        else:
            for el in range(self.nelement):
                # extract element degrees of freedom
                elDOF = self.elDOF[el]

                # sparse assemble element mass matrix
                coo.extend(
                    self.M_el(q[elDOF], el), (self.uDOF[elDOF], self.uDOF[elDOF])
                )

    def f_gyr_el(self, t, qe, ue, el):
        f_gyr_el = np.zeros(self.nq_element, dtype=np.common_type(qe, ue))

        for i in range(self.nquadrature):
            # interpoalte angular velocity
            K_Omega = np.zeros(3, dtype=ue.dtype)
            for node in range(self.nnodes_element_psi):
                K_Omega += self.N_psi[el, i, node] * ue[self.nodalDOF_element_psi[node]]

            # vector of gyroscopic forces
            f_gyr_el_psi = (
                cross3(K_Omega, self.K_I_rho0 @ K_Omega)
                * self.J[el, i]
                * self.qw[el, i]
            )

            # multiply vector of gyroscopic forces with nodal virtual rotations
            for node in range(self.nnodes_element_psi):
                f_gyr_el[self.nodalDOF_element_psi[node]] += (
                    self.N_psi[el, i, node] * f_gyr_el_psi
                )

        return f_gyr_el

    def f_gyr(self, t, q, u):
        f_gyr = np.zeros(self.nu, dtype=float)
        for el in range(self.nelement):
            f_gyr[self.elDOF[el]] += self.f_gyr_el(
                t, q[self.elDOF[el]], u[self.elDOF[el]], el
            )
        return f_gyr

    def f_gyr_u_el(self, t, qe, ue, el):
        f_gyr_u_el = np.zeros((self.nq_element, self.nq_element), dtype=float)

        for i in range(self.nquadrature):
            # interpoalte angular velocity
            K_Omega = np.zeros(3, dtype=float)
            for node in range(self.nnodes_element_psi):
                K_Omega += self.N_psi[el, i, node] * ue[self.nodalDOF_element_psi[node]]

            # derivative of vector of gyroscopic forces
            f_gyr_u_el_psi = (
                ((ax2skew(K_Omega) @ self.K_I_rho0 - ax2skew(self.K_I_rho0 @ K_Omega)))
                * self.J[el, i]
                * self.qw[el, i]
            )

            # multiply derivative of gyroscopic force vector with nodal virtual rotations
            for node_a in range(self.nnodes_element_psi):
                nodalDOF_a = self.nodalDOF_element_psi[node_a]
                for node_b in range(self.nnodes_element_psi):
                    nodalDOF_b = self.nodalDOF_element_psi[node_b]
                    f_gyr_u_el[nodalDOF_a[:, None], nodalDOF_b] += f_gyr_u_el_psi * (
                        self.N_psi[el, i, node_a] * self.N_psi[el, i, node_b]
                    )

        return f_gyr_u_el

    def f_gyr_u(self, t, q, u, coo):
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            f_gyr_u_el = self.f_gyr_u_el(t, q[elDOF], u[elDOF], el)
            coo.extend(f_gyr_u_el, (self.uDOF[elDOF], self.uDOF[elDOF]))

    # #########################################
    # # equations of motion
    # #########################################
    # def M_el_constant(self, el):
    #     M_el = np.zeros((self.nq_element, self.nq_element), dtype=float)

    #     for i in range(self.nquadrature):
    #         # extract reference state variables
    #         qwi = self.qw[el, i]
    #         Ji = self.J[el, i]

    #         # delta_r A_rho0 r_ddot part
    #         M_el_r_r = np.eye(3) * self.A_rho0 * Ji * qwi
    #         for node_a in range(self.nnodes_element):
    #             nodalDOF_a = self.nodalDOF_element_r[node_a]
    #             for node_b in range(self.nnodes_element):
    #                 nodalDOF_b = self.nodalDOF_element_r[node_b]
    #                 M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_r_r * (
    #                     self.N[el, i, node_a] * self.N[el, i, node_b]
    #                 )

    #         # first part delta_phi (I_rho0 omega_dot + omega_tilde I_rho0 omega)
    #         M_el_psi_psi = self.K_I_rho0 * Ji * qwi
    #         for node_a in range(self.nnodes_element):
    #             nodalDOF_a = self.nodalDOF_element_psi[node_a]
    #             for node_b in range(self.nnodes_element):
    #                 nodalDOF_b = self.nodalDOF_element_psi[node_b]
    #                 M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_psi_psi * (
    #                     self.N[el, i, node_a] * self.N[el, i, node_b]
    #                 )

    #     return M_el

    # def M_el(self, qe, el):
    #     M_el = np.zeros((self.nq_element, self.nq_element), dtype=float)

    #     for i in range(self.nquadrature):
    #         # extract reference state variables
    #         qwi = self.qw[el, i]
    #         Ji = self.J[el, i]

    #         # delta_r A_rho0 r_ddot part
    #         M_el_r_r = np.eye(3) * self.A_rho0 * Ji * qwi
    #         for node_a in range(self.nnodes_element):
    #             nodalDOF_a = self.nodalDOF_element_r[node_a]
    #             for node_b in range(self.nnodes_element):
    #                 nodalDOF_b = self.nodalDOF_element_r[node_b]
    #                 M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_r_r * (
    #                     self.N[el, i, node_a] * self.N[el, i, node_b]
    #                 )

    #         # first part delta_phi (I_rho0 omega_dot + omega_tilde I_rho0 omega)
    #         M_el_psi_psi = self.K_I_rho0 * Ji * qwi
    #         for node_a in range(self.nnodes_element):
    #             nodalDOF_a = self.nodalDOF_element_psi[node_a]
    #             for node_b in range(self.nnodes_element):
    #                 nodalDOF_b = self.nodalDOF_element_psi[node_b]
    #                 M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_psi_psi * (
    #                     self.N[el, i, node_a] * self.N[el, i, node_b]
    #                 )

    #         # For non symmetric cross sections there are also other parts
    #         # involved in the mass matrix. These parts are configuration
    #         # dependent and lead to configuration dependent mass matrix.
    #         _, A_IK, _, _ = self.eval(qe, self.qp[el, i])
    #         M_el_r_psi = A_IK @ self.K_S_rho0 * Ji * qwi
    #         M_el_psi_r = A_IK @ self.K_S_rho0 * Ji * qwi

    #         for node_a in range(self.nnodes_element):
    #             nodalDOF_a = self.nodalDOF_element_r[node_a]
    #             for node_b in range(self.nnodes_element):
    #                 nodalDOF_b = self.nodalDOF_element_psi[node_b]
    #                 M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_r_psi * (
    #                     self.N[el, i, node_a] * self.N[el, i, node_b]
    #                 )
    #         for node_a in range(self.nnodes_element):
    #             nodalDOF_a = self.nodalDOF_element_psi[node_a]
    #             for node_b in range(self.nnodes_element):
    #                 nodalDOF_b = self.nodalDOF_element_r[node_b]
    #                 M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_psi_r * (
    #                     self.N[el, i, node_a] * self.N[el, i, node_b]
    #                 )

    #     return M_el

    # def _M_coo(self):
    #     self._M = Coo((self.nu, self.nu))
    #     for el in range(self.nelement):
    #         # extract element degrees of freedom
    #         elDOF = self.elDOF[el]

    #         # sparse assemble element mass matrix
    #         self._M.extend(self.M_el_constant(el), (self.uDOF[elDOF], self.uDOF[elDOF]))

    # # TODO: Compute derivative of mass matrix for non constant mass matrix case!
    # def M(self, t, q, coo):
    #     if self.constant_mass_matrix:
    #         coo.extend_sparse(self._M)
    #     else:
    #         for el in range(self.nelement):
    #             # extract element degrees of freedom
    #             elDOF = self.elDOF[el]

    #             # sparse assemble element mass matrix
    #             coo.extend(
    #                 self.M_el(q[elDOF], el), (self.uDOF[elDOF], self.uDOF[elDOF])
    #             )

    # def f_gyr_el(self, t, qe, ue, el):
    #     f_gyr_el = np.zeros(self.nq_element, dtype=np.common_type(qe, ue))

    #     for i in range(self.nquadrature):
    #         # interpoalte angular velocity
    #         K_Omega = np.zeros(3, dtype=float)
    #         for node in range(self.nnodes_element):
    #             K_Omega += self.N[el, i, node] * ue[self.nodalDOF_element_psi[node]]

    #         # vector of gyroscopic forces
    #         f_gyr_el_psi = (
    #             cross3(K_Omega, self.K_I_rho0 @ K_Omega)
    #             * self.J[el, i]
    #             * self.qw[el, i]
    #         )

    #         # multiply vector of gyroscopic forces with nodal virtual rotations
    #         for node in range(self.nnodes_element):
    #             f_gyr_el[self.nodalDOF_element_psi[node]] += (
    #                 self.N[el, i, node] * f_gyr_el_psi
    #             )

    #     return f_gyr_el

    # def f_gyr_u_el(self, t, qe, ue, el):
    #     f_gyr_u_el = np.zeros((self.nq_element, self.nq_element), dtype=float)

    #     for i in range(self.nquadrature):
    #         # interpoalte angular velocity
    #         K_Omega = np.zeros(3, dtype=float)
    #         for node in range(self.nnodes_element):
    #             K_Omega += self.N[el, i, node] * ue[self.nodalDOF_element_psi[node]]

    #         # derivative of vector of gyroscopic forces
    #         f_gyr_u_el_psi = (
    #             ((ax2skew(K_Omega) @ self.K_I_rho0 - ax2skew(self.K_I_rho0 @ K_Omega)))
    #             * self.J[el, i]
    #             * self.qw[el, i]
    #         )

    #         # multiply derivative of gyroscopic force vector with nodal virtual rotations
    #         for node_a in range(self.nnodes_element):
    #             nodalDOF_a = self.nodalDOF_element_psi[node_a]
    #             for node_b in range(self.nnodes_element):
    #                 nodalDOF_b = self.nodalDOF_element_psi[node_b]
    #                 f_gyr_u_el[nodalDOF_a[:, None], nodalDOF_b] += f_gyr_u_el_psi * (
    #                     self.N[el, i, node_a] * self.N[el, i, node_b]
    #                 )

    #     return f_gyr_u_el

    ############
    # vtk export
    ############
    def export(self, sol_i, **kwargs):
        q = sol_i.q

        level = kwargs["level"]

        if "num" in kwargs:
            num = kwargs["num"]
        else:
            num = self.nelement * 4

        r_OPs, d1s, d2s, d3s = self.frames(q, n=num)

        if level == "centerline + directors":
            #######################################
            # simple export of points and directors
            #######################################
            r_OPs, d1s, d2s, d3s = self.frames(q, n=num)

            vtk_points = r_OPs.T

            cells = [("line", [[i, i + 1] for i in range(num - 1)])]

            cell_data = {}

            point_data = {
                "d1": d1s.T,
                "d2": d2s.T,
                "d3": d3s.T,
            }

        elif level == "volume":
            ################################
            # project on cubic Bezier volume
            ################################
            if "n_segments" in kwargs:
                n_segments = kwargs["n_segments"]
            else:
                n_segments = self.nelement

            r_OPs, d1s, d2s, d3s = self.frames(q, n=num)
            target_points_centerline = r_OPs.T

            # create points of the target curves (three characteristic points
            # of the cross section)
            if self.cross_section == "circle":
                target_points_0 = target_points_centerline
                target_points_1 = np.array(
                    [
                        r_OP + d2 * self.radius
                        for i, (r_OP, d2) in enumerate(zip(r_OPs.T, d2s.T))
                    ]
                )
                target_points_2 = np.array(
                    [
                        r_OP + d3 * self.radius
                        for i, (r_OP, d3) in enumerate(zip(r_OPs.T, d3s.T))
                    ]
                )
            elif self.cross_section == "rectangle":
                target_points_0 = target_points_centerline
                target_points_1 = np.array(
                    [
                        r_OP + d2 * self.a
                        for i, (r_OP, d2) in enumerate(zip(r_OPs.T, d2s.T))
                    ]
                )
                target_points_2 = np.array(
                    [
                        r_OP + d3 * self.b
                        for i, (r_OP, d3) in enumerate(zip(r_OPs.T, d3s.T))
                    ]
                )
            elif self.cross_section == "circle_wedge":
                ri = self.radius
                ru = 2 * ri
                a = 2 * np.sqrt(3) * ri

                target_points_0 = np.array(
                    [r_OP - ri * d3 for i, (r_OP, d3) in enumerate(zip(r_OPs.T, d3s.T))]
                )

                target_points_1 = np.array(
                    [
                        r_OP + d2 * a / 2 - ri * d3
                        for i, (r_OP, d2, d3) in enumerate(zip(r_OPs.T, d2s.T, d3s.T))
                    ]
                )

                target_points_2 = np.array(
                    [r_OP + d3 * ru for i, (r_OP, d3) in enumerate(zip(r_OPs.T, d3s.T))]
                )
            else:
                raise NotImplementedError

            # project target points on cubic C1 Bézier curve
            _, _, points_segments_0 = L2_projection_Bezier_curve(
                target_points_0, n_segments, case="C1"
            )
            _, _, points_segments_1 = L2_projection_Bezier_curve(
                target_points_1, n_segments, case="C1"
            )
            _, _, points_segments_2 = L2_projection_Bezier_curve(
                target_points_2, n_segments, case="C1"
            )

            if self.cross_section == "circle":

                def compute_missing_points(segment, layer):
                    P2 = points_segments_2[segment, layer]
                    P1 = points_segments_1[segment, layer]
                    P8 = points_segments_0[segment, layer]
                    P0 = 2 * P8 - P2
                    P3 = 2 * P8 - P1
                    P4 = (P0 + P1) - P8
                    P5 = (P1 + P2) - P8
                    P6 = (P2 + P3) - P8
                    P7 = (P3 + P0) - P8

                    dim = len(P0)
                    s22 = np.sqrt(2) / 2
                    points_weights = np.zeros((9, dim + 1))
                    points_weights[0] = np.array([*P0, 1])
                    points_weights[1] = np.array([*P1, 1])
                    points_weights[2] = np.array([*P2, 1])
                    points_weights[3] = np.array([*P3, 1])
                    points_weights[4] = np.array([*P4, s22])
                    points_weights[5] = np.array([*P5, s22])
                    points_weights[6] = np.array([*P6, s22])
                    points_weights[7] = np.array([*P7, s22])
                    points_weights[8] = np.array([*P8, 1])

                    return points_weights

                # create correct VTK ordering, see
                # https://coreform.com/papers/implementation-of-rational-bezier-cells-into-VTK-report.pdf:
                vtk_points_weights = []
                for i in range(n_segments):
                    # compute all missing points of the layer
                    points_layer0 = compute_missing_points(i, 0)
                    points_layer1 = compute_missing_points(i, 1)
                    points_layer2 = compute_missing_points(i, 2)
                    points_layer3 = compute_missing_points(i, 3)

                    #######################
                    # 1. vertices (corners)
                    #######################

                    # bottom
                    for j in range(4):
                        vtk_points_weights.append(points_layer0[j])

                    # top
                    for j in range(4):
                        vtk_points_weights.append(points_layer3[j])

                    ##########
                    # 2. edges
                    ##########

                    # bottom
                    for j in range(4, 8):
                        vtk_points_weights.append(points_layer0[j])

                    # top
                    for j in range(4, 8):
                        vtk_points_weights.append(points_layer3[j])

                    # first and second
                    for j in [0, 1, 3, 2]:
                        vtk_points_weights.append(points_layer1[j])
                        vtk_points_weights.append(points_layer2[j])

                    ##########
                    # 3. faces
                    ##########

                    # first and second
                    for j in [7, 5, 4, 6]:
                        vtk_points_weights.append(points_layer1[j])
                        vtk_points_weights.append(points_layer2[j])

                    # bottom and top
                    vtk_points_weights.append(points_layer0[0])
                    vtk_points_weights.append(points_layer3[-1])

                    ############
                    # 3. volumes
                    ############

                    # first and second
                    vtk_points_weights.append(points_layer1[0])
                    vtk_points_weights.append(points_layer2[-1])

            elif self.cross_section == "rectangle":

                def compute_missing_points(segment, layer):
                    Q0 = points_segments_0[segment, layer]
                    Q1 = points_segments_1[segment, layer]
                    Q2 = points_segments_2[segment, layer]
                    P0 = Q0 - (Q2 - Q0) - (Q1 - Q0)
                    P1 = Q0 - (Q2 - Q0) + (Q1 - Q0)
                    P2 = Q0 + (Q2 - Q0) + (Q1 - Q0)
                    P3 = Q0 + (Q2 - Q0) - (Q1 - Q0)

                    dim = len(P0)
                    points_weights = np.zeros((4, dim + 1))
                    points_weights[0] = np.array([*P0, 1])
                    points_weights[1] = np.array([*P1, 1])
                    points_weights[2] = np.array([*P2, 1])
                    points_weights[3] = np.array([*P3, 1])

                    return points_weights

                # create correct VTK ordering, see
                # https://coreform.com/papers/implementation-of-rational-bezier-cells-into-VTK-report.pdf:
                vtk_points_weights = []
                for i in range(n_segments):
                    # compute all missing points of the layer
                    points_layer0 = compute_missing_points(i, 0)
                    points_layer1 = compute_missing_points(i, 1)
                    points_layer2 = compute_missing_points(i, 2)
                    points_layer3 = compute_missing_points(i, 3)

                    #######################
                    # 1. vertices (corners)
                    #######################

                    # bottom
                    for j in range(4):
                        vtk_points_weights.append(points_layer0[j])

                    # top
                    for j in range(4):
                        vtk_points_weights.append(points_layer3[j])

                    ##########
                    # 2. edges
                    ##########

                    # first and second
                    for j in [0, 1, 3, 2]:
                        vtk_points_weights.append(points_layer1[j])
                        vtk_points_weights.append(points_layer2[j])

            elif self.cross_section == "circle_wedge":

                def compute_missing_points(segment, layer):
                    P0 = points_segments_0[segment, layer]
                    P3 = points_segments_1[segment, layer]
                    P4 = points_segments_2[segment, layer]

                    P5 = 2 * P0 - P3
                    P1 = 0.5 * (P3 + P4)
                    P0 = 0.5 * (P5 + P3)
                    P2 = 0.5 * (P4 + P5)

                    dim = len(P0)
                    points_weights = np.zeros((6, dim + 1))
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
                for i in range(n_segments):
                    # compute all missing points of the layer
                    points_layer0 = compute_missing_points(i, 0)
                    points_layer1 = compute_missing_points(i, 1)
                    points_layer2 = compute_missing_points(i, 2)
                    points_layer3 = compute_missing_points(i, 3)

                    #######################
                    # 1. vertices (corners)
                    #######################

                    # bottom
                    for j in range(3):
                        vtk_points_weights.append(points_layer0[j])

                    # top
                    for j in range(3):
                        vtk_points_weights.append(points_layer3[j])

                    ##########
                    # 2. edges
                    ##########

                    # bottom
                    for j in range(3, 6):
                        vtk_points_weights.append(points_layer0[j])

                    # top
                    for j in range(3, 6):
                        vtk_points_weights.append(points_layer3[j])

                    # first and second
                    for j in range(3):
                        vtk_points_weights.append(points_layer1[j])
                        vtk_points_weights.append(points_layer2[j])

                    ##########
                    # 3. faces
                    ##########

                    # first and second
                    for j in range(3, 6):
                        vtk_points_weights.append(points_layer1[j])
                        vtk_points_weights.append(points_layer2[j])

            p_zeta = 3
            if self.cross_section == "circle":
                n_layer = 9
                n_cell = (p_zeta + 1) * n_layer

                higher_order_degrees = [
                    (np.array([2, 2, p_zeta]),) for _ in range(n_segments)
                ]

                cells = [
                    (
                        "VTK_BEZIER_HEXAHEDRON",
                        np.arange(i * n_cell, (i + 1) * n_cell)[None],
                    )
                    for i in range(n_segments)
                ]
            elif self.cross_section == "rectangle":
                n_layer = 4
                n_cell = (p_zeta + 1) * n_layer

                higher_order_degrees = [
                    (np.array([1, 1, p_zeta]),) for _ in range(n_segments)
                ]

                cells = [
                    (
                        "VTK_BEZIER_HEXAHEDRON",
                        np.arange(i * n_cell, (i + 1) * n_cell)[None],
                    )
                    for i in range(n_segments)
                ]
            elif self.cross_section == "circle_wedge":
                n_layer = 6
                n_cell = (p_zeta + 1) * n_layer

                higher_order_degrees = [
                    (np.array([2, 2, p_zeta]),) for _ in range(n_segments)
                ]

                cells = [
                    ("VTK_BEZIER_WEDGE", np.arange(i * n_cell, (i + 1) * n_cell)[None])
                    for i in range(n_segments)
                ]

            vtk_points_weights = np.array(vtk_points_weights)
            vtk_points = vtk_points_weights[:, :3]

            point_data = {
                "RationalWeights": vtk_points_weights[:, 3],
            }

            cell_data = {
                "HigherOrderDegrees": higher_order_degrees,
            }

        else:
            raise NotImplementedError

        return vtk_points, cells, point_data, cell_data
