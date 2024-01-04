import numpy as np


class Force_line_distributed:
    r"""Line distributed Force for rods"""

    def __init__(self, force, rod):
        if not callable(force):
            self.force = lambda t, xi: force
        else:
            self.force = force
        self.rod = rod

    def assembler_callback(self):
        self.qDOF = self.rod.qDOF
        self.uDOF = self.rod.uDOF

        # ####################################################
        # # body force
        # ####################################################
        # def distributed_force1D_pot_el(self, force, t, qe, el):
        #     Ve = 0.0

        #     for i in range(self.nquadrature):
        #         # extract reference state variables
        #         qpi = self.qp[el, i]
        #         qwi = self.qw[el, i]
        #         Ji = self.J[el, i]

        #         # interpolate centerline position
        #         r_C = np.zeros(3, dtype=float)
        #         for node in range(self.nnodes_element_r):
        #             r_C += self.N_r[el, i, node] * qe[self.nodalDOF_element_r[node]]

        #         # compute potential value at given quadrature point
        #         Ve -= (r_C @ force(t, qpi)) * Ji * qwi

        #     # for i in range(self.nquadrature_dyn):
        #     #     # extract reference state variables
        #     #     qpi = self.qp_dyn[el, i]
        #     #     qwi = self.qw_dyn[el, i]
        #     #     Ji = self.J_dyn[el, i]

        #     #     # interpolate centerline position
        #     #     r_C = np.zeros(3, dtype=float)
        #     #     for node in range(self.nnodes_element_r):
        #     #         r_C += self.N_r_dyn[el, i, node] * qe[self.nodalDOF_element_r[node]]

        #     #     # compute potential value at given quadrature point
        #     #     Ve -= (r_C @ force(t, qpi)) * Ji * qwi

        #     return Ve

        # def distributed_force1D_pot(self, t, q, force):
        #     V = 0
        #     for el in range(self.nelement):
        #         qe = q[self.elDOF[el]]
        #         V += self.distributed_force1D_pot_el(force, t, qe, el)
        #     return V

    def h_el(self, t, el):
        he = np.zeros(self.rod.nu_element, dtype=float)

        for i in range(self.rod.nquadrature):
            # extract reference state variables
            qpi = self.rod.qp[el, i]
            qwi = self.rod.qw[el, i]
            Ji = self.rod.J[el, i]

            # compute local force vector
            he_qp = self.force(t, qpi) * Ji * qwi

            # multiply local force vector with variation of centerline
            for node in range(self.rod.nnodes_element_r):
                he[self.rod.nodalDOF_element_r[node]] += (
                    self.rod.N_r[el, i, node] * he_qp
                )

        return he

    def h(self, t, q, u):
        h = np.zeros(self.rod.nu, dtype=np.common_type(q, u))
        for el in range(self.rod.nelement):
            h[self.rod.elDOF_u[el]] += self.h_el(t, el)
        return h

    def h_q(self, t, q, u):
        return None
