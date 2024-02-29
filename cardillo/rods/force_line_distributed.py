import numpy as np


class Force_line_distributed:
    def __init__(self, force, rod):
        r"""Line distributed dead load for rods

        Parameters
        ----------
        force : np.ndarray (3,)
            Force w.r.t. inertial I-basis as a callable function in time t and
            rod position xi.
        rod : CosseratRod
            Cosserat rod from Cardillo.

        """
        if not callable(force):
            self.force = lambda t, xi: force
        else:
            self.force = force
        self.rod = rod

    def assembler_callback(self):
        self.qDOF = self.rod.qDOF
        self.uDOF = self.rod.uDOF

    ##################
    # potential energy
    ##################
    def E_pot(self, t, q):
        E_pot = 0
        for el in range(self.nelement):
            qe = q[self.elDOF[el]]
            E_pot += self.E_pot_el(t, qe, el)
        return E_pot

    def E_pot_el(self, t, qe, el):
        # TODO: nullify with initial configuration q0
        E_pot_el = 0.0

        for i in range(self.rod.nquadrature):
            # extract reference state variables
            qpi = self.rod.qp[el, i]
            qwi = self.rod.qw[el, i]
            Ji = self.rod.J[el, i]

            # interpolate centerline position
            r_OC = np.zeros(3, dtype=float)
            for node in range(self.nnodes_element_r):
                r_OC += self.N_r[el, i, node] * qe[self.nodalDOF_element_r[node]]

            E_pot_el -= (r_OC @ self.force(t, qpi)) * Ji * qwi

        return E_pot_el

    #####################
    # equations of motion
    #####################
    def h(self, t, q, u):
        h = np.zeros(self.rod.nu, dtype=np.common_type(q, u))
        for el in range(self.rod.nelement):
            h[self.rod.elDOF_u[el]] += self.h_el(t, el)
        return h

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

    def h_q(self, t, q, u):
        return None
