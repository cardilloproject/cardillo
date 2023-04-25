class DistributedForce2DContinuum:
    def __init__(self, force_distr2D, subsystem, srf_idx):
        if not callable(force_distr2D):
            self.force_distr2D = lambda t, xi, eta: force_distr2D
        else:
            self.force_distr2D = force_distr2D
        self.subsystem = subsystem
        self.srf_idx = srf_idx
        self.srf_mesh = self.subsystem.mesh.surface_mesh[self.srf_idx]

    def assembler_callback(self):
        self.qDOF = self.subsystem.qDOF
        self.uDOF = self.subsystem.uDOF
        self.srf_qDOF = self.subsystem.mesh.surface_qDOF[self.srf_idx].ravel()
        self.Q_srf = self.subsystem.Z[self.srf_qDOF]
        self.srf_w_J0 = self.srf_mesh.reference_mappings(self.Q_srf)

    def E_pot(self, t, q):
        return self.subsystem.force_distr2D_pot(t, q, self.force_distr2D, self.srf_idx)

    def h(self, t, q, u):
        return self.subsystem.force_distr2D(
            t, q, self.force_distr2D, self.srf_idx, self.srf_w_J0
        )

    def h_q(self, t, q, u):
        return self.subsystem.force_distr2D_q(t, q, self.force_distr2D, self.srf_idx)
