# TODO: only works with 3D continua
class DistributedDoubleforce2DContinuum:
    def __init__(self, doubleforce_distr2D, subsystem, srf_idx):
        if not callable(doubleforce_distr2D):
            self.force_distr1D = lambda t, xi: doubleforce_distr2D
        else:
            self.doubleforce_distr2D = doubleforce_distr2D
        self.subsystem = subsystem
        self.srf_idx = srf_idx
        self.bc_el = self.subsystem.mesh.bc_el[srf_idx]
        self.srf_mesh = self.subsystem.mesh.surface_mesh[self.srf_idx]
        self.Nb_X = self.subsystem.mesh.Nb_X(self.subsystem.Z, srf_idx)

    def assembler_callback(self):
        self.qDOF = self.subsystem.qDOF
        self.uDOF = self.subsystem.uDOF
        self.srf_qDOF = self.subsystem.mesh.surface_qDOF[self.srf_idx].ravel()
        self.Q_srf = self.subsystem.Z[self.srf_qDOF]
        self.nz_srf = len(self.Q_srf)
        self.srf_elDOF_global = self.srf_qDOF[self.srf_mesh.elDOF]
        self.srf_w_J0 = self.srf_mesh.reference_mappings(self.Q_srf)

    def E_pot(self, t, q):
        return self.subsystem.doubleforce_distr2D_pot(
            t, q, self.force_distr1D, self.srf_idx, self.srf_w_J0
        )

    def h(self, t, q, u):
        return self.subsystem.doubleforce_distr2D(
            t, q, self.doubleforce_distr2D, self.srf_idx, self.srf_w_J0, self.Nb_X
        )

    def h_q(self, t, q, u, coo):
        self.subsystem.doubleforce_distr2D_q(
            t, q, coo, self.doubleforce_distr2D, self.srf_idx, self.srf_w_J0
        )
