# TODO: Why is this not called line_idx?
class DistributedForce2D:
    def __init__(self, force, subsystem, srf_idx):
        if not callable(force):
            self.force = lambda t, xi, eta: force
        else:
            self.force = force
        self.subsystem = subsystem
        self.srf_idx = srf_idx

    def assembler_callback(self):
        self.qDOF = self.subsystem.qDOF
        self.uDOF = self.subsystem.uDOF

    def E_pot(self, t, q):
        return self.subsystem.distributed_force2D_pot(t, q, self.force, self.srf_idx)

    def f_pot(self, t, q):
        return self.subsystem.distributed_force2D(t, q, self.force, self.srf_idx)

    def f_pot_q(self, t, q, coo):
        self.subsystem.distributed_force2D_q(t, q, coo, self.force, self.srf_idx)
