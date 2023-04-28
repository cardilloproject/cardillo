class DistributedForce3DContinuum:
    def __init__(self, force, subsystem):
        if not callable(force):
            self.force = lambda t, xi, eta, zeta: force
        else:
            self.force = force
        self.subsystem = subsystem

    def assembler_callback(self):
        self.qDOF = self.subsystem.qDOF
        self.uDOF = self.subsystem.uDOF

    def E_pot(self, t, q):
        return self.subsystem.distributed_force3D_pot(t, q, self.force)

    def h(self, t, q, u):
        return self.subsystem.distributed_force3D(t, q, self.force)

    def h_q(self, t, q, u):
        return self.subsystem.distributed_force3D_q(t, q, self.force)
