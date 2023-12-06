import numpy as np




def PDcontroller(Transmission):
    class _PDcontroller(Transmission):
        def __init__(self, kp, kd, tau, **kwargs):
            if not callable(tau):
                self.tau = lambda t: tau #TODO: I don't like this to much. Maybe tau should be just a System property. Otherwise the implementation self.la_tau = self.tau may be tempting.
            else:
                self.tau = tau
            self.nla_tau = 1
            self.ntau = 2
            super().__init__(**kwargs)

            self.kp = kp
            self.kd = kd
            self.W_tau = self.W_l

        def la_tau(self, t, q, u, tau):
            return -(self.kp * (self.l(t, q) - tau[0]) + self.kd * (self.l_dot(t, q, u) - tau[1]))

    return _PDcontroller
