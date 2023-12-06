import numpy as np




def PDcontroller(Transmission):
    class _PDcontroller(Transmission):
        def __init__(self, kp, kd, tau, **kwargs):
            if not callable(tau):
                self.tau = lambda t: tau 
            else:
                self.tau = tau
            self.nla_tau = 1
            self.ntau = 2
            super().__init__(**kwargs)

            self.kp = kp
            self.kd = kd
            self.W_tau = self.W_l

        def la_tau(self, t, q, u):
            return -(self.kp * (self.l(t, q) - self.tau(t)[0]) + self.kd * (self.l_dot(t, q, u) - self.tau(t)[1]))

    return _PDcontroller
