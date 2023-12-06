import numpy as np

def PIDcontroller(Transmission):
    class _PIDcontroller(Transmission):
        def __init__(self, kp, ki, kd, tau, **kwargs):
            if not callable(tau):
                self.tau = lambda t: tau #TODO: I don't like this to much. Maybe tau should be just a System property. Otherwise the implementation self.la_tau = self.tau may be tempting.
            else:
                self.tau = tau
            self.nla_tau = 1
            self.ntau = 2
            self.nq = 1
            self.q0 = np.zeros(1)
            super().__init__(**kwargs)

            self.kp = kp
            self.ki = ki
            self.kd = kd
    

        def assembler_callback(self):
            super().assembler_callback()
            self.qDOF = np.concatenate([self.qDOF, self.q_dotDOF])
        
        def q_dot(self, t, q, u):
            return self.l(t, q[:-1]) - self.tau(t)[0] # TODO: wie machen wir das?? optionales tau argument für q_dot??
        
        def W_tau(self, t, q):
            return self.W_l(t, q[:-1])

        def la_tau(self, t, q, u):
            tau = self.tau(t)
            integral_error = q[-1]
            return -(self.ki * integral_error + self.kp * (self.l(t, q[:-1]) - tau[0]) + self.kd * (self.l_dot(t, q[:-1], u) - tau[1]))

    return _PIDcontroller
