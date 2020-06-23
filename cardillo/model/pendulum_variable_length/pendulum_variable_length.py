import numpy as np

class Pendulum_variable_length():
    def __init__(self, m, l, l_t, F, q0=None, u0=None):
        self.m = m
        self.l = l
        self.l_t = l_t
        self.F = F

        self.nq = 2
        self.nu = 1
        self.q0 = np.zeros(self.nq) if q0 is None else q0
        self.u0 = np.zeros(self.nu) if u0 is None else u0

    def M(self, t, q, M_coo):
        M = np.array([[self.m * self.l(t)**2]])
        M_coo.extend(M, (self.uDOF, self.uDOF))

    def f_gyr(self, t, q, u):
        return np.array([2 * self.m * self.l(t) * self.l_t(t) * u[0]])

    def f_npot(self, t, q, u):
        F = self.F(t)
        return np.array([F[0] * q[1] + F[1] * q[0]])

    def B(self, t, q, B_coo):
        B = np.array([[q[1]], \
                      [-q[0]]])
        B_coo.extend(B, (self.qDOF, self.uDOF))
        
    def beta(self, t, q):
        return self.l_t(t) / self.l(t) * q

if __name__ == "__main__":
    m = 1
    L = 2
    g = 9.81

    F = lambda t: np.array([0, -m * g])

    l = lambda t: L + np.sin(t)
    l_t = lambda t: np.cos(t)
    l_tt = lambda t: -np.sin(t)

    pendulum = Pendulum_variable_length(m, l, l_t, F)
    'f_pot' in pendulum.__class__.__dict__ and callable(pendulum.f_pot)

    # getattr(pendulum, "M", False) and callable(pendulum.M)

    # # 'f_pot' in pendulum.__class__.__dict__} and callable(pendulum.f_pot)
    # print(f"'M' in pendulum.__class__.__dict__: {'M' in pendulum.__class__.__dict__}")
    # print(f"'f_pot' in pendulum.__class__.__dict__: {'f_pot' in pendulum.__class__.__dict__}")
    # pass
    # # # dict__ = Pendulum_variable_length.__dict__
    # # # pass