import numpy as np

class Pendulum_variable_length():
    def __init__(self, m, l, l_t, F, r_pivot=np.zeros(3), q0=None, u0=None):
        self.m = m
        self.l = l
        self.l_t = l_t
        self.F = F
        self.r_pivot = r_pivot

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

    def B_dense(self, t, q):
        return np.array([[q[1]], \
                      [-q[0]]])

    def B(self, t, q, B_coo):
        B_coo.extend(self.B_dense(t, q), (self.qDOF, self.uDOF))
        
    def beta(self, t, q):
        return self.l_t(t) / self.l(t) * q

    def callback(self, t, q, u):
        l_act = np.linalg.norm(q)
        q = self.l(t) / l_act * q
        return q, u

    def point(self, ID):
        return Point(self, ID)

class Point():
    def __init__(self, subsystem, ID):
        self.subsystem = subsystem
        self.qDOF = subsystem.qDOF
        self.ID = ID

        self.B = subsystem.B_dense

    def position(self, t, q):
        return self.subsystem.r_pivot + self.ID * np.array([q[0], -q[1], 0])
    
    def position_q(self, t, q):
        return self.ID * np.array([[1 , 0 ], [0, -1], [0, 0]])

if __name__ == "__main__":
    m = 1
    L = 2
    g = 9.81

    F = lambda t: np.array([0, -m * g])

    l = lambda t: L + np.sin(t)
    l_t = lambda t: np.cos(t)
    l_tt = lambda t: -np.sin(t)

    pendulum = Pendulum_variable_length(m, l, l_t, F)

    Pt = pendulum.cosserat_point(1)

    pass
