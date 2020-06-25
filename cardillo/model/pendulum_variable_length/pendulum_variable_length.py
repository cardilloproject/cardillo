import numpy as np

class Pendulum_variable_length():
    def __init__(self, m, l, l_t, F, r_pivot=np.zeros(3), q0=None, u0=None):
        self.m = m
        self.l = l
        self.l_t = l_t
        self.F = F
        self.r_pivot = r_pivot
        self.__qDOF = None
        self.__uDOF = None

        self.nq = 2
        self.nu = 1
        self.q0 = np.zeros(self.nq) if q0 is None else q0
        self.u0 = np.zeros(self.nu) if u0 is None else u0

    @property
    def qDOF(self):
        return self.__qDOF

    @qDOF.setter
    def qDOF(self, qDOF):
        self.__qDOF = qDOF

    @property
    def uDOF(self):
        return self.__uDOF

    @uDOF.setter
    def uDOF(self, uDOF):
        self.__uDOF = uDOF

    def M(self, t, q, coo):
        M = np.array([[self.m * self.l(t)**2]])
        coo.extend(M, (self.uDOF, self.uDOF))

    def f_gyr(self, t, q, u):
        return np.array([2 * self.m * self.l(t) * self.l_t(t) * u[0]])

    def f_gyr_u(self, t, q, u, coo):
        dense = np.array([[2 * self.m * self.l(t) * self.l_t(t)]])
        coo.extend(dense, (self.uDOF, self.uDOF))

    def f_pot(self, t, q):
        F = self.F(t)
        return np.array([F[0] * q[1] + F[1] * q[0]])

    def f_pot_q(self, t, q, coo):
        F = self.F(t)
        dense = np.array([[ F[1], F[0] ]])
        coo.extend(dense, (self.uDOF, self.qDOF))

    def q_dot(self, t, q, u):
        return np.array([q[1] * u[0], \
                         -q[0] * u[0]]) + self.l_t(t) / self.l(t) * q

    def q_dot_q(self, t, q, u, coo):
        dense = self.l_t(t) / self.l(t) * np.eye(2) \
                + np.array([[0, u[0]], \
                            [-u[0], 0]])
        coo.extend(dense, (self.qDOF, self.qDOF))        

    def B_dense(self, t, q):
        return np.array([[q[1]], \
                         [-q[0]]])

    def B(self, t, q, coo):
        coo.extend(self.B_dense(t, q), (self.qDOF, self.uDOF))

    def callback(self, t, q, u):
        l_act = np.linalg.norm(q)
        q = self.l(t) / l_act * q
        return q, u

    def point(self, ID):
        return Point(self, ID)

class Point():
    def __init__(self, subsystem, ID):
        self.subsystem = subsystem
        self.ID = ID

        self.B = subsystem.B_dense

    @property
    def qDOF(self):
        return self.subsystem.qDOF

    @property
    def uDOF(self):
        return self.subsystem.uDOF        

    def position(self, t, q):
        return self.subsystem.r_pivot + self.ID * np.array([q[0], -q[1], 0])
    
    def position_q(self, t, q):
        return self.ID * np.array([[1 , 0 ], [0, -1], [0, 0]])

    def position_qq(self, t, q):
        nq = len(self.qDOF)
        return np.zeros((3, nq, nq))

    def rotation(self, t, q):
        dr = np.array([q[0], -q[1], 0]) - self.subsystem.r_pivot
        d1 = dr / np.linalg.norm(dr)
        d2 = np.array([dr[1], -dr[0], 0])
        d3 = np.array([0, 0, 1])
        return np.vstack((d1, d2, d3)).T

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
