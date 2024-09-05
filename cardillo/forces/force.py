from numpy import einsum, zeros
from vtk import VTK_VERTEX


class Force:
    r"""Force represented w.r.t. I-basis

    Parameters
    ----------
    force : np.ndarray (3,)
        Force w.r.t. inertial I-basis as a callable function of time t.
    subsystem : object
        Object on which force acts.
    xi : #TODO
    B_r_CP : np.ndarray (3,)
        Position vector of point of attack (P) w.r.t. center of mass (C) in body-fixed B-basis.
    name : str
        Name of contribution.
    """

    def __init__(self, force, subsystem, xi=zeros(3), B_r_CP=zeros(3), name="force"):
        if not callable(force):
            self.force = lambda t: force
        else:
            self.force = force
        self.subsystem = subsystem
        self.xi = xi
        self.name = name

        self.r_OP = lambda t, q: subsystem.r_OP(t, q, xi, B_r_CP)
        self.J_P = lambda t, q: subsystem.J_P(t, q, xi, B_r_CP)
        self.J_P_q = lambda t, q: subsystem.J_P_q(t, q, xi, B_r_CP)

    def assembler_callback(self):
        self.qDOF = self.subsystem.qDOF[self.subsystem.local_qDOF_P(self.xi)]
        self.uDOF = self.subsystem.uDOF[self.subsystem.local_uDOF_P(self.xi)]

    def E_pot(self, t, q):
        return -(self.force(t) @ self.r_OP(t, q))

    def h(self, t, q, u):
        return self.force(t) @ self.J_P(t, q)

    def h_q(self, t, q, u):
        return einsum("i,ijk->jk", self.force(t), self.J_P_q(t, q))

    def export(self, sol_i, **kwargs):
        points = [self.r_OP(sol_i.t, sol_i.q[self.qDOF])]
        cells = [(VTK_VERTEX, [0])]
        cell_data = dict(F=[self.force(sol_i.t)])
        return points, cells, None, cell_data


class B_Force:
    r"""Force represented w.r.t. body-fixed B-basis

    Parameters
    ----------
    force : np.ndarray (3,)
        Force w.r.t. body-fixed B-basis as a callable function of time t.
    subsystem : object
        Object on which force acts.
    xi : #TODO
    B_r_CP : np.ndarray (3,)
        Position vector of point of attack (P) w.r.t. center of mass (C) in body-fixed B-basis.
    name : str
        Name of contribution.
    """

    def __init__(self, force, subsystem, xi=zeros(3), B_r_CP=zeros(3), name="force"):
        if not callable(force):
            self.force = lambda t: force
        else:
            self.force = force
        self.subsystem = subsystem
        self.xi = xi
        self.name = name

        self.A_IB = lambda t, q: subsystem.A_IB(t, q, xi=xi)
        self.A_IB_q = lambda t, q: subsystem.A_IB_q(t, q, xi=xi)
        self.r_OP = lambda t, q: subsystem.r_OP(t, q, xi=xi, B_r_CP=B_r_CP)
        self.J_P = lambda t, q: subsystem.J_P(t, q, xi=xi, B_r_CP=B_r_CP)
        self.J_P_q = lambda t, q: subsystem.J_P_q(t, q, xi=xi, B_r_CP=B_r_CP)

    def assembler_callback(self):
        self.qDOF = self.subsystem.qDOF[self.subsystem.local_qDOF_P(self.xi)]
        self.uDOF = self.subsystem.uDOF[self.subsystem.local_uDOF_P(self.xi)]

    def h(self, t, q, u):
        return (self.A_IB(t, q) @ self.force(t)) @ self.J_P(t, q)

    def h_q(self, t, q, u):
        return einsum(
            "ijk,j,il->lk", self.A_IB_q(t, q), self.force(t), self.J_P(t, q)
        ) + einsum("i,ijk->jk", self.A_IB(t, q) @ self.force(t), self.J_P_q(t, q))

    def export(self, sol_i, **kwargs):
        r_OP = self.r_OP(sol_i.t, sol_i.q[self.qDOF])
        A_IB = self.A_IB(sol_i.t, sol_i.q[self.qDOF])
        I_F = A_IB @ self.force(sol_i.t)

        points = [r_OP]
        cells = [(VTK_VERTEX, [0])]
        cell_data = dict(F=[I_F])
        return points, cells, None, cell_data
