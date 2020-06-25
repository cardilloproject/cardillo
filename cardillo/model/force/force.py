from numpy import einsum
from cardillo.math import Numerical_derivative

class Force(object):
    r"""Force implementation."""

    def __init__(self, force, point):
        if not callable(force):
            self.force = lambda t: force
        else:
            self.force = force
        self.point = point

    @property
    def qDOF(self):
        return self.point.qDOF

    @property
    def uDOF(self):
        return self.point.uDOF
    
    def potential(self, t, q):
        return - ( self.force(t) @ self.point.position(t, q) )

    def f_pot(self, t, q):
        return (self.force(t) @ self.point.position_q(t, q)) @ self.point.B(t, q)

    def f_pot_q(self, t, q, coo):
        # f_q = einsum('i,ijk->jk', self.force(t), self.point.position_qq(t, q))
        f_q = Numerical_derivative(self.f_pot)._x(t, q)
        coo.extend(f_q, (self.uDOF, self.qDOF))

class Follower_force(object):
    r"""Follower force implementation."""

    def __init__(self, force, point):
        if not callable(force):
            self.force = lambda t: force
        else:
            self.force = force
        self.point = point

    @property
    def qDOF(self):
        return self.point.qDOF

    @property
    def uDOF(self):
        return self.point.uDOF

    def f_npot(self, t, q, u):
        R = self.point.rotation(t, q)
        r_q = self.point.position_q(t, q)
        f_d = self.force(t)
        B = self.point.B(t, q)
        return B.T @ r_q.T @ R @ f_d

    def f_npot_q(self, t, q, u, coo):
        f_q = Numerical_derivative(self.f_npot)._x(t, q, u)
        coo.extend(f_q, (self.uDOF, self.qDOF))

    # def force_dense(self, t, q, u):
    #     """
    #     Return the external force vector.
    #     """
    #     R = self.body.rotation(t, q, self.point_identifier)
    #     r_q = self.body.position_q(t, q, self.point_identifier)
    #     f_d = self.follower_force(t)
    #     return f_d @ R.T @ r_q

    # def force_q_dense(self, t, q, u):
    #     """
    #     Return the partial derivative of the external force vector with respect to the generalized coordinates of the involved body.
    #     """
    #     # force_q_num = NumericalDerivativeNew(self.force_dense).dR_dq(t, q, u)[:, self.qDOF_point]
    #     # # # return force_q_num

    #     r_q = self.body.position_q(t, q, self.point_identifier)
    #     r_qq = self.body.position_qq(t, q, self.point_identifier)
    #     R = self.body.rotation(t, q, self.point_identifier)
    #     R_q = self.body.rotation_q(t, q, self.point_identifier)

    #     f_d = self.follower_force(t)
    #     f_e = f_d @ R.T
    #     f_e_q = np.einsum('i,jik->jk', f_d, R_q)

    #     force_q = np.einsum('ik,ij->jk', f_e_q, r_q) + np.einsum('i,ijk->jk', f_e, r_qq)

    #     # diff = force_q - force_q_num
    #     # error = np.linalg.norm(diff)
    #     # print(error)
    #     # return force_q_num

    #     return force_q

    # def force(self, t, q, u, vec, globalDOF):
    #     """
    #     Assemble the external force vector.
    #     """
    #     vec[globalDOF[self.qDOF_point]] += self.force_dense(t, q, u)

    # def force_q(self, t, q, u, coo_matrix, globalDOF):
    #     """
    #     Sparse assembling of the partial derivative of the external force vector with respect to the generalized coordinates of the involved body.
    #     """
    #     coo_matrix.extend(self.force_q_dense(t, q, u), (globalDOF[0][self.qDOF_point], globalDOF[1][self.qDOF_point]))

    # def force_u(self, t, q, u, coo_matrix, globalDOF):
    #     """
    #     Sparse assembling of the partial derivative of the external force vector with respect to the generalized velocities of the involved body.

    #     Attention
    #     ---------
    #     This function does nothing but has to be implemented for interface reasons.
    #     """
    #     pass