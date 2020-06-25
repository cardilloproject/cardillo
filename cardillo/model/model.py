import numpy as np
from cardillo.utility.sparse import Coo
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve 

properties = []
properties.extend(['M', 'Mu_q'])
properties.extend(['f_gyr', 'f_gyr_q', 'f_gyr_u'])
properties.extend(['f_pot', 'f_pot_q'])
properties.extend(['f_npot', 'f_npot_q', 'f_npot_u'])

# properties.extend(['B', 'beta'])
properties.extend(['q_dot', 'q_dot_q', 'B'])

properties.extend(['g', 'g_q'])
properties.extend(['gamma', 'gamma_q', 'gamma_u'])

properties.extend(['callback'])

class Model(object):
    """Sparse model implementation which assembles all global objects without copying on body and element level. 

    Notes
    -----

    All model functions which return matrices have :py:class:`scipy.sparse.coo_matrix` as default scipy sparse matrix type (:py:class:`scipy.sparse.spmatrix`). This is due to the fact that the assembling of global iteration matrices is done using :py:func:`scipy.sparse.bmat` which in a first step transforms all matrices to :py:class:`scipy.sparse.coo_matrix`. A :py:class:`scipy.sparse.coo_matrix`, inherits form :py:class:`scipy.sparse._data_matrix` `[1] <https://github.com/scipy/scipy/blob/adc4f4f7bab120ccfab9383aba272954a0a12fb0/scipy/sparse/data.py#L21-L126>`_, have limited support for arithmetic operations, only a few operations as :py:func:`__neg__`, :py:func:`__imul__`, :py:func:`__itruediv__` are implemented. For all other operations the matrix is first transformed to a :py:class:`scipy.sparse.csr_matrix` `[2] <https://github.com/scipy/scipy/blob/adc4f4f7bab120ccfab9383aba272954a0a12fb0/scipy/sparse/base.py#L330-L335>`_. Slicing is also not supported for matrices of type :py:class:`scipy.sparse.coo_matrix`, we have to use other formats as :py:class:`scipy.sparse.csr_matrix` or :py:class:`scipy.sparse.csc_matrix` for that.

    """

    def __init__(self):
        # self.nDOF = 0
        self.nq = 0
        self.nu = 0
        self.nla_g = 0
        self.nla_gamma = 0
        # self.nla_N = 0
        # self.nla_T = 0

        self.contributions = []

        for p in properties:
            setattr(self, f'_{self.__class__.__name__}__{p}_contr',[])

    def add(self, contr):
        if not contr in self.contributions:
            self.contributions.append(contr)
        else:
            raise ValueError(f'contribution {str(contr)} already added')

    def remove(self, contr):
        if contr in self.contributions:
            self.contributions.remove(contr)
        else:
            raise ValueError(f'no contribution {str(contr)} to remove')

    def pop(self, index):
        self.contributions.pop(index)

    def assemble(self):
        # self.nDOF = 0
        self.nq = 0
        self.nu = 0
        self.nla_g = 0
        self.nla_gamma = 0
        # self.nla_N = 0
        # self.nla_T = 0
        q0 = []
        u0 = []
        la_g0 = []
        # la_gamma0 = []
        
        for contr in self.contributions:
            for p in properties:
                # if property is implemented as class function append to property contribution
                # - p in contr.__class__.__dict__: has global class attribute p
                # - callable(getattr(contr, p, None)): p is callable
                if (p in contr.__class__.__dict__ and callable(getattr(contr, p, None)) ):
                    getattr(self, f'_{self.__class__.__name__}__{p}_contr').append(contr)

            if getattr(contr, 'nq', False):
                contr.qDOF = np.arange(0, contr.nq) + self.nq
                self.nq += contr.nq
                q0.extend(contr.q0.tolist())

            if getattr(contr, 'nu', False):
                contr.uDOF = np.arange(0, contr.nu) + self.nu 
                self.nu += contr.nu
                u0.extend(contr.u0.tolist())
            
            if getattr(contr, 'nla_g', False):
                contr.la_gDOF = np.arange(0, contr.nla_g) + self.nla_g
                self.nla_g += contr.nla_g
                la_g0.extend(contr.la_g0.tolist())

                # TOOD: same for nla, nla_N, ...
                # if getattr(contr, 'nla_g', False):
                #     contr.uDOF = np.arange(0, contr.nu) + offset_u
                #     offset_u += contr.nu

                # if getattr(contr, 'nla_gamma', False):
                #     contr.uDOF = np.arange(0, contr.nu) + offset_u
                #     offset_u += contr.nu

        self.q0 = np.array(q0)
        self.u0 = np.array(u0)
        self.la_g0 = np.array(la_g0)
        # self.la_gamma0 = np.array(la_gamma0)
    
    # def __assemble_bilateral_constraints(self):
    #     self.la0 = np.zeros(self.n_laDOF)
    #     n_laDOF_tot = 0
    #     for bilateralConstr in self.bilateralConstraintList:
    #         bilateralConstr.laDOF = np.arange(n_laDOF_tot, n_laDOF_tot + bilateralConstr.n_laDOF)
    #         bilateralConstr.qDOF = bilateralConstr.get_qDOF()
    #         bilateralConstr.n_laDOF = len(bilateralConstr.laDOF)
    #         bilateralConstr.n_qDOF = len(bilateralConstr.qDOF)
    #         self.la0[n_laDOF_tot:n_laDOF_tot + bilateralConstr.n_laDOF] = bilateralConstr.la0
    #         n_laDOF_tot += bilateralConstr.n_laDOF

    #     # store total number of constraint forces  
    #     # self.n_laDOF = n_laDOF_tot
    #     self.laDOF = np.arange(self.n_qDOF)

    #########################################################################
    # functions are implemented with fill in and contraction on element level
    #########################################################################

    def M(self, t, q, scipy_matrix=coo_matrix):
        coo = Coo((self.nu, self.nu))
        for contr in self.__M_contr:
            contr.M(t, q[contr.qDOF], coo)
        return coo.tosparse(scipy_matrix)

    # def Mu(self, t, q, u):
    #     f = np.zeros(self.nu)
    #     for contr in self.__M_contr:
    #         f[contr.uDOF] += contr.Mu(t, q[contr.qDOF], u[contr.uDOF])
    #     return f

    def Mu_q(self, t, q, u, scipy_matrix=coo_matrix):
        coo = Coo((self.nu, self.nq))
        for contr in self.__Mu_q_contr:
            contr.Mu_q(t, q[contr.qDOF], u[contr.uDOF], coo)
        return coo.tosparse(scipy_matrix)

    def f_gyr(self, t, q, u):
        f = np.zeros(self.nu)
        for contr in self.__f_gyr_contr:
            f[contr.uDOF] += contr.f_gyr(t, q[contr.qDOF], u[contr.uDOF])
        return f

    def f_gyr_q(self, t, q, u, scipy_matrix=coo_matrix):
        coo = Coo((self.nu, self.nq))
        for contr in self.__f_gyr_q_contr:
            contr.f_gyr_q(t, q[contr.qDOF], u[contr.uDOF], coo)
        return coo.tosparse(scipy_matrix)

    def f_gyr_u(self, t, q, u, scipy_matrix=coo_matrix):
        coo = Coo((self.nu, self.nu))
        for contr in self.__f_gyr_u_contr:
            contr.f_gyr_u(t, q[contr.qDOF], u[contr.uDOF], coo)
        return coo.tosparse(scipy_matrix)

    def f_pot(self, t, q):
        f = np.zeros(self.nu)
        for contr in self.__f_pot_contr:
            f[contr.uDOF] += contr.f_pot(t, q[contr.qDOF])
        return f

    def f_pot_q(self, t, q, scipy_matrix=coo_matrix):
        coo = Coo((self.nu, self.nq))
        for contr in self.__f_pot_q_contr:
            contr.f_pot_q(t, q[contr.qDOF], coo)
        return coo.tosparse(scipy_matrix)

    def f_npot(self, t, q, u):
        f = np.zeros(self.nu)
        for contr in self.__f_npot_contr:
            f[contr.uDOF] += contr.f_npot(t, q[contr.qDOF], u[contr.uDOF])
        return f

    def f_npot_q(self, t, q, u, scipy_matrix=coo_matrix):
        coo = Coo((self.nu, self.nq))
        for contr in self.__f_npot_q_contr:
            contr.f_npot_q(t, q[contr.qDOF], u[contr.uDOF], coo)
        return coo.tosparse(scipy_matrix)

    def f_npot_u(self, t, q, u, scipy_matrix=coo_matrix):
        coo = Coo((self.nu, self.nu))
        for contr in self.__f_npot_u_contr:
            contr.f_npot_u(t, q[contr.qDOF], u[contr.uDOF], coo)
        return coo.tosparse(scipy_matrix)

    def h(self, t, q, u):
        return self.f_pot(t, q) + self.f_npot(t, q, u) - self.f_gyr(t, q, u)

    def h_q(self, t, q, u):
        return self.f_pot_q(t, q) + self.f_npot_q(t, q, u) - self.f_gyr_q(t, q, u)

    def h_u(self, t, q, u):
        return self.f_npot_u(t, q, u) - self.f_gyr_u(t, q, u)

    # def u_dot(self, t, q, u, la_g):
    #     rhs = self.h(t, q, u) + self.Wla_g(t, q, la_g)
    #     return spsolve(self.M(t, q, csr_matrix), rhs)

    def q_dot(self, t, q, u):
        # return self.B(t, q, csr_matrix) @ u + self.beta(t, q)
        q_dot = np.zeros(self.nq)
        for contr in self.__q_dot_contr:
            q_dot[contr.qDOF] += contr.q_dot(t, q[contr.qDOF], u[contr.uDOF])
        return q_dot

    def q_dot_q(self, t, q, u, scipy_matrix=coo_matrix):
        coo = Coo((self.nq, self.nq))
        for contr in self.__q_dot_q_contr:
            contr.q_dot_q(t, q[contr.qDOF], u[contr.uDOF], coo)
        return coo.tosparse(scipy_matrix)

    def B(self, t, q, scipy_matrix=coo_matrix):
        coo = Coo((self.nq, self.nu))
        for contr in self.__B_contr:
            contr.B(t, q[contr.qDOF], coo)
        return coo.tosparse(scipy_matrix)

    # def beta(self, t, q):
    #     b = np.zeros(self.nq)
    #     for contr in self.__beta_contr:
    #         b[contr.qDOF] += contr.beta(t, q[contr.qDOF])
    #     return b

    def callback(self, t, q, u):
        for contr in self.__callback_contr:
            q[contr.qDOF], u[contr.uDOF] = contr.callback(t, q[contr.qDOF], u[contr.uDOF])
        return q, u

    def g(self, t, q):
        g = np.zeros(self.nla_g)
        for contr in self.__g_contr:
            g[contr.la_gDOF] = contr.g(t, q[contr.qDOF])
        return g

    def g_q(self, t, q, scipy_matrix=coo_matrix):
        coo = Coo((self.nla_g, self.nq))
        for contr in self.__g_contr:
            contr.g_q(t, q[contr.qDOF], coo)
        return coo.tosparse(scipy_matrix)

    def W_g(self, t, q, scipy_matrix=coo_matrix):
        coo = Coo((self.nu, self.nla_g))
        for contr in self.__g_contr:
            contr.W_g(t, q[contr.qDOF], coo)
        return coo.tosparse(scipy_matrix)

    def Wla_g_q(self, t, q, la_g, scipy_matrix=coo_matrix):
        coo = Coo((self.nu, self.nq))
        for contr in self.__g_contr:
            contr.Wla_g_q(t, q[contr.qDOF], la_g[contr.la_gDOF], coo)
        return coo.tosparse(scipy_matrix)


if __name__ == "__main__":
    from cardillo.model.pendulum_variable_length import Pendulum_variable_length
    m = 1
    L = 2
    g = 9.81

    F = lambda t: np.array([0, -m * g])

    l = lambda t: L + np.sin(t)
    l_t = lambda t: np.cos(t)
    l_tt = lambda t: -np.sin(t)

    # pendulum1 = Pendulum_variable_length(m, l, l_t, F)
    q0 = np.array([L, 0])
    u0 = np.array([-3])
    pendulum1 = Pendulum_variable_length(m, l, l_t, F, q0=q0, u0=u0)

    model = Model()
    model.add(pendulum1)

    pendulum2 = Pendulum_variable_length(2 * m, l, l_t, F)
    model.add(pendulum2)

    # model.remove(pendulum1)
    # model.remove(model)
    # model.pop(5)

    model.assemble()

    print(f'M = \n{model.M(0, model.q0).toarray()}')
    print(f'f_gyr = {model.f_gyr(0, model.q0, model.u0)}')
    print(f'f_pot = {model.f_pot(0, model.q0)}')
    print(f'f_npot = {model.f_npot(0, model.q0, model.u0)}')
    print(f'h = {model.h(0, model.q0, model.u0)}')
    print(f'B = \n{model.B(0, model.q0).toarray()}')
    print(f'beta = {model.beta(0, model.q0)}')

    # Pt = pendulum1.cosserat_point(1)
    pass
