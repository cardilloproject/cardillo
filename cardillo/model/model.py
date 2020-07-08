import numpy as np
from cardillo.utility.coo import Coo
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve 

properties = []
properties.extend(['M', 'Mu_q'])
properties.extend(['f_gyr', 'f_gyr_q', 'f_gyr_u'])
properties.extend(['f_pot', 'f_pot_q'])
properties.extend(['f_npot', 'f_npot_q', 'f_npot_u'])

properties.extend(['q_dot', 'q_dot_q', 'B'])

properties.extend(['g', 'g_q', 'g_t'])
properties.extend(['gamma', 'gamma_q', 'gamma_u'])

properties.extend(['assembler_callback', 'solver_step_callback'])

class Model(object):
    """Sparse model implementation which assembles all global objects without copying on body and element level. 

    Notes
    -----

    All model functions which return matrices have :py:class:`scipy.sparse.coo_matrix` as default scipy sparse matrix type (:py:class:`scipy.sparse.spmatrix`). This is due to the fact that the assembling of global iteration matrices is done using :py:func:`scipy.sparse.bmat` which in a first step transforms all matrices to :py:class:`scipy.sparse.coo_matrix`. A :py:class:`scipy.sparse.coo_matrix`, inherits form :py:class:`scipy.sparse._data_matrix` `[1] <https://github.com/scipy/scipy/blob/adc4f4f7bab120ccfab9383aba272954a0a12fb0/scipy/sparse/data.py#L21-L126>`_, have limited support for arithmetic operations, only a few operations as :py:func:`__neg__`, :py:func:`__imul__`, :py:func:`__itruediv__` are implemented. For all other operations the matrix is first transformed to a :py:class:`scipy.sparse.csr_matrix` `[2] <https://github.com/scipy/scipy/blob/adc4f4f7bab120ccfab9383aba272954a0a12fb0/scipy/sparse/base.py#L330-L335>`_. Slicing is also not supported for matrices of type :py:class:`scipy.sparse.coo_matrix`, we have to use other formats as :py:class:`scipy.sparse.csr_matrix` or :py:class:`scipy.sparse.csc_matrix` for that.

    """

    def __init__(self, t0=0):
        self.t0 = t0
        self.nq = 0
        self.nu = 0
        self.nla_g = 0
        self.nla_gamma = 0
        # self.nla_N = 0
        # self.nla_T = 0

        self.contributions = []

        for p in properties:
            setattr(self, f'_{self.__class__.__name__}__{p}_contr', [])

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
        self.nq = 0
        self.nu = 0
        self.nla_g = 0
        self.nla_gamma = 0
        # self.nla_N = 0
        # self.nla_T = 0
        q0 = []
        u0 = []
        la_g0 = []
        la_gamma0 = []
        
        for contr in self.contributions:
            contr.t0 = self.t0
            for p in properties:
                # if property is implemented as class function append to property contribution
                # - p in contr.__class__.__dict__: has global class attribute p
                # - callable(getattr(contr, p, None)): p is callable
                if (hasattr(contr, p) and callable(getattr(contr, p)) ):
                    getattr(self, f'_{self.__class__.__name__}__{p}_contr').append(contr)

            # if contribution has position degrees of freedom address position coordinates
            if hasattr(contr, 'nq'):
                contr.qDOF = np.arange(0, contr.nq) + self.nq
                self.nq += contr.nq
                q0.extend(contr.q0.tolist())

            # if contribution has velocity degrees of freedom address velocity coordinates
            if hasattr(contr, 'nu'):
                contr.uDOF = np.arange(0, contr.nu) + self.nu 
                self.nu += contr.nu
                u0.extend(contr.u0.tolist())
            
            # if contribution has constraints on position level address constraint coordinates
            if hasattr(contr, 'nla_g'):
                contr.la_gDOF = np.arange(0, contr.nla_g) + self.nla_g
                self.nla_g += contr.nla_g
                la_g0.extend(contr.la_g0.tolist())
            
            # if contribution has constraints on velocity level address constraint coordinates
            if hasattr(contr, 'nla_gamma'):
                contr.la_gammaDOF = np.arange(0, contr.nla_gamma) + self.nla_gamma
                self.nla_gamma += contr.nla_gamma
                la_gamma0.extend(contr.la_gamma0.tolist())

        self.q0 = np.array(q0)
        self.u0 = np.array(u0)
        self.la_g0 = np.array(la_g0)
        self.la_gamma0 = np.array(la_gamma0)

        # call assembler callback: call methods that require first an assembly of the system
        self.assembler_callback()

    def assembler_callback(self):
        for contr in self.__assembler_callback_contr:
            contr.assembler_callback()
    
    #====================
    # equations of motion
    #====================
    def M(self, t, q, scipy_matrix=coo_matrix):
        coo = Coo((self.nu, self.nu))
        for contr in self.__M_contr:
            contr.M(t, q[contr.qDOF], coo)
        return coo.tosparse(scipy_matrix)

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

    def E_pot(self, t, q):
        E_pot = 0
        for contr in self.__f_pot_contr:
            E_pot += contr.E_pot(t, q[contr.qDOF])
        return E_pot

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

    #====================
    # kinematic equations
    #====================
    def q_dot(self, t, q, u):
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

    def q_ddot(self, t, q, u, u_dot):
        q_ddot = np.zeros(self.nq)
        for contr in self.__q_dot_contr:
            q_ddot[contr.qDOF] += contr.q_ddot(t, q[contr.qDOF], u[contr.uDOF], u_dot[contr.uDOF])
        return q_ddot

    def solver_step_callback(self, t, q, u):
        for contr in self.__solver_step_callback_contr:
            q[contr.qDOF], u[contr.uDOF] = contr.solver_step_callback(t, q[contr.qDOF], u[contr.uDOF])
        return q, u

    #========================================
    # bilateral constraints on position level
    #========================================
    def g(self, t, q):
        g = np.zeros(self.nla_g)
        for contr in self.__g_contr:
            g[contr.la_gDOF] = contr.g(t, q[contr.qDOF])
        return g

    def g_t(self, t, q):
        g_t = np.zeros(self.nla_g)
        for contr in self.__g_t_contr:
            g_t[contr.la_gDOF] = contr.g_t(t, q[contr.qDOF])
        return g_t

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

    def g_dot(self, t, q, u):
        g_dot = np.zeros(self.nla_g)
        for contr in self.__g_contr:
            g_dot[contr.la_gDOF] = contr.g_dot(t, q[contr.qDOF], u[contr.uDOF])
        return g_dot

    def chi_g(self, t, q):
        return self.g_dot(t, q, np.zeros(self.nu))

    def g_dot_u(self, t, q, scipy_matrix=coo_matrix):
        coo = Coo((self.nla_g, self.nu))
        for contr in self.__g_contr:
            contr.g_dot_u(t, q[contr.qDOF], coo)
        return coo.tosparse(scipy_matrix)

    def g_ddot(self, t, q, u, u_dot):
        g_ddot = np.zeros(self.nla_g)
        for contr in self.__g_contr:
            g_ddot[contr.la_gDOF] = contr.g_ddot(t, q[contr.qDOF], u[contr.uDOF], u_dot[contr.uDOF])
        return g_ddot

    def zeta_g(self, t, q, u):
        return self.g_ddot(t, q, u, np.zeros(self.nu))

    #========================================
    # bilateral constraints on velocity level
    #========================================
    def gamma(self, t, q, u):
        gamma = np.zeros(self.nla_gamma)
        for contr in self.__gamma_contr:
            gamma[contr.la_gammaDOF] = contr.gamma(t, q[contr.qDOF], u[contr.uDOF])
        return gamma

    def chi_gamma(self, t, q):
        return self.gamma(t, q, np.zeros(self.nu))

    def gamma_dot(self, t, q, u, u_dot):
        gamma_dot = np.zeros(self.nla_gamma)
        for contr in self.__gamma_contr:
            gamma_dot[contr.la_gammaDOF] = contr.gamma_dot(t, q[contr.qDOF], u[contr.uDOF], u_dot[contr.uDOF])
        return gamma_dot

    # def gamma_dot(self, t, q, u, u_dot):
    #     gamma_dot = np.zeros(self.nla_gamma)
    #     gamma_dot += self.gamma_q(t, q, u) @ self.q_dot(t, q, u)
    #     gamma_dot += self.gamma_u(t, q) @ u_dot
    #     return gamma_dot

    def zeta_gamma(self, t, q, u):
        return self.gamma_dot(t, q, u, np.zeros(self.nu))

    def gamma_q(self, t, q, u, scipy_matrix=coo_matrix):
        coo = Coo((self.nla_gamma, self.nq))
        for contr in self.__gamma_contr:
            contr.gamma_q(t, q[contr.qDOF], u[contr.uDOF], coo)
        return coo.tosparse(scipy_matrix)

    def gamma_u(self, t, q, scipy_matrix=coo_matrix):
        coo = Coo((self.nla_gamma, self.nu))
        for contr in self.__gamma_contr:
            contr.gamma_u(t, q[contr.qDOF], coo)
        return coo.tosparse(scipy_matrix)

    def W_gamma(self, t, q, scipy_matrix=coo_matrix):
        coo = Coo((self.nu, self.nla_gamma))
        for contr in self.__gamma_contr:
            contr.W_gamma(t, q[contr.qDOF], coo)
        return coo.tosparse(scipy_matrix)

    def Wla_gamma_q(self, t, q, la_gamma, scipy_matrix=coo_matrix):
        coo = Coo((self.nu, self.nq))
        for contr in self.__gamma_contr:
            contr.Wla_gamma_q(t, q[contr.qDOF], la_gamma[contr.la_gammaDOF], coo)
        return coo.tosparse(scipy_matrix)