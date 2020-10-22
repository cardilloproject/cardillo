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

properties.extend(['g_N'])#, 'g_N_q', 'g_N_t'])

properties.extend(['gamma_T'])

properties.extend(['assembler_callback', 'solver_step_callback'])

properties.extend(['c'])

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
        self.nka_c = 0
        self.nla_g = 0
        self.nla_gamma = 0
        self.nla_N = 0
        self.nla_T = 0

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

    def pre_iteration_update(self, t, q, u):
        """ Update or precalculate any system variables before next solver iteration """
        for contr in self.contributions:
            if callable(getattr(contr, 'pre_iteration_update', None)):
                contr.pre_iteration_update(t, q, u)

    def assemble(self):
        self.nq = 0
        self.nu = 0
        self.nla_g = 0
        self.nla_gamma = 0
        self.nka_c = 0
        self.nla_N = 0
        self.nla_T = 0
        q0 = []
        u0 = []
        la_g0 = []
        la_gamma0 = []
        ka_c0 = []
        la_N0 = []
        la_T0 = []
        e_N = []
        e_T = []
        mu = []
        prox_r_N = []
        prox_r_T = []
        NT_connectivity = []
        N_has_friction = []
        Ncontr_connectivity = []

        n_laN_contr = 0
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
            
            # if contribution has stabilization conditions for the kinematic equation
            if hasattr(contr, 'nka_c'):
                contr.ka_cDOF = np.arange(0, contr.nka_c) + self.nka_c
                self.nka_c += contr.nka_c
                ka_c0.extend(contr.ka_c0.tolist())

            # if contribution has contacts address constraint coordinates
            if hasattr(contr, 'nla_N'):
                # normal
                contr.la_NDOF = np.arange(0, contr.nla_N) + self.nla_N
                self.nla_N += contr.nla_N
                la_N0.extend(contr.la_N0.tolist())
                e_N.extend(contr.e_N.tolist())
                prox_r_N.extend(contr.prox_r_N.tolist())
                # tangential
                contr.la_TDOF = np.arange(0, contr.nla_T) + self.nla_T
                self.nla_T += contr.nla_T
                la_T0.extend(contr.la_T0.tolist())
                e_T.extend(contr.e_T.tolist())
                prox_r_T.extend(contr.prox_r_T.tolist())
                mu.extend(contr.mu.tolist())
                for i in range(contr.nla_N):
                    NT_connectivity.append(contr.la_TDOF[np.array(contr.NT_connectivity[i], dtype=int)].tolist())
                    N_has_friction.append(True if contr.NT_connectivity[i] else False)
                    Ncontr_connectivity.append(n_laN_contr)
                n_laN_contr += 1
                
        self.q0 = np.array(q0)
        self.u0 = np.array(u0)
        self.la_g0 = np.array(la_g0)
        self.la_gamma0 = np.array(la_gamma0)
        self.ka_c0 = np.array(ka_c0)
        self.la_N0 = np.array(la_N0)
        self.la_T0 = np.array(la_T0)
        self.NT_connectivity = NT_connectivity
        self.N_has_friction = np.array(N_has_friction, dtype=bool)
        self.Ncontr_connectivity = np.array(Ncontr_connectivity, dtype=int)
        self.e_N = np.array(e_N)
        self.prox_r_N = np.array(prox_r_N)
        self.e_T = np.array(e_T)
        self.prox_r_T = np.array(prox_r_T)
        self.mu = np.array(mu)

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

    def h_q(self, t, q, u, scipy_matrix=coo_matrix):
        return self.f_pot_q(t, q, scipy_matrix=scipy_matrix) + self.f_npot_q(t, q, u, scipy_matrix=scipy_matrix) - self.f_gyr_q(t, q, u, scipy_matrix=scipy_matrix)

    def h_u(self, t, q, u, scipy_matrix=coo_matrix):
        return self.f_npot_u(t, q, u, scipy_matrix=scipy_matrix) - self.f_gyr_u(t, q, u, scipy_matrix=scipy_matrix)

    #====================
    # kinematic equations
    #====================
    def q_dot(self, t, q, u):
        q_dot = np.zeros(self.nq)
        for contr in self.__q_dot_contr:
            q_dot[contr.qDOF] = contr.q_dot(t, q[contr.qDOF], u[contr.uDOF])
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
            q_ddot[contr.qDOF] = contr.q_ddot(t, q[contr.qDOF], u[contr.uDOF], u_dot[contr.uDOF])
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

    def g_dot_q(self, t, q, u, scipy_matrix=coo_matrix):
        coo = Coo((self.nla_g, self.nq))
        for contr in self.__g_contr:
            contr.g_dot_q(t, q[contr.qDOF], u[contr.uDOF], coo)
        return coo.tosparse(scipy_matrix)

    def g_ddot(self, t, q, u, u_dot):
        g_ddot = np.zeros(self.nla_g)
        for contr in self.__g_contr:
            g_ddot[contr.la_gDOF] = contr.g_ddot(t, q[contr.qDOF], u[contr.uDOF], u_dot[contr.uDOF])
        return g_ddot
    
    def g_ddot_q(self, t, q, u, u_dot, scipy_matrix=coo_matrix):
        coo = Coo((self.nla_g, self.nq))
        for contr in self.__g_contr:
            contr.g_ddot_q(t, q[contr.qDOF], u[contr.uDOF], u_dot[contr.uDOF], coo)
        return coo.tosparse(scipy_matrix)

    def g_ddot_u(self, t, q, u, u_dot, scipy_matrix=coo_matrix):
        coo = Coo((self.nla_g, self.nu))
        for contr in self.__g_contr:
            contr.g_ddot_u(t, q[contr.qDOF], u[contr.uDOF], u_dot[contr.uDOF], coo)
        return coo.tosparse(scipy_matrix)

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

    #========================================
    # stabilization conditions for the kinematic equation
    #========================================
    def c(self, t, q):
        c = np.zeros(self.nka_c)
        for contr in self.__c_contr:
            c[contr.ka_cDOF] = contr.c(t, q[contr.qDOF])
        return c

    def c_q(self, t, q, scipy_matrix=coo_matrix):
        coo = Coo((self.nka_c, self.nq))
        for contr in self.__c_contr:
            contr.c_q(t, q[contr.qDOF], coo)
        return coo.tosparse(scipy_matrix)

    #========================================
    # contacts in normal direction
    #========================================
    def g_N(self, t, q):
        g_N = np.zeros(self.nla_N)
        for contr in self.__g_N_contr:
            g_N[contr.la_NDOF] = contr.g_N(t, q[contr.qDOF])
        return g_N

    def g_N_q(self, t, q, scipy_matrix=coo_matrix):
        coo = Coo((self.nla_N, self.nq))
        for contr in self.__g_N_contr:
            contr.g_N_q(t, q[contr.qDOF], coo)
        return coo.tosparse(scipy_matrix)

    def W_N(self, t, q, scipy_matrix=coo_matrix):
        coo = Coo((self.nu, self.nla_N))
        for contr in self.__g_N_contr:
            contr.W_N(t, q[contr.qDOF], coo)
        return coo.tosparse(scipy_matrix)

    def g_N_dot(self, t, q, u):
        g_N_dot = np.zeros(self.nla_N)
        for contr in self.__g_N_contr:
            g_N_dot[contr.la_NDOF] = contr.g_N_dot(t, q[contr.qDOF], u[contr.uDOF])
        return g_N_dot

    def g_N_ddot(self, t, q, u, a):
        g_N_ddot = np.zeros(self.nla_N)
        for contr in self.__g_N_contr:
            g_N_ddot[contr.la_NDOF] = contr.g_N_ddot(t, q[contr.qDOF], u[contr.uDOF], a[contr.uDOF])
        return g_N_ddot

    def xi_N(self, t, q, u_pre, u_post):
        xi_N = np.zeros(self.nla_N)
        for contr in self.__g_N_contr:
            # xi_N[contr.la_NDOF] = contr.g_N_dot(t, q[contr.qDOF], u_post[contr.uDOF]) + contr.e_N * contr.g_N_dot(t, q[contr.qDOF], u_pre[contr.uDOF])
            xi_N[contr.la_NDOF] = contr.xi_N(t, q[contr.qDOF], u_pre[contr.uDOF], u_post[contr.uDOF])
        return xi_N

    def xi_N_q(self, t, q, u_pre, u_post, scipy_matrix=coo_matrix):
        coo = Coo((self.nla_N, self.nq))
        for contr in self.__g_N_contr:
            contr.xi_N_q(t, q[contr.qDOF], u_pre[contr.uDOF], u_post[contr.uDOF], coo)
        return coo.tosparse(scipy_matrix)

    def chi_N(self, t, q):
        return self.g_N_dot(t, q, np.zeros(self.nu))

    def g_N_dot_u(self, t, q, scipy_matrix=coo_matrix):
        coo = Coo((self.nla_N, self.nu))
        for contr in self.__g_N_contr:
            contr.g_N_dot_u(t, q[contr.qDOF], coo)
        return coo.tosparse(scipy_matrix)

    def g_N_ddot_q(self, t, q, u, u_dot, scipy_matrix=coo_matrix):
        coo = Coo((self.nla_N, self.nu))
        for contr in self.__g_N_contr:
            contr.g_N_ddot_q(t, q[contr.qDOF], u[contr.uDOF], u_dot[contr.uDOF], coo)
        return coo.tosparse(scipy_matrix)

    def g_N_ddot_u(self, t, q, u, u_dot, scipy_matrix=coo_matrix):
        coo = Coo((self.nla_N, self.nu))
        for contr in self.__g_N_contr:
            contr.g_N_ddot_u(t, q[contr.qDOF], u[contr.uDOF], u_dot[contr.uDOF], coo)
        return coo.tosparse(scipy_matrix)

    def Wla_N_q(self, t, q, la_N, scipy_matrix=coo_matrix):
        coo = Coo((self.nu, self.nq))
        for contr in self.__g_N_contr:
            contr.Wla_N_q(t, q[contr.qDOF], la_N[contr.la_NDOF], coo)
        return coo.tosparse(scipy_matrix)

    #========================================
    # contacts in tangential direction
    #========================================
    def gamma_T(self, t, q, u):
        gamma_T = np.zeros(self.nla_T)
        for contr in self.__gamma_T_contr:
            gamma_T[contr.la_TDOF] = contr.gamma_T(t, q[contr.qDOF], u[contr.uDOF])
        return gamma_T

    def gamma_T_dot(self, t, q, u, a):
        gamma_T_dot = np.zeros(self.nla_T)
        for contr in self.__gamma_T_contr:
            gamma_T_dot[contr.la_TDOF] = contr.gamma_T_dot(t, q[contr.qDOF], u[contr.uDOF], a[contr.uDOF])
        return gamma_T_dot

    def xi_T(self, t, q, u_pre, u_post):
        xi_T = np.zeros(self.nla_T)
        for contr in self.__gamma_T_contr:
            # TODO: dimension if subsystem has multiple contacts
            xi_T[contr.la_TDOF] = contr.gamma_T(t, q[contr.qDOF], u_post[contr.uDOF]) + self.e_T[contr.la_NDOF] * contr.gamma_T(t, q[contr.qDOF], u_pre[contr.uDOF])
        return xi_T

    def xi_T_q(self, t, q, u_pre, u_post, scipy_matrix=coo_matrix):
        coo = Coo((self.nla_T, self.nq))
        for contr in self.__gamma_T_contr:
            contr.xi_T_q(t, q[contr.qDOF], u_pre[contr.uDOF], u_post[contr.uDOF], coo)
        return coo.tosparse(scipy_matrix)

    def gamma_T_q(self, t, q, u, scipy_matrix=coo_matrix):
        coo = Coo((self.nla_T, self.nq))
        for contr in self.__gamma_T_contr:
            contr.gamma_T_q(t, q[contr.qDOF], u[contr.uDOF], coo)
        return coo.tosparse(scipy_matrix)

    def gamma_T_u(self, t, q, scipy_matrix=coo_matrix):
        coo = Coo((self.nla_T, self.nu))
        for contr in self.__gamma_T_contr:
            contr.gamma_T_u(t, q[contr.qDOF], coo)
        return coo.tosparse(scipy_matrix)
    
    def gamma_T_dot_q(self, t, q, u, u_dot, scipy_matrix=coo_matrix):
        coo = Coo((self.nla_T, self.nq))
        for contr in self.__gamma_T_contr:
            contr.gamma_T_dot_q(t, q[contr.qDOF], u[contr.uDOF], u_dot[contr.uDOF], coo)
        return coo.tosparse(scipy_matrix)

    def gamma_T_dot_u(self, t, q, u, u_dot, scipy_matrix=coo_matrix):
        coo = Coo((self.nla_T, self.nu))
        for contr in self.__gamma_T_contr:
            contr.gamma_T_dot_u(t, q[contr.qDOF], u[contr.uDOF], u_dot[contr.uDOF], coo)
        return coo.tosparse(scipy_matrix)

    def W_T(self, t, q, scipy_matrix=coo_matrix):
        coo = Coo((self.nu, self.nla_T))
        for contr in self.__gamma_T_contr:
            contr.W_T(t, q[contr.qDOF], coo)
        return coo.tosparse(scipy_matrix)

    def Wla_T_q(self, t, q, la_T, scipy_matrix=coo_matrix):
        coo = Coo((self.nu, self.nq))
        for contr in self.__gamma_T_contr:
            contr.Wla_T_q(t, q[contr.qDOF], la_T[contr.la_TDOF], coo)
        return coo.tosparse(scipy_matrix)
    #========================================
    # contact force
    #========================================
    # def contact_force_fixpoint_update(self, t, q, u_pre, u_post, la_N, la_T, I_N=None):
    #     la_N1 = np.zeros(self.nla_N)
    #     la_T1 = np.zeros(self.nla_T)

    #     if I_N is None:
    #         contributions = self.__g_N_contr
    #     else:
    #         indices = np.unique(self.Ncontr_connectivity[I_N])
    #         contributions = [self.__g_N_contr[i] for i in indices]

    #     for contr in contributions:
    #         la_N1[contr.la_NDOF], la_T1[contr.la_TDOF] = contr.contact_force_fixpoint_update(t, q[contr.qDOF], u_pre[contr.uDOF], u_post[contr.uDOF], la_N[contr.la_NDOF], la_T[contr.la_TDOF])
    #     return la_N1, la_T1