import numpy as np
from cardillo.utility.coo import Coo
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

properties = []
properties.extend(["M", "Mu_q"])
properties.extend(["f_gyr", "f_gyr_q", "f_gyr_u"])
properties.extend(["f_pot", "f_pot_q"])
properties.extend(["f_npot", "f_npot_q", "f_npot_u"])
properties.extend(["f_scaled", "f_scaled_q"])

properties.extend(["q_dot", "q_dot_q", "B"])

properties.extend(["g", "g_q", "g_t"])
properties.extend(["gamma", "gamma_q", "gamma_u"])

properties.extend(["g_N"])

properties.extend(["gamma_F"])

properties.extend(["assembler_callback", "pre_iteration_update", "step_callback"])

properties.extend(["g_S"])


class System:
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
        self.nla_S = 0
        self.nla_N = 0
        self.nla_F = 0

        self.contributions = []

        for p in properties:
            setattr(self, f"_{self.__class__.__name__}__{p}_contr", [])

    def add(self, contr):
        if not contr in self.contributions:
            self.contributions.append(contr)
        else:
            raise ValueError(f"contribution {str(contr)} already added")

    def remove(self, contr):
        if contr in self.contributions:
            self.contributions.remove(contr)
        else:
            raise ValueError(f"no contribution {str(contr)} to remove")

    def pop(self, index):
        self.contributions.pop(index)

    def extend(self, contr_list):
        list(map(self.add, contr_list))

    # def pre_iteration_update(self, t, q, u):
    #     """Update or precalculate any system variables before next solver iteration"""
    #     for contr in self.contributions:
    #         if callable(getattr(contr, "pre_iteration_update", None)):
    #             contr.pre_iteration_update(t, q, u)

    def assemble(self):
        self.nq = 0
        self.nu = 0
        self.nla_g = 0
        self.nla_gamma = 0
        self.nla_S = 0
        self.nla_N = 0
        self.nla_F = 0
        q0 = []
        u0 = []
        la_g0 = []
        la_gamma0 = []
        la_S0 = []
        la_N0 = []
        la_F0 = []
        e_N = []
        e_F = []
        mu = []
        # prox_r_N = []
        # prox_r_F = []
        NF_connectivity = []
        N_has_friction = []
        Ncontr_connectivity = []

        n_laN_contr = 0
        for contr in self.contributions:
            contr.t0 = self.t0
            for p in properties:
                # if property is implemented as class function append to property contribution
                # - p in contr.__class__.__dict__: has global class attribute p
                # - callable(getattr(contr, p, None)): p is callable
                if hasattr(contr, p) and callable(getattr(contr, p)):
                    getattr(self, f"_{self.__class__.__name__}__{p}_contr").append(
                        contr
                    )

            # if contribution has position degrees of freedom address position coordinates
            if hasattr(contr, "nq"):
                contr.qDOF = np.arange(0, contr.nq) + self.nq
                self.nq += contr.nq
                q0.extend(contr.q0.tolist())

            # if contribution has velocity degrees of freedom address velocity coordinates
            if hasattr(contr, "nu"):
                contr.uDOF = np.arange(0, contr.nu) + self.nu
                self.nu += contr.nu
                u0.extend(contr.u0.tolist())

            # if contribution has constraints on position level address constraint coordinates
            if hasattr(contr, "nla_g"):
                contr.la_gDOF = np.arange(0, contr.nla_g) + self.nla_g
                self.nla_g += contr.nla_g
                la_g0.extend(contr.la_g0.tolist())

            # if contribution has constraints on velocity level address constraint coordinates
            if hasattr(contr, "nla_gamma"):
                contr.la_gammaDOF = np.arange(0, contr.nla_gamma) + self.nla_gamma
                self.nla_gamma += contr.nla_gamma
                la_gamma0.extend(contr.la_gamma0.tolist())

            # if contribution has stabilization conditions for the kinematic equation
            if hasattr(contr, "nla_S"):
                contr.la_SDOF = np.arange(0, contr.nla_S) + self.nla_S
                self.nla_S += contr.nla_S
                la_S0.extend(contr.la_S0.tolist())

            # if contribution has contacts address constraint coordinates
            if hasattr(contr, "nla_N"):
                # normal
                contr.la_NDOF = np.arange(0, contr.nla_N) + self.nla_N
                self.nla_N += contr.nla_N
                la_N0.extend(contr.la_N0.tolist())
                e_N.extend(contr.e_N.tolist())
                # tangential
                contr.la_FDOF = np.arange(0, contr.nla_F) + self.nla_F
                self.nla_F += contr.nla_F
                la_F0.extend(contr.la_F0.tolist())
                e_F.extend(contr.e_F.tolist())
                mu.extend(contr.mu.tolist())
                for i in range(contr.nla_N):
                    NF_connectivity.append(
                        contr.la_FDOF[
                            np.array(contr.NF_connectivity[i], dtype=int)
                        ].tolist()
                    )
                    N_has_friction.append(True if contr.NF_connectivity[i] else False)
                    Ncontr_connectivity.append(n_laN_contr)
                n_laN_contr += 1

        self.q0 = np.array(q0)
        self.u0 = np.array(u0)
        self.la_g0 = np.array(la_g0)
        self.la_gamma0 = np.array(la_gamma0)
        self.la_S0 = np.array(la_S0)
        self.la_N0 = np.array(la_N0)
        self.la_F0 = np.array(la_F0)
        self.NF_connectivity = NF_connectivity
        self.N_has_friction = np.array(N_has_friction, dtype=bool)
        self.Ncontr_connectivity = np.array(Ncontr_connectivity, dtype=int)
        self.e_N = np.array(e_N)
        self.e_F = np.array(e_F)
        self.mu = np.array(mu)

        # call assembler callback: call methods that require first an assembly of the system
        self.assembler_callback()

    def assembler_callback(self):
        for contr in self.__assembler_callback_contr:
            contr.assembler_callback()

    #####################
    # kinematic equations
    #####################
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
            q_ddot[contr.qDOF] = contr.q_ddot(
                t, q[contr.qDOF], u[contr.uDOF], u_dot[contr.uDOF]
            )
        return q_ddot

    def pre_iteration_update(self, t, q, u):
        for contr in self.__pre_iteration_update_contr:
            q[contr.qDOF], u[contr.uDOF] = contr.pre_iteration_update(
                t, q[contr.qDOF], u[contr.uDOF]
            )
        return q, u

    def step_callback(self, t, q, u):
        for contr in self.__step_callback_contr:
            q[contr.qDOF], u[contr.uDOF] = contr.step_callback(
                t, q[contr.qDOF], u[contr.uDOF]
            )
        return q, u

    #####################
    # equations of motion
    #####################
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
        return (
            self.f_pot_q(t, q, scipy_matrix=scipy_matrix)
            + self.f_npot_q(t, q, u, scipy_matrix=scipy_matrix)
            - self.f_gyr_q(t, q, u, scipy_matrix=scipy_matrix)
        )

    def h_u(self, t, q, u, scipy_matrix=coo_matrix):
        return self.f_npot_u(t, q, u, scipy_matrix=scipy_matrix) - self.f_gyr_u(
            t, q, u, scipy_matrix=scipy_matrix
        )

    # TODO: do this better!
    # scaled forces for arc-length solvers
    def f_scaled(self, t, q):
        f = np.zeros(self.nu)
        for contr in self.__f_scaled_contr:
            f[contr.uDOF] += contr.f_scaled(t, q[contr.qDOF])
        return f

    def f_scaled_q(self, t, q, scipy_matrix=coo_matrix):
        coo = Coo((self.nu, self.nq))
        for contr in self.__f_scaled_q_contr:
            contr.f_scaled_q(t, q[contr.qDOF], coo)
        return coo.tosparse(scipy_matrix)

    #########################################
    # bilateral constraints on position level
    #########################################
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

    def g_q_T_mu_g(self, t, q, mu_g, scipy_matrix=coo_matrix):
        coo = Coo((self.nq, self.nq))
        for contr in self.__g_contr:
            contr.g_q_T_mu_g(t, q[contr.qDOF], mu_g[contr.la_gDOF], coo)
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
            g_ddot[contr.la_gDOF] = contr.g_ddot(
                t, q[contr.qDOF], u[contr.uDOF], u_dot[contr.uDOF]
            )
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

    #########################################
    # bilateral constraints on velocity level
    #########################################
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
            gamma_dot[contr.la_gammaDOF] = contr.gamma_dot(
                t, q[contr.qDOF], u[contr.uDOF], u_dot[contr.uDOF]
            )
        return gamma_dot

    def zeta_gamma(self, t, q, u):
        return self.gamma_dot(t, q, u, np.zeros(self.nu))

    def gamma_q(self, t, q, u, scipy_matrix=coo_matrix):
        coo = Coo((self.nla_gamma, self.nq))
        for contr in self.__gamma_contr:
            contr.gamma_q(t, q[contr.qDOF], u[contr.uDOF], coo)
        return coo.tosparse(scipy_matrix)

    def gamma_dot_q(self, t, q, u, u_dot, scipy_matrix=coo_matrix):
        coo = Coo((self.nla_gamma, self.nq))
        for contr in self.__gamma_contr:
            contr.gamma_dot_q(t, q[contr.qDOF], u[contr.uDOF], u_dot[contr.uDOF], coo)
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

    #####################################################
    # stabilization conditions for the kinematic equation
    #####################################################
    def g_S(self, t, q):
        g_S = np.zeros(self.nla_S)
        for contr in self.__g_S_contr:
            g_S[contr.la_SDOF] = contr.g_S(t, q[contr.qDOF])
        return g_S

    def g_S_q(self, t, q, scipy_matrix=coo_matrix):
        coo = Coo((self.nla_S, self.nq))
        for contr in self.__g_S_contr:
            contr.g_S_q(t, q[contr.qDOF], coo)
        return coo.tosparse(scipy_matrix)

    # TODO: Do we need them for stabilization of kinematic equation?
    # def W_S
    # def W_la_S_q

    #################
    # normal contacts
    #################
    def active_contacts(self, t, q):
        active_contacts = np.zeros(self.nla_N, dtype=bool)
        for contr in self.__g_N_contr:
            active_contacts[contr.la_NDOF] = contr.active_contact(t, q[contr.qDOF])
        return active_contacts

    # TODO: How can we use fixed prox parameters on subsystem level?
    def prox_r_N(self, t, q):
        # M_coo = Coo((subsystem.nu, subsystem.nu))
        # self.subsystem.M(t, q, M_coo)
        M = self.M(t, q, csc_matrix)
        W_N = self.W_N(t, q, csc_matrix)

        G = np.atleast_2d(W_N.T @ spsolve(M, W_N))
        return 1.0 / np.diag(G)

        # prox_r_N = np.zeros([self.nla_N])
        # for contr in self.__g_N_contr:
        #     # G_ii_N = contr.G_ii_N(t, q[contr.qDOF])
        #     # prox_r_N[contr.la_NDOF] = 1 / G_ii_N if G_ii_N > 0 else G_ii_N

        #     prox_r_N[contr.la_NDOF] = contr.prox_r_N(t, q[contr.qDOF])
        # return prox_r_N

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
            g_N_ddot[contr.la_NDOF] = contr.g_N_ddot(
                t, q[contr.qDOF], u[contr.uDOF], a[contr.uDOF]
            )
        return g_N_ddot

    def xi_N(self, t, q, u_pre, u_post):
        xi_N = np.zeros(self.nla_N)
        for contr in self.__g_N_contr:
            xi_N[contr.la_NDOF] = contr.g_N_dot(
                t, q[contr.qDOF], u_post[contr.uDOF]
            ) + contr.e_N * contr.g_N_dot(t, q[contr.qDOF], u_pre[contr.uDOF])
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

    #################
    # friction
    #################
    # TODO: Use estimation with Delassus matrix.
    def prox_r_F(self, t, q):
        prox_r_F = np.zeros(self.nla_F)
        for contr in self.__gamma_F_contr:
            # G_ii_F = np.max(contr.G_ii_F(t, q[contr.qDOF]))
            # R_F = np.full((contr.nla_F), 1 / G_ii_F if G_ii_F > 0 else G_ii_F)
            # prox_r_F[contr.la_FDOF] = R_F
            prox_r_F[contr.la_FDOF] = contr.prox_r_F(t, q[contr.qDOF])
        return prox_r_F

    def gamma_F(self, t, q, u):
        gamma_F = np.zeros(self.nla_F)
        for contr in self.__gamma_F_contr:
            gamma_F[contr.la_FDOF] = contr.gamma_F(t, q[contr.qDOF], u[contr.uDOF])
        return gamma_F

    def gamma_F_dot(self, t, q, u, a):
        gamma_F_dot = np.zeros(self.nla_F)
        for contr in self.__gamma_F_contr:
            gamma_F_dot[contr.la_FDOF] = contr.gamma_F_dot(
                t, q[contr.qDOF], u[contr.uDOF], a[contr.uDOF]
            )
        return gamma_F_dot

    def xi_F(self, t, q, u_pre, u_post):
        xi_F = np.zeros(self.nla_F)
        for contr in self.__gamma_F_contr:
            xi_F[contr.la_FDOF] = contr.gamma_F(
                t, q[contr.qDOF], u_post[contr.uDOF]
            ) + self.e_F[contr.la_NDOF] * contr.gamma_F(
                t, q[contr.qDOF], u_pre[contr.uDOF]
            )
        return xi_F

    def xi_F_q(self, t, q, u_pre, u_post, scipy_matrix=coo_matrix):
        coo = Coo((self.nla_F, self.nq))
        for contr in self.__gamma_F_contr:
            contr.xi_F_q(t, q[contr.qDOF], u_pre[contr.uDOF], u_post[contr.uDOF], coo)
        return coo.tosparse(scipy_matrix)

    def gamma_F_q(self, t, q, u, scipy_matrix=coo_matrix):
        coo = Coo((self.nla_F, self.nq))
        for contr in self.__gamma_F_contr:
            contr.gamma_F_q(t, q[contr.qDOF], u[contr.uDOF], coo)
        return coo.tosparse(scipy_matrix)

    def gamma_F_u(self, t, q, scipy_matrix=coo_matrix):
        coo = Coo((self.nla_F, self.nu))
        for contr in self.__gamma_F_contr:
            contr.gamma_F_u(t, q[contr.qDOF], coo)
        return coo.tosparse(scipy_matrix)

    def gamma_F_dot_q(self, t, q, u, u_dot, scipy_matrix=coo_matrix):
        coo = Coo((self.nla_F, self.nq))
        for contr in self.__gamma_F_contr:
            contr.gamma_F_dot_q(t, q[contr.qDOF], u[contr.uDOF], u_dot[contr.uDOF], coo)
        return coo.tosparse(scipy_matrix)

    def gamma_F_dot_u(self, t, q, u, u_dot, scipy_matrix=coo_matrix):
        coo = Coo((self.nla_F, self.nu))
        for contr in self.__gamma_F_contr:
            contr.gamma_F_dot_u(t, q[contr.qDOF], u[contr.uDOF], u_dot[contr.uDOF], coo)
        return coo.tosparse(scipy_matrix)

    def W_F(self, t, q, scipy_matrix=coo_matrix):
        coo = Coo((self.nu, self.nla_F))
        for contr in self.__gamma_F_contr:
            contr.W_F(t, q[contr.qDOF], coo)
        return coo.tosparse(scipy_matrix)

    def Wla_F_q(self, t, q, la_F, scipy_matrix=coo_matrix):
        coo = Coo((self.nu, self.nq))
        for contr in self.__gamma_F_contr:
            contr.Wla_F_q(t, q[contr.qDOF], la_F[contr.la_FDOF], coo)
        return coo.tosparse(scipy_matrix)
