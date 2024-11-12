import numpy as np
import warnings
from copy import deepcopy
from scipy.sparse import diags

from cardillo.utility.coo_matrix import CooMatrix
from cardillo.discrete.frame import Frame
from cardillo.discrete.meshed import Axis
from cardillo.solver import consistent_initial_conditions
from cardillo.visualization import Export

properties = []

properties.extend(["E_kin", "E_pot"])

properties.extend(["M", "Mu_q"])

properties.extend(["h", "h_q", "h_u"])

properties.extend(["q_dot", "q_dot_q", "q_dot_u"])

properties.extend(["g"])
properties.extend(["gamma"])

properties.extend(["c", "c_q", "c_u"])
properties.extend(["g_S"])

properties.extend(["la_tau"])
properties.extend(["tau"])

properties.extend(["g_N"])
properties.extend(["gamma_F", "gamma_F_q"])

properties.extend(["assembler_callback", "step_callback"])


class System:
    """Sparse model implementation which assembles all global objects without
    copying on body and element level.

    Parameters
    ----------
    t0 : float
        Initial time of the initial state of the system.
    origin_size: float
        Origin size for trimesh visualization.
        If origin_size>0, the origin of the system is added as trimesh.axis with the specified origin size. Otherwise the system origin is just a cardillo Frame.

    Notes
    -----

    All model functions which return matrices have :py:class:`scipy.sparse.coo_array`
    as default scipy sparse matrix type (:py:class:`scipy.sparse.spmatrix`).
    This is due to the fact that the assembling of global iteration matrices
    is done using :py:func:`scipy.sparse.bmat` which in a first step transforms
    all matrices to :py:class:`scipy.sparse.coo_array`. A :py:class:`scipy.sparse.coo_array`,
    inherits form :py:class:`scipy.sparse._data_matrix`
    `[1] <https://github.com/scipy/scipy/blob/adc4f4f7bab120ccfab9383aba272954a0a12fb0/scipy/sparse/data.py#L21-L126>`_,
    have limited support for arithmetic operations, only a few operations as
    :py:func:`__neg__`, :py:func:`__imul__`, :py:func:`__itruediv__` are implemented.
    For all other operations the matrix is first transformed to a :py:class:`scipy.sparse.csr_array`
    `[2] <https://github.com/scipy/scipy/blob/adc4f4f7bab120ccfab9383aba272954a0a12fb0/scipy/sparse/base.py#L330-L335>`_.
    Slicing is also not supported for matrices of type :py:class:`scipy.sparse.coo_array`,
    we have to use other formats as :py:class:`scipy.sparse.csr_array` or
    :py:class:`scipy.sparse.csc_array` for that.

    """

    def __init__(self, t0=0, origin_size=0):
        self.t0 = t0
        self.nq = 0
        self.nu = 0
        self.nla_g = 0
        self.nla_gamma = 0
        self.nla_c = 0
        self.nla_tau = 0
        self.ntau = 0
        self.nla_S = 0
        self.nla_N = 0
        self.nla_F = 0

        self.contributions = []
        self.contributions_map = {}
        self.ncontr = 0

        if origin_size > 0:
            self.origin = Axis(Frame)(origin_size=origin_size)
        else:
            self.origin = Frame()

        self.origin.name = "cardillo_origin"
        self.add(self.origin)

    def add(self, *contrs):
        """Adds contributions to the system.

        Parameters
        ----------
        contrs : object or list
            Single object or list of objects to add to the system.
        """
        for contr in contrs:
            if not contr in self.contributions:
                self.contributions.append(contr)
                if not hasattr(contr, "name"):
                    contr.name = "contr" + str(self.ncontr)

                if contr.name in self.contributions_map:
                    new_name = contr.name + "_contr" + str(self.ncontr)
                    print(
                        f"There is another contribution named '{contr.name}' which is already part of the system. Changed the name to '{new_name}' and added it to the system."
                    )
                    contr.name = new_name
                self.contributions_map[contr.name] = contr
                self.ncontr += 1
            else:
                raise ValueError(f"contribution {str(contr)} already added")

    def remove(self, *contrs):
        for contr in contrs:
            if contr in self.contributions:
                self.contributions.remove(contr)
            else:
                raise ValueError(f"no contribution {str(contr)} to remove")

    def pop(self, index):
        self.contributions.pop(index)

    def extend(self, contr_list):
        list(map(self.add, contr_list))

    def deepcopy(self):
        """
        Create a deepcopy of the system.

        Returns:
        --------
        system: cardillo.System
            deepcopy of the system
        """
        return deepcopy(self)

    def set_new_initial_state(self, q0, u0, t0=None, **assemble_kwargs):
        """
        Sets the initial state of the system.

        Parameters:
        -----------
        q0 : np.ndarray
            initial position coordinates
        u0 : np.ndarray
            initial velocity coordinates
        t0 : float
            initial time

        """
        self.t0 = t0 if t0 is not None else self.t0

        # extract final generalized coordiantes and distribute to subsystems
        for contr in self.contributions:
            if hasattr(contr, "nq"):
                contr.q0 = q0[contr.my_qDOF]

        # optionally distribute all other solution fields
        for contr in self.contributions:
            if hasattr(contr, "nu"):
                contr.u0 = u0[contr.my_uDOF]

        self.assemble(**assemble_kwargs)

    def export(self, path, folder_name, solution, overwrite=True, fps=50):
        e = Export(path, folder_name, overwrite, fps, solution)
        for contr in self.contributions:
            if hasattr(contr, "export"):
                e.export_contr(contr, file_name=contr.name)
        return e

    def get_contribution_list(self, contr):
        return getattr(self, f"_{self.__class__.__name__}__{contr}_contr")

    def reset(self):
        for contr in self.contributions:
            if hasattr(contr, "reset"):
                contr.reset()

    def assemble(self, *args, **kwargs):
        """Assembles the system, i.e., counts degrees of freedom, sets connectivities and assembles global initial state.

        Parameters
        ----------
        slice_active_contacts : bool
            When computing consistent initial conditions, slice friction forces to contemplate only those corresponding to active normal contact.
        options : cardillo.solver.SolverOptions
            Solver options for the computation of the constraint/contact forces.
        """
        self.nq = 0
        self.nu = 0
        self.nla_g = 0
        self.nla_gamma = 0
        self.nla_c = 0
        self.nla_tau = 0
        self.ntau = 0
        self.nla_S = 0
        self.nla_N = 0
        self.nla_F = 0
        q0 = []
        u0 = []
        e_N = []
        e_F = []
        self.constant_force_reservoir = False

        for p in properties:
            setattr(self, f"_{self.__class__.__name__}__{p}_contr", [])

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
                contr.my_qDOF = np.arange(0, contr.nq) + self.nq
                contr.qDOF = contr.my_qDOF.copy()
                self.nq += contr.nq
                q0.extend(contr.q0.tolist())

            # if contribution has velocity degrees of freedom address velocity coordinates
            if hasattr(contr, "nu"):
                contr.my_uDOF = np.arange(0, contr.nu) + self.nu
                contr.uDOF = contr.my_uDOF.copy()
                self.nu += contr.nu
                u0.extend(contr.u0.tolist())

            # if contribution has compliance contribution
            if hasattr(contr, "nla_c"):
                contr.la_cDOF = np.arange(0, contr.nla_c) + self.nla_c
                self.nla_c += contr.nla_c

            # if contribution of actuator forces
            if hasattr(contr, "nla_tau"):
                contr.la_tauDOF = np.arange(0, contr.nla_tau) + self.nla_tau
                self.nla_tau += contr.nla_tau

            # if contribution of control inputs
            if hasattr(contr, "ntau"):
                contr.tauDOF = np.arange(0, contr.ntau) + self.ntau
                self.ntau += contr.ntau

            # if contribution has constraints on position level address constraint coordinates
            if hasattr(contr, "nla_g"):
                contr.la_gDOF = np.arange(0, contr.nla_g) + self.nla_g
                self.nla_g += contr.nla_g

            # if contribution has constraints on velocity level address constraint coordinates
            if hasattr(contr, "nla_gamma"):
                contr.la_gammaDOF = np.arange(0, contr.nla_gamma) + self.nla_gamma
                self.nla_gamma += contr.nla_gamma

            # if contribution has stabilization conditions for the kinematic equation
            if hasattr(contr, "nla_S"):
                contr.la_SDOF = np.arange(0, contr.nla_S) + self.nla_S
                self.nla_S += contr.nla_S

            # if contribution has contact
            if hasattr(contr, "nla_N"):
                contr.la_NDOF = np.arange(0, contr.nla_N) + self.nla_N
                self.nla_N += contr.nla_N
                e_N.extend(contr.e_N.tolist())

            # if contribution has friction
            if hasattr(contr, "nla_F"):
                contr.la_FDOF = np.arange(0, contr.nla_F) + self.nla_F
                self.nla_F += contr.nla_F
                e_F.extend(contr.e_F.tolist())

                # identify friction forces with constant force reservoirs
                for i_N, i_F, force_law in contr.friction_laws:
                    if len(i_N) == 0:
                        self.constant_force_reservoir = True

        self.e_N = np.array(e_N)
        self.e_F = np.array(e_F)

        # call assembler callback: call methods that require first an assembly of the system
        self.assembler_callback()

        # compute consisten initial conditions
        self.q0 = np.array(q0)
        self.u0 = np.array(u0)

        # compute constant system parts
        # - parts of the mass matrix
        coo = CooMatrix((self.nu, self.nu))
        if self.__M_contr:
            I_constant_mass_matrix = np.array(
                [
                    (
                        contr.constant_mass_matrix
                        if hasattr(contr, "constant_mass_matrix")
                        else False
                    )
                    for contr in self.__M_contr
                ]
            )
            self.I_M = ~I_constant_mass_matrix
            self.__M_contr = np.array(self.__M_contr)
            for contr in self.__M_contr[I_constant_mass_matrix]:
                coo[contr.uDOF, contr.uDOF] = contr.M(self.t0, self.q0[contr.qDOF])
        self._M0 = coo.tocoo()

        # - compliance matrix
        coo = CooMatrix((self.nla_c, self.nla_c))
        for contr in self.__c_contr:
            coo[contr.la_cDOF, contr.la_cDOF] = contr.c_la_c()
        self._c_la_c0 = coo.tocoo()

        # compute consistent initial conditions
        (
            self.t0,
            self.q0,
            self.u0,
            self.q_dot0,
            self.u_dot0,
            self.la_g0,
            self.la_gamma0,
            self.la_c0,
            self.la_N0,
            self.la_F0,
        ) = consistent_initial_conditions(self, *args, **kwargs)

    def assembler_callback(self):
        for contr in self.__assembler_callback_contr:
            contr.assembler_callback()

    #####################
    # kinematic equations
    #####################
    def q_dot(self, t, q, u):
        q_dot = np.zeros(self.nq, dtype=np.common_type(q, u))
        for contr in self.__q_dot_contr:
            q_dot[contr.my_qDOF] = contr.q_dot(t, q[contr.qDOF], u[contr.uDOF])
        return q_dot

    def q_dot_q(self, t, q, u, format="coo"):
        coo = CooMatrix((self.nq, self.nq))
        for contr in self.__q_dot_q_contr:
            coo[contr.my_qDOF, contr.qDOF] = contr.q_dot_q(
                t, q[contr.qDOF], u[contr.uDOF]
            )
        return coo.asformat(format)

    def q_dot_u(self, t, q, format="coo"):
        coo = CooMatrix((self.nq, self.nu))
        for contr in self.__q_dot_u_contr:
            coo[contr.my_qDOF, contr.uDOF] = contr.q_dot_u(t, q[contr.qDOF])
        return coo.asformat(format)

    def step_callback(self, t, q, u):
        for contr in self.__step_callback_contr:
            q[contr.qDOF], u[contr.uDOF] = contr.step_callback(
                t, q[contr.qDOF], u[contr.uDOF]
            )
        return q, u

    ################
    # total energies
    ################
    def E_pot(self, t, q):
        E_pot = 0
        for contr in self.__E_pot_contr:
            E_pot += contr.E_pot(t, q[contr.qDOF])
        return E_pot

    def E_kin(self, t, q, u):
        E_kin = 0
        for contr in self.__E_kin_contr:
            E_kin += contr.E_kin(t, q[contr.qDOF], u[contr.uDOF])
        return E_kin

    #####################
    # equations of motion
    #####################
    def M(self, t, q, format="coo"):
        if np.any(self.I_M):
            coo = CooMatrix((self.nu, self.nu))
            for contr in self.__M_contr[self.I_M]:  # only loop over variable mass parts
                coo[contr.uDOF, contr.uDOF] = contr.M(t, q[contr.qDOF])
            return coo.asformat(format) + self._M0.asformat(format)
        else:
            return self._M0.asformat(format)

    def Mu_q(self, t, q, u, format="coo"):
        coo = CooMatrix((self.nu, self.nq))
        for contr in self.__Mu_q_contr:
            coo[contr.uDOF, contr.qDOF] = contr.Mu_q(t, q[contr.qDOF], u[contr.uDOF])
        return coo.asformat(format)

    def h(self, t, q, u):
        h = np.zeros(self.nu, dtype=np.common_type(q, u))
        for contr in self.__h_contr:
            h[contr.uDOF] += contr.h(t, q[contr.qDOF], u[contr.uDOF])
        return h

    def h_q(self, t, q, u, format="coo"):
        coo = CooMatrix((self.nu, self.nq))
        for contr in self.__h_q_contr:
            coo[contr.uDOF, contr.qDOF] = contr.h_q(t, q[contr.qDOF], u[contr.uDOF])
        return coo.asformat(format)

    def h_u(self, t, q, u, format="coo"):
        coo = CooMatrix((self.nu, self.nu))
        for contr in self.__h_u_contr:
            coo[contr.uDOF, contr.uDOF] = contr.h_u(t, q[contr.qDOF], u[contr.uDOF])
        return coo.asformat(format)

    ############
    # compliance
    ############
    def la_c(self, t, q, u):
        la_c = np.zeros(self.nla_c, dtype=np.common_type(q, u))
        for contr in self.__c_contr:
            la_c[contr.la_cDOF] = contr.la_c(t, q[contr.qDOF], u[contr.uDOF])
        return la_c

    def c(self, t, q, u, la_c):
        c = np.zeros(self.nla_c, dtype=np.common_type(q, u, la_c))
        for contr in self.__c_contr:
            c[contr.la_cDOF] = contr.c(
                t, q[contr.qDOF], u[contr.uDOF], la_c[contr.la_cDOF]
            )
        return c

    def c_q(self, t, q, u, la_c, format="coo"):
        coo = CooMatrix((self.nla_c, self.nq))
        for contr in self.__c_q_contr:
            coo[contr.la_cDOF, contr.qDOF] = contr.c_q(
                t, q[contr.qDOF], u[contr.uDOF], la_c[contr.la_cDOF]
            )
        return coo.asformat(format)

    def c_u(self, t, q, u, la_c, format="coo"):
        coo = CooMatrix((self.nla_c, self.nu))
        for contr in self.__c_u_contr:
            coo[contr.la_cDOF, contr.uDOF] = contr.c_u(
                t, q[contr.qDOF], u[contr.uDOF], la_c[contr.la_cDOF]
            )
        return coo.asformat(format)

    def c_la_c(self, format="coo"):
        return self._c_la_c0.asformat(format)

    def W_c(self, t, q, format="coo"):
        coo = CooMatrix((self.nu, self.nla_c))
        for contr in self.__c_contr:
            coo[contr.uDOF, contr.la_cDOF] = contr.W_c(t, q[contr.qDOF])
        return coo.asformat(format)

    def Wla_c_q(self, t, q, la_c, format="coo"):
        coo = CooMatrix((self.nu, self.nq))
        for contr in self.__c_q_contr:
            coo[contr.uDOF, contr.qDOF] = contr.Wla_c_q(
                t, q[contr.qDOF], la_c[contr.la_cDOF]
            )
        return coo.asformat(format)

    ###########
    # actuators
    ###########
    def W_tau(self, t, q, format="coo"):
        coo = CooMatrix((self.nu, self.nla_tau))
        for contr in self.__la_tau_contr:
            coo[contr.uDOF, contr.la_tauDOF] = contr.W_tau(t, q[contr.qDOF])
        return coo.asformat(format)

    def la_tau(self, t, q, u):
        la_tau = np.zeros(self.nla_tau, dtype=np.common_type(q, u))
        for contr in self.__la_tau_contr:
            la_tau[contr.la_tauDOF] = contr.la_tau(t, q[contr.qDOF], u[contr.uDOF])
        return la_tau

    def Wla_tau_q(self, t, q, u, format="coo"):
        coo = CooMatrix((self.nu, self.nq))
        for contr in self.__la_tau_contr:
            coo[contr.uDOF, contr.qDOF] = contr.Wla_tau_q(
                t, q[contr.qDOF], u[contr.uDOF]
            )
        return coo.asformat(format)

    def Wla_tau_u(self, t, q, u, format="coo"):
        coo = CooMatrix((self.nu, self.nu))
        for contr in self.__la_tau_contr:
            coo[contr.uDOF, contr.uDOF] = contr.Wla_tau_u(
                t, q[contr.qDOF], u[contr.uDOF]
            )
        return coo.asformat(format)

    def tau(self, t):
        tau = np.zeros(self.ntau)
        for contr in self.__tau_contr:
            tau[contr.tauDOF] = contr.tau(t)
        return tau

    def set_tau(self, tau):
        if callable(tau):
            for contr in self.__tau_contr:
                contr.tau = lambda t: tau(t)[contr.tauDOF]
        else:
            for contr in self.__tau_contr:
                contr.tau = lambda t: tau[contr.tauDOF]

    def set_tau_from_dict(self, tau_dict):
        raise NotImplementedError
        # this is not tested!
        for name, tau in tau_dict.items():
            contr = self.contributions_map[name]
            if callable(tau):
                contr.tau = lambda t: tau(t)[contr.tauDOF]
            else:
                contr.tau = lambda t: tau[contr.tauDOF]

    #########################################
    # bilateral constraints on position level
    #########################################
    def g(self, t, q):
        g = np.zeros(self.nla_g, dtype=q.dtype)
        for contr in self.__g_contr:
            g[contr.la_gDOF] = contr.g(t, q[contr.qDOF])
        return g

    def g_q(self, t, q, format="coo"):
        coo = CooMatrix((self.nla_g, self.nq))
        for contr in self.__g_contr:
            coo[contr.la_gDOF, contr.qDOF] = contr.g_q(t, q[contr.qDOF])
        return coo.asformat(format)

    def g_q_T_mu_q(self, t, q, mu_g, format="coo"):
        coo = CooMatrix((self.nq, self.nq))
        for contr in self.__g_contr:
            coo[contr.qDOF, contr.qDOF] = contr.g_q_T_mu_q(
                t, q[contr.qDOF], mu_g[contr.la_gDOF]
            )
        return coo.asformat(format)

    def W_g(self, t, q, format="coo"):
        coo = CooMatrix((self.nu, self.nla_g))
        for contr in self.__g_contr:
            coo[contr.uDOF, contr.la_gDOF] = contr.W_g(t, q[contr.qDOF])
        return coo.asformat(format)

    def Wla_g_q(self, t, q, la_g, format="coo"):
        coo = CooMatrix((self.nu, self.nq))
        for contr in self.__g_contr:
            coo[contr.uDOF, contr.qDOF] = contr.Wla_g_q(
                t, q[contr.qDOF], la_g[contr.la_gDOF]
            )
        return coo.asformat(format)

    def g_dot(self, t, q, u):
        g_dot = np.zeros(self.nla_g, dtype=np.common_type(q, u))
        for contr in self.__g_contr:
            g_dot[contr.la_gDOF] = contr.g_dot(t, q[contr.qDOF], u[contr.uDOF])
        return g_dot

    # TODO: Assemble chi_g for efficiency
    def chi_g(self, t, q):
        return self.g_dot(t, q, np.zeros(self.nu))

    def g_dot_u(self, t, q, format="coo"):
        coo = CooMatrix((self.nla_g, self.nu))
        for contr in self.__g_contr:
            coo[contr.la_gDOF, contr.uDOF] = contr.g_dot_u(t, q[contr.qDOF])
        return coo.asformat(format)

    def g_dot_q(self, t, q, u, format="coo"):
        coo = CooMatrix((self.nla_g, self.nq))
        for contr in self.__g_contr:
            coo[contr.la_gDOF, contr.qDOF] = contr.g_dot_q(
                t, q[contr.qDOF], u[contr.uDOF]
            )
        return coo.asformat(format)

    def g_ddot(self, t, q, u, u_dot):
        g_ddot = np.zeros(self.nla_g, dtype=np.common_type(q, u, u_dot))
        for contr in self.__g_contr:
            g_ddot[contr.la_gDOF] = contr.g_ddot(
                t, q[contr.qDOF], u[contr.uDOF], u_dot[contr.uDOF]
            )
        return g_ddot

    # TODO: Assemble zeta_g for efficency
    def zeta_g(self, t, q, u):
        return self.g_ddot(t, q, u, np.zeros(self.nu))

    #########################################
    # bilateral constraints on velocity level
    #########################################
    def gamma(self, t, q, u):
        gamma = np.zeros(self.nla_gamma, dtype=np.common_type(q, u))
        for contr in self.__gamma_contr:
            gamma[contr.la_gammaDOF] = contr.gamma(t, q[contr.qDOF], u[contr.uDOF])
        return gamma

    # TODO: Assemble chi_gamma for efficency
    def chi_gamma(self, t, q):
        return self.gamma(t, q, np.zeros(self.nu))

    def gamma_q(self, t, q, u, format="coo"):
        coo = CooMatrix((self.nla_gamma, self.nq))
        for contr in self.__gamma_contr:
            coo[contr.la_gammaDOF, contr.qDOF] = contr.gamma_q(
                t, q[contr.qDOF], u[contr.uDOF]
            )
        return coo.asformat(format)

    def gamma_u(self, t, q, format="coo"):
        coo = CooMatrix((self.nla_gamma, self.nu))
        for contr in self.__gamma_contr:
            coo[contr.la_gammaDOF, contr.uDOF] = contr.gamma_u(t, q[contr.qDOF])
        return coo.asformat(format)

    def gamma_dot(self, t, q, u, u_dot):
        gamma_dot = np.zeros(self.nla_gamma, dtype=np.common_type(q, u, u_dot))
        for contr in self.__gamma_contr:
            gamma_dot[contr.la_gammaDOF] = contr.gamma_dot(
                t, q[contr.qDOF], u[contr.uDOF], u_dot[contr.uDOF]
            )
        return gamma_dot

    def gamma_dot_q(self, t, q, u, u_dot, format="coo"):
        coo = CooMatrix((self.nla_gamma, self.nq))
        for contr in self.__gamma_contr:
            coo[contr.la_gammaDOF, contr.qDOF] = contr.gamma_dot_q(
                t, q[contr.qDOF], u[contr.uDOF], u_dot[contr.uDOF]
            )
        return coo.asformat(format)

    def gamma_dot_u(self, t, q, u, u_dot, format="coo"):
        coo = CooMatrix((self.nla_gamma, self.nu))
        for contr in self.__gamma_contr:
            coo[contr.la_gammaDOF, contr.uDOF] = contr.gamma_dot_u(
                t, q[contr.qDOF], u[contr.uDOF], u_dot[contr.uDOF]
            )
        return coo.asformat(format)

    # TODO: Assemble zeta_gamma for efficency
    def zeta_gamma(self, t, q, u):
        return self.gamma_dot(t, q, u, np.zeros(self.nu))

    def W_gamma(self, t, q, format="coo"):
        coo = CooMatrix((self.nu, self.nla_gamma))
        for contr in self.__gamma_contr:
            coo[contr.uDOF, contr.la_gammaDOF] = contr.W_gamma(t, q[contr.qDOF])
        return coo.asformat(format)

    def Wla_gamma_q(self, t, q, la_gamma, format="coo"):
        coo = CooMatrix((self.nu, self.nq))
        for contr in self.__gamma_contr:
            coo[contr.uDOF, contr.qDOF] = contr.Wla_gamma_q(
                t, q[contr.qDOF], la_gamma[contr.la_gammaDOF]
            )
        return coo.asformat(format)

    #####################################################
    # stabilization conditions for the kinematic equation
    #####################################################
    def g_S(self, t, q):
        g_S = np.zeros(self.nla_S, dtype=q.dtype)
        for contr in self.__g_S_contr:
            g_S[contr.la_SDOF] = contr.g_S(t, q[contr.qDOF])
        return g_S

    def g_S_q(self, t, q, format="coo"):
        coo = CooMatrix((self.nla_S, self.nq))
        for contr in self.__g_S_contr:
            coo[contr.la_SDOF, contr.qDOF] = contr.g_S_q(t, q[contr.qDOF])
        return coo.asformat(format)

    #################
    # normal contacts
    #################
    def g_N(self, t, q):
        g_N = np.zeros(self.nla_N, dtype=q.dtype)
        for contr in self.__g_N_contr:
            g_N[contr.la_NDOF] = contr.g_N(t, q[contr.qDOF])
        return g_N

    def g_N_q(self, t, q, format="coo"):
        coo = CooMatrix((self.nla_N, self.nq))
        for contr in self.__g_N_contr:
            coo[contr.la_NDOF, contr.qDOF] = contr.g_N_q(t, q[contr.qDOF])
        return coo.asformat(format)

    def W_N(self, t, q, format="coo"):
        coo = CooMatrix((self.nu, self.nla_N))
        for contr in self.__g_N_contr:
            coo[contr.uDOF, contr.la_NDOF] = contr.W_N(t, q[contr.qDOF])
        return coo.asformat(format)

    def g_N_dot(self, t, q, u):
        g_N_dot = np.zeros(self.nla_N, dtype=np.common_type(q, u))
        for contr in self.__g_N_contr:
            g_N_dot[contr.la_NDOF] = contr.g_N_dot(t, q[contr.qDOF], u[contr.uDOF])
        return g_N_dot

    def g_N_ddot(self, t, q, u, u_dot):
        g_N_ddot = np.zeros(self.nla_N, dtype=np.common_type(q, u, u_dot))
        for contr in self.__g_N_contr:
            g_N_ddot[contr.la_NDOF] = contr.g_N_ddot(
                t, q[contr.qDOF], u[contr.uDOF], u_dot[contr.uDOF]
            )
        return g_N_ddot

    def xi_N(self, t_pre, t_post, q_pre, q_post, u_pre, u_post):
        xi_N = np.zeros(self.nla_N, dtype=np.common_type(q_post, u_post))
        for contr in self.__g_N_contr:
            xi_N[contr.la_NDOF] = contr.g_N_dot(
                t_post, q_post[contr.qDOF], u_post[contr.uDOF]
            ) + contr.e_N * contr.g_N_dot(t_pre, q_pre[contr.qDOF], u_pre[contr.uDOF])
        return xi_N

    def xi_N_q(self, t_post, q_post, u_post, format="coo"):
        coo = CooMatrix((self.nla_N, self.nq))
        for contr in self.__g_N_contr:
            coo[contr.la_NDOF, contr.qDOF] = contr.g_N_dot_q(
                t_post, q_post[contr.qDOF], u_post[contr.uDOF]
            )
        return coo.asformat(format)

    # TODO: Assemble chi_N for efficency
    def chi_N(self, t, q):
        return self.g_N_dot(t, q, np.zeros(self.nu), dtype=q.dtype)

    def g_N_dot_u(self, t, q, format="coo"):
        warnings.warn(
            "We assume g_N_dot_u(t, q) == W_N(t, q).T. This function will be deleted soon!"
        )
        coo = CooMatrix((self.nla_N, self.nu))
        for contr in self.__g_N_contr:
            coo[contr.la_NDOF, contr.uDOF] = contr.g_N_dot_u(t, q[contr.qDOF])
        return coo.asformat(format)

    def Wla_N_q(self, t, q, la_N, format="coo"):
        coo = CooMatrix((self.nu, self.nq))
        for contr in self.__g_N_contr:
            coo[contr.uDOF, contr.qDOF] = contr.Wla_N_q(
                t, q[contr.qDOF], la_N[contr.la_NDOF]
            )
        return coo.asformat(format)

    #################
    # friction
    #################
    def gamma_F(self, t, q, u):
        gamma_F = np.zeros(self.nla_F, dtype=np.common_type(q, u))
        for contr in self.__gamma_F_contr:
            gamma_F[contr.la_FDOF] = contr.gamma_F(t, q[contr.qDOF], u[contr.uDOF])
        return gamma_F

    def gamma_F_dot(self, t, q, u, u_dot):
        gamma_F_dot = np.zeros(self.nla_F, dtype=np.common_type(q, u, u_dot))
        for contr in self.__gamma_F_contr:
            gamma_F_dot[contr.la_FDOF] = contr.gamma_F_dot(
                t, q[contr.qDOF], u[contr.uDOF], u_dot[contr.uDOF]
            )
        return gamma_F_dot

    def xi_F(self, t_pre, t_post, q_pre, q_post, u_pre, u_post):
        xi_F = np.zeros(self.nla_F, dtype=np.common_type(q_post, u_post))
        for contr in self.__gamma_F_contr:
            xi_F[contr.la_FDOF] = contr.gamma_F(
                t_post, q_post[contr.qDOF], u_post[contr.uDOF]
            ) + contr.e_F * contr.gamma_F(t_pre, q_pre[contr.qDOF], u_pre[contr.uDOF])
        return xi_F

    def xi_F_q(self, t_post, q_post, u_post, format="coo"):
        coo = CooMatrix((self.nla_F, self.nq))
        for contr in self.__gamma_F_contr:
            coo[contr.la_FDOF, contr.qDOF] = contr.gamma_F_q(
                t_post, q_post[contr.qDOF], u_post[contr.uDOF]
            )
        return coo.asformat(format)

    def gamma_F_q(self, t, q, u, format="coo"):
        coo = CooMatrix((self.nla_F, self.nq))
        for contr in self.__gamma_F_q_contr:
            coo[contr.la_FDOF, contr.qDOF] = contr.gamma_F_q(
                t, q[contr.qDOF], u[contr.uDOF]
            )
        return coo.asformat(format)

    def gamma_F_u(self, t, q, format="coo"):
        warnings.warn(
            "We assume gamma_F_u(t, q) == W_F(t, q).T. This function will be deleted soon!"
        )
        coo = CooMatrix((self.nla_F, self.nu))
        for contr in self.__gamma_F_contr:
            coo[contr.la_FDOF, contr.uDOF] = contr.gamma_F_u(t, q[contr.qDOF])
        return coo.asformat(format)

    def gamma_F_dot_q(self, t, q, u, u_dot, format="coo"):
        coo = CooMatrix((self.nla_F, self.nq))
        for contr in self.__gamma_F_contr:
            coo[contr.la_FDOF, contr.qDOF] = contr.gamma_F_dot_q(
                t, q[contr.qDOF], u[contr.uDOF], u_dot[contr.uDOF]
            )
        return coo.asformat(format)

    def gamma_F_dot_u(self, t, q, u, u_dot, format="coo"):
        coo = CooMatrix((self.nla_F, self.nu))
        for contr in self.__gamma_F_contr:
            coo[contr.la_FDOF, contr.uDOF] = contr.gamma_F_dot_u(
                t, q[contr.qDOF], u[contr.uDOF], u_dot[contr.uDOF]
            )
        return coo.asformat(format)

    def W_F(self, t, q, format="coo"):
        coo = CooMatrix((self.nu, self.nla_F))
        for contr in self.__gamma_F_contr:
            coo[contr.uDOF, contr.la_FDOF] = contr.W_F(t, q[contr.qDOF])
        return coo.asformat(format)

    def Wla_F_q(self, t, q, la_F, format="coo"):
        coo = CooMatrix((self.nu, self.nq))
        for contr in self.__gamma_F_contr:
            coo[contr.uDOF, contr.qDOF] = contr.Wla_F_q(
                t, q[contr.qDOF], la_F[contr.la_FDOF]
            )
        return coo.asformat(format)
