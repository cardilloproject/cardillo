import numpy as np
from cardillo_urdf.urdf import load_urdf

from cardillo import System
from cardillo.math import A_IK_basic
from pathlib import Path
from cardillo.solver import Rattle, Moreau, BackwardEuler
from cardillo.discrete import Box, Frame
from cardillo.contacts import Sphere2Plane
from cardillo.actuators import PDcontroller, Motor
from cardillo.visualization import Export
from cardillo.visualization.trimesh import animate_system

from cardillo.actuators._base import BaseActuator

from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve
from cardillo.definitions import IS_CLOSE_ATOL
from cardillo.solver._base import compute_I_F


class VM_controller(BaseActuator):
    def __init__(self, system, trunk, joint_names, tau, kp=1, kd=0.1):
        super().__init__(trunk, tau, nla_tau=len(joint_names), ntau=6)
        self.system = system
        self.joint_names = joint_names
        self.kp = kp
        self.kd = kd

    def assembler_callback(self):
        self.qDOF = np.arange(self.system.nq)
        self._nq = len(self.qDOF)
        self.uDOF = np.arange(self.system.nu)
        self._nu = len(self.uDOF)

    def F(self, t, q, u):
        q_t = q[self.subsystem.qDOF]
        u_t = u[self.subsystem.uDOF]
        r_OS = self.tau(t)[:3]
        F = -self.kp * (self.subsystem.r_OP(t, q_t) - r_OS)
        print(F)
        return F

    def W_tau(self, t, q):
        W_tau = np.zeros((self._nu, self.nla_tau))
        for i, joint_name in enumerate(self.joint_names):
            contr = self.system.contributions_map[joint_name]
            W_tau[contr.uDOF[:, None], i] = contr.W_l(t, q[contr.qDOF])
        return W_tau

    def W_tau_q(self, t, q):
        W_tau_q = np.zeros((self._nu, self.nla_tau, self._nq))
        for i, joint_name in enumerate(self.joint_names):
            contr = self.system.contributions_map[joint_name]
            W_tau_q[contr.uDOF[:, None], i, contr.qDOF] = contr.W_l_q(t, q[contr.qDOF])[
                :, 0, :
            ]
        return W_tau_q

    def la_tau(self, t, q, u):
        J_S = np.zeros((3, self.system.nu))
        q_t = q[self.subsystem.qDOF]
        J_S[:, self.subsystem.uDOF] = self.subsystem.J_P(t, q_t)
        M = self.system.M(t, q)
        h = self.system.h(t, q, u)
        W_g = self.system.W_g(t, q)
        W_gamma = self.system.W_gamma(t, q)
        W_c = self.system.W_c(t, q)
        la_c = self.system.la_c(t, q, u)
        zeta_g = self.system.zeta_g(t, q, u)
        zeta_gamma = self.system.zeta_gamma(t, q, u)

        # compute constant contact quantities
        g_N = self.system.g_N(t, q)
        g_N_dot = self.system.g_N_dot(t, q, u)
        A_N = np.isclose(g_N, np.zeros(self.system.nla_N), atol=IS_CLOSE_ATOL)
        B_N = A_N * np.isclose(g_N_dot, np.zeros(self.system.nla_N), atol=IS_CLOSE_ATOL)
        # get set of active normal contacts
        B_N = np.where(B_N)[0]
        B_F, global_active_friction_laws = compute_I_F(B_N, system, slice=True)

        W_N = self.system.W_N(t, q, format="csc")[:, B_N]
        W_F = self.system.W_F(t, q, format="csc")[:, B_F]
        zeta_N = self.system.g_N_ddot(t, q, u, np.zeros_like(u))[B_N]
        zeta_F = self.system.gamma_F_dot(t, q, u, np.zeros_like(u))[B_F]

        W_tau = self.system.W_tau(t, q, format="csc")
        # Build matrix A for computation of new velocities and bilateral constraint percussions
        # fmt: off
        A = bmat([[         M, -W_g, -W_gamma, -W_N, -W_F], \
                  [    -W_g.T, None,     None, None, None], \
                  [-W_gamma.T, None,     None, None, None],
                  [-W_N.T, None,     None, None, None],
                  [-W_F.T, None,     None, None, None]], format="csc")
        # fmt: on

        # initial right hand side without contact forces
        b = np.concatenate(
            (
                h + W_c @ la_c + J_S.T @ self.F(t, q, u),
                zeta_g,
                zeta_gamma,
                zeta_N,
                zeta_F,
            )
        )
        x = spsolve(A, b)
        x[:6] = 0
        f = bmat([[M, -W_g, -W_gamma, -W_N, -W_F]], format="csc") @ x - h
        la = np.linalg.pinv(W_tau.todense()) @ f
        print(la)
        return la


if __name__ == "__main__":
    from os import path

    dir_name = path.dirname(__file__)

    PD_joint_controller = False
    virtual_model_controller = True

    # Method 1
    initial_config = {}
    initial_config["FR_hip_joint"] = -np.pi / 12  # np.pi / 2
    initial_config["FR_thigh_joint"] = np.pi / 4
    initial_config["FR_calf_joint"] = -np.pi / 2

    initial_config["FL_hip_joint"] = np.pi / 12  # np.pi / 2
    initial_config["FL_thigh_joint"] = np.pi / 4
    initial_config["FL_calf_joint"] = -np.pi / 2

    initial_config["RL_hip_joint"] = np.pi / 12  # np.pi / 2
    initial_config["RL_thigh_joint"] = np.pi / 4
    initial_config["RL_calf_joint"] = -np.pi / 2

    initial_config["RR_hip_joint"] = -np.pi / 12  # np.pi / 2
    initial_config["RR_thigh_joint"] = np.pi / 4
    initial_config["RR_calf_joint"] = -np.pi / 2

    joint_names = [name for name in initial_config]

    initial_vel = {}
    # initial_vel["world_trunk"] = np.array([0.5, 0.5, 0, 0, 0, 0])
    # initial_vel["joint2"] = 5

    # Method 2
    # initial_config = (np.pi / 2, np.pi/2)
    # initial_vel = (1, 1)
    system = System()
    load_urdf(
        system,
        path.join(
            dir_name,
            "urdf",
            "go1.urdf",
        ),
        r_OS0=np.array([0, 0, 0.3]),
        A_IS0=A_IK_basic(0).y(),
        v_S0=np.array([0, 0, 0]),
        S0_Omega_0=np.array([0, 0, 0]),
        initial_config=initial_config,
        initial_vel=initial_vel,
        base_link_is_floating=False,
        gravitational_acceleration=np.array([0, 0, -10]),
        redundant_coordinates=False,
    )

    radius = 0.022
    mu = 0.3
    foot_contact_FR = Sphere2Plane(
        system.origin, system.contributions_map["FR_foot"], mu=mu, r=radius
    )
    foot_contact_FL = Sphere2Plane(
        system.origin, system.contributions_map["FL_foot"], mu=mu, r=radius
    )
    foot_contact_RR = Sphere2Plane(
        system.origin, system.contributions_map["RR_foot"], mu=mu, r=radius
    )
    foot_contact_RL = Sphere2Plane(
        system.origin, system.contributions_map["RL_foot"], mu=mu, r=radius
    )
    system.add(foot_contact_FR, foot_contact_FL, foot_contact_RR, foot_contact_RL)

    if virtual_model_controller:
        trunk = system.contributions_map["trunk"]
        r_OS0 = trunk.r_OP(0, trunk.q0)
        controller = VM_controller(
            system,
            trunk,
            joint_names,
            tau=lambda t: np.concatenate([r_OS0, np.zeros(3)]),
            kp=1000,
        )
        system.add(controller)

    if PD_joint_controller:
        kp = 50
        kd = 1
        for joint_name in joint_names:
            # controller = PDcontroller(system.contributions_map[joint_name], kp, kd, np.array([0, 0]))
            motor = PDcontroller(
                system.contributions_map[joint_name],
                kp,
                kd,
                np.array([initial_config[joint_name], 0]),
            )
            motor.name = "PD_" + system.contributions_map[joint_name].name
            system.add(motor)

        # kp_hip = 30
        # system.contributions_map["PD_FR_hip_joint"].kp = kp_hip
        # system.contributions_map["PD_FL_hip_joint"].kp = kp_hip
        # system.contributions_map["PD_RR_hip_joint"].kp = kp_hip
        # system.contributions_map["PD_RL_hip_joint"].kp = kp_hip
        frq = 1.25
        angle_d_calf = -np.deg2rad(30)
        for joint_name in [
            "FR_calf_joint",
            "FL_calf_joint",
            "RR_calf_joint",
            "RL_calf_joint",
        ]:
            system.contributions_map["PD_" + joint_name].tau = lambda t: np.array(
                [
                    initial_config[joint_name]
                    + angle_d_calf * np.sin(2 * np.pi * frq * t),
                    angle_d_calf * 2 * np.pi * frq * np.cos(2 * np.pi * frq * t),
                ]
            )

            # angle_d_thigh = np.deg2rad(2)
            # for joint_name in ["FR_thigh_joint", "FL_thigh_joint", "RR_thigh_joint", "RL_thigh_joint"]:
            #     system.contributions_map["PD_" + joint_name].tau = lambda t: np.array([initial_config[joint_name] + angle_d_thigh * np.sin(2 * np.pi * frq * t), angle_d_thigh * 2 * np.pi * frq * np.cos(2 * np.pi * frq * t)])

    plane = Box(Frame)(dimensions=[4, 4, 0.001], axis=2)
    plane.name = "Plane"
    system.add(plane)
    system.assemble()
    # sol = Rattle(system, 0.5, 1e-2).solve()
    sol = Moreau(system, 0.25, 1e-3).solve()
    # from cardillo.solver import SolverOptions
    # sol = BackwardEuler(system, 1.5, 1e-2, options=SolverOptions(reuse_lu_decomposition=True)).solve()

    # animate_system(system, sol.t, sol.q)
    if True:
        path = Path(__file__)
        e = Export(
            path=path.parent,
            folder_name=path.stem,
            overwrite=True,
            fps=30,
            solution=sol,
        )

        for b in system.contributions:
            if hasattr(b, "export") and not ("contr" in b.name):
                print(f"exporting {b.name}")
                # print(f"b.r_OP(0, sol.q[0, b.qDOF]) = {b.r_OP(0, sol.q[1, b.qDOF])}")
                e.export_contr(b)
