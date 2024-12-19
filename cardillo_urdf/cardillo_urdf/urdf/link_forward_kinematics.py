from collections import OrderedDict
import numpy as np
import trimesh

from urchin.utils import configure_origin

from cardillo.math import SE3inv, cross3


def link_forward_kinematics(
    urdfpy_sys,
    cfg=None,
    vel=None,
    r_OS0=np.zeros(3),
    A_IS0=np.eye(3),
    v_S0=np.zeros(3),
    S0_Omega_0=np.zeros(0),
):
    """Computes the poses and velocities of the URDF's links via forward kinematics. Code based on a copy of 'link_fk' form urdfpy:0.0.22.

    Parameters
    ----------
    urdfpy_sys : urdfpy.urdf.URDF
        Instance of URDF defining the system.
    cfg : dict or (n,) float
        A map from joints or joint names to configuration values for
        each joint, or a list containing a value for each actuated joint
        in sorted order from the base link.
        If not specified, all joints are assumed to be in their default
        configurations.
    vel : dict or (n,) float
        A map from joints or joint names to velocity values for
        each joint, or a list containing a value for each actuated joint
        in sorted order from the base link.
        If not specified, all joints are assumed to be at rest.
    r_OS0 : (3,) float
        Position of center of mass (S) of base link
        at configuration described by 'cfg'.
    A_IS0 : (3,3) float
        Transformation between (S) frame of base link and inertial frame
        at configuration described by 'cfg'.
    v_S0 : (3,) float
        Velocity of center of mass (S) of base link
        at state described by 'cfg' and 'vel'.
    S0_Omega_0 : (3,) float
        Angular velocity of base link represented in base link frame (S)
        at state described by 'cfg' and 'vel'.


    Returns
    -------
    H_L0Li : dict or (4,4) float
            A map from links to 4x4 homogenous transform matrices that
            position the link frame relative to the base link's frame.
    ui :  dict or (6,) float
            A map from links to 6-dim array 'u' containing the
            velocity of the center of mass and the angular velocity
            of the link represented with respect to the frame (S), i.e.,
            u =[L0_v_S, K_Omega]
    """
    # Process configuration and velocity value
    joint_cfg = urdfpy_sys._process_cfg(cfg)
    joint_vel = urdfpy_sys._process_cfg(vel)

    # Read link set
    link_set = urdfpy_sys.links

    # Process relative motion of base link to origin
    H_IS0 = np.eye(4)
    H_IS0[:3, :3] = A_IS0
    H_IS0[:3, 3] = r_OS0

    H_L0S0 = urdfpy_sys.base_link.inertial.origin
    H_IL0 = H_IS0 @ SE3inv(H_L0S0)

    # Compute forward kinematics in reverse topological order
    H_L0L = OrderedDict()
    H_IL = OrderedDict()
    H_IJ = OrderedDict()
    H_IS = OrderedDict()
    H_SV = OrderedDict()
    v_S = OrderedDict()
    S_Omega = OrderedDict()
    for child in urdfpy_sys._reverse_topo:
        if child == urdfpy_sys.base_link:
            H_L0L[child] = np.eye(4)
            H_IS[child] = H_IS0
            H_IL[child] = H_IL0
            if len(child.visuals) != 0:
                # TODO: How do we deal with multiple visuals?
                H_LcVc = child.visuals[0].origin
                H_LcSc = child.inertial.origin
                H_ScVc = SE3inv(H_LcSc) @ H_LcVc
            else:
                H_ScVc = np.eye(4)
            H_SV[child] = H_ScVc

            v_S[child] = v_S0
            S_Omega[child] = S0_Omega_0

        else:
            path = urdfpy_sys._paths_to_base[child]
            parent = path[1]
            joint = urdfpy_sys._G.get_edge_data(child, parent)["joint"]

            # Check if parent is already processed
            if parent not in H_L0L:
                raise ValueError("Topology contains disconnected graphs.")

            cfg = None
            vel = None
            if joint.mimic is not None:
                mimic_joint = urdfpy_sys.joint_map[joint.mimic.joint]
                if mimic_joint in joint_cfg:
                    cfg = joint.mimic.multiplier * joint_cfg[mimic_joint] + joint.mimic.offset
                if mimic_joint in joint_vel:
                    vel = joint.mimic.multiplier * joint_vel[mimic_joint]
            else:
                if joint in joint_cfg:
                    cfg = joint_cfg[joint]
                if joint in joint_vel:
                    vel = joint_vel[joint]

            H_JLc, J_r_JLc_dot, J_omega_JLc = get_child_state(joint, cfg=cfg, vel=vel)

            H_LpJ = joint.origin
            H_LpLc = H_LpJ @ H_JLc
            H_LpSp = parent.inertial.origin
            H_LcSc = child.inertial.origin
            H_SpSc = SE3inv(H_LpSp) @ H_LpLc @ H_LcSc
            H_JSc = SE3inv(H_LpJ) @ H_LpLc @ H_LcSc
            H_SpJ = SE3inv(H_LpSp) @ H_LpJ
            H_L0Lc = H_L0L[parent].dot(H_LpLc)
            H_IJ[joint] = H_IL[parent] @ H_LpJ

            A_SpSc = H_SpSc[:3, :3]
            A_JSc = H_JSc[:3, :3]
            A_ISp = H_IS[parent][:3, :3]
            A_LcSc = H_LcSc[:3, :3]
            A_IJ = H_IJ[joint][:3, :3]
            A_SpJ = H_SpJ[:3, :3]

            Sp_r_SpJ = H_SpJ[:3, 3]
            J_r_JLc = H_JLc[:3, 3]
            Sc_r_LcSc = A_LcSc.T @ H_LcSc[:3, 3]

            Sp_Omega_p = S_Omega[parent]
            Sc_Omega_p = A_SpSc.T @ Sp_Omega_p

            v_Sp = v_S[parent]

            H_L0L[child] = H_L0Lc
            H_ILc = H_IL0 @ H_L0Lc
            H_IL[child] = H_ILc

            H_ISc = H_ILc @ H_LcSc
            H_IS[child] = H_ISc
            A_ISc = H_ISc[:3, :3]

            if len(child.visuals) != 0:
                # TODO: How do we deal with multiple visuals?
                H_LcVc = child.visuals[0].origin
                H_ScVc = SE3inv(H_LcSc) @ H_LcVc
            else:
                H_ScVc = np.eye(4)
            H_SV[child] = H_ScVc

            Sc_Omega_c = Sc_Omega_p + A_JSc.T @ J_omega_JLc  # + J_omega_LpJ (=0)
            S_Omega[child] = Sc_Omega_c

            v_J = v_Sp + A_ISp @ cross3(Sp_Omega_p, Sp_r_SpJ)

            J_omega_IJ = A_SpJ.T @ Sp_Omega_p  # J_omega_ISp + J_omega_SpJ(=0)

            v_S[child] = (
                v_J
                + A_IJ @ (J_r_JLc_dot + cross3(J_omega_IJ, J_r_JLc))
                + A_ISc @ cross3(Sc_Omega_c, Sc_r_LcSc)
            )

    return H_IS, H_IL, H_IJ, H_SV, v_S, S_Omega


def get_child_state(joint, cfg=None, vel=None):
    """Computes the child state relative to a parent state for a given
    configuration value. Based on get_child_pose of urdfpy:0.0.22

    Parameters
    ----------
    cfg : float, (2,) float, (6,) float, or (4,4) float
        The configuration values for this joint. They are interpreted
        based on the joint type as follows:

        - ``fixed`` - not used.
        - ``prismatic`` - a translation along the axis in meters.
        - ``revolute`` - a rotation about the axis in radians.
        - ``continuous`` - a rotation about the axis in radians.
        - ``planar`` - the x and y translation values in the plane.
        - ``floating`` - the xyz values followed by the rpy values,
            or a (4,4) matrix.

        If ``cfg`` is ``None``, then this just returns the joint pose.

    vel : float, (2,) float, (6,) float, or (4,4) float
        The configuration values for this joint. They are interpreted
        based on the joint type as follows:

        - ``fixed`` - not used.
        - ``prismatic`` - a translational vel. along the axis in meters per second.
        - ``revolute`` - a rotation vel. about the axis in radians per second.
        - ``continuous`` - a rotation vel. about the axis in radians per second.
        - ``planar`` - the x and y translational vel. values in the plane.
        - ``floating`` - the v_S followed by the K_Omega values.

        If ``vel`` is ``None``, the joint velocity is assumed to be zero.

    Returns
    -------
    H_JLc : (4,4) float
        The pose of the child relative to the parent.
    rel_vel : (6,) float
        gen. velocity of the child link relative to the parent. ui = [(J_r_JLc)_dot, J_omega_JLc]
    """
    # (J) frame = joint frame in reference configuration (body fixed with parent link)
    # (Lc) frame = child frame in actual configuration
    # for joint coordinates equal zero, (J)=(Lc)

    H_LpJ = joint.origin
    if cfg is None:
        H_JLc = np.eye(4)
    elif joint.joint_type == "fixed":
        H_JLc = np.eye(4)
    elif joint.joint_type in ["revolute", "continuous"]:
        if cfg is None:
            cfg = 0.0
        else:
            cfg = float(cfg)
        H_JLc = trimesh.transformations.rotation_matrix(cfg, joint.axis)
    elif joint.joint_type == "prismatic":
        if cfg is None:
            cfg = 0.0
        else:
            cfg = float(cfg)
        H_JLc = np.eye(4, dtype=np.float64)
        H_JLc[:3, 3] = joint.axis * cfg
    elif joint.joint_type == "planar":
        if cfg is None:
            cfg = np.zeros(2, dtype=np.float64)
        else:
            cfg = np.asanyarray(cfg, dtype=np.float64)
        if cfg.shape != (2,):
            raise ValueError("(2,) float configuration required for planar joints")
        H_JLc = np.eye(4, dtype=np.float64)
        H_JLc[:3, 3] = H_LpJ[:3, :2].dot(cfg)
    elif joint.joint_type == "floating":
        if cfg is None:
            H_JLc = np.eye(4, dtype=np.float64)
        else:
            H_JLc = configure_origin(cfg)
        if H_JLc is None:
            raise ValueError("Invalid configuration for floating joint")
    else:
        raise ValueError("Invalid configuration")

    # compute v = (J_r_JLc)_dot, omega = J_omega_JLc
    if vel is None:
        v = np.zeros(3)
        omega = np.zeros(3)
    elif joint.joint_type == "fixed":
        v = np.zeros(3)
        omega = np.zeros(3)
    elif joint.joint_type in ["revolute", "continuous"]:
        if vel is None:
            vel = 0.0
        else:
            vel = float(vel)
        J_omega_JLc = vel * joint.axis
        v = np.zeros(3)
        omega = J_omega_JLc  # + J_omega_LpJ (=0)

    elif joint.joint_type == "prismatic":
        if vel is None:
            vel = 0.0
        else:
            vel = float(vel)
        J_r_JLc_dot = vel * joint.axis
        omega = np.zeros(3)
        v = J_r_JLc_dot

    elif joint.joint_type == "planar":
        # if cfg is None:
        #     cfg = np.zeros(2, dtype=np.float64)
        # else:
        #     cfg = np.asanyarray(cfg, dtype=np.float64)
        # if cfg.shape != (2,):
        #     raise ValueError(
        #         '(2,) float configuration required for planar joints'
        #     )
        # translation = np.eye(4, dtype=np.float64)
        # translation[:3,3] = joint.origin[:3,:2].dot(cfg)
        # rel_pose = joint.origin.dot(translation)
        raise NotImplementedError(
            "Forward kinematics for velocities of ``planar`` joint not implemented."
        )
    elif joint.joint_type == "floating":
        if vel is None:
            vel = np.zeros(6, dtype=np.float64)
        v = vel[:3]
        omega = vel[3:]

    else:
        raise ValueError("Invalid velocity.")

    return H_JLc, v, omega  # v = (J_r_JLc)_dot, omega = J_omega_JLc
