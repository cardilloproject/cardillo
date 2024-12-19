from collections import OrderedDict
import numpy as np
import trimesh

from urchin.utils import configure_origin

from cardillo.math import SE3inv, cross3


def link_forward_kinematics(
    urdfpy_sys,
    cfg=None,
    vel=None,
    r_OC0=np.zeros(3),
    A_IC0=np.eye(3),
    v_C0=np.zeros(3),
    C0_Omega_0=np.zeros(0),
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
    r_OC0 : (3,) float
        Position of center of mass (C) of base link
        at configuration described by 'cfg'.
    A_IC0 : (3,3) float
        Transformation between (C) frame of base link and inertial frame
        at configuration described by 'cfg'.
    v_C0 : (3,) float
        Velocity of center of mass (C) of base link
        at state described by 'cfg' and 'vel'.
    C0_Omega_0 : (3,) float
        Angular velocity of base link represented in base link frame (C)
        at state described by 'cfg' and 'vel'.


    Returns
    -------
    H_L0Li : dict or (4,4) float
            A map from links to 4x4 homogenous transform matrices that
            position the link frame relative to the base link's frame.
    ui :  dict or (6,) float
            A map from links to 6-dim array 'u' containing the
            velocity of the center of mass and the angular velocity
            of the link represented with respect to the frame (C), i.e.,
            u =[L0_v_C, B_Omega]
    """
    # Process configuration and velocity value
    joint_cfg = urdfpy_sys._process_cfg(cfg)
    joint_vel = urdfpy_sys._process_cfg(vel)

    # Read link set
    link_set = urdfpy_sys.links

    # Process relative motion of base link to origin
    H_IC0 = np.eye(4)
    H_IC0[:3, :3] = A_IC0
    H_IC0[:3, 3] = r_OC0

    H_L0C0 = urdfpy_sys.base_link.inertial.origin
    H_IL0 = H_IC0 @ SE3inv(H_L0C0)

    # Compute forward kinematics in reverse topological order
    H_L0L = OrderedDict()
    H_IL = OrderedDict()
    H_IJ = OrderedDict()
    H_IC = OrderedDict()
    H_CV = OrderedDict()
    v_C = OrderedDict()
    C_Omega = OrderedDict()
    for child in urdfpy_sys._reverse_topo:
        if child == urdfpy_sys.base_link:
            H_L0L[child] = np.eye(4)
            H_IC[child] = H_IC0
            H_IL[child] = H_IL0
            if len(child.visuals) != 0:
                # TODO: How do we deal with multiple visuals?
                H_LcVc = child.visuals[0].origin
                H_LcCc = child.inertial.origin
                H_CcVc = SE3inv(H_LcCc) @ H_LcVc
            else:
                H_CcVc = np.eye(4)
            H_CV[child] = H_CcVc

            v_C[child] = v_C0
            C_Omega[child] = C0_Omega_0

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
            H_LpCp = parent.inertial.origin
            H_LcCc = child.inertial.origin
            H_CpCc = SE3inv(H_LpCp) @ H_LpLc @ H_LcCc
            H_JCc = SE3inv(H_LpJ) @ H_LpLc @ H_LcCc
            H_CpJ = SE3inv(H_LpCp) @ H_LpJ
            H_L0Lc = H_L0L[parent].dot(H_LpLc)
            H_IJ[joint] = H_IL[parent] @ H_LpJ

            A_CpCc = H_CpCc[:3, :3]
            A_JCc = H_JCc[:3, :3]
            A_ICp = H_IC[parent][:3, :3]
            A_LcCc = H_LcCc[:3, :3]
            A_IJ = H_IJ[joint][:3, :3]
            A_CpJ = H_CpJ[:3, :3]

            Cp_r_CpJ = H_CpJ[:3, 3]
            J_r_JLc = H_JLc[:3, 3]
            Cc_r_LcCc = A_LcCc.T @ H_LcCc[:3, 3]

            Cp_Omega_p = C_Omega[parent]
            Cc_Omega_p = A_CpCc.T @ Cp_Omega_p

            v_Cp = v_C[parent]

            H_L0L[child] = H_L0Lc
            H_ILc = H_IL0 @ H_L0Lc
            H_IL[child] = H_ILc

            H_ICc = H_ILc @ H_LcCc
            H_IC[child] = H_ICc
            A_ICc = H_ICc[:3, :3]

            if len(child.visuals) != 0:
                # TODO: How do we deal with multiple visuals?
                H_LcVc = child.visuals[0].origin
                H_CcVc = SE3inv(H_LcCc) @ H_LcVc
            else:
                H_CcVc = np.eye(4)
            H_CV[child] = H_CcVc

            Cc_Omega_c = Cc_Omega_p + A_JCc.T @ J_omega_JLc  # + J_omega_LpJ (=0)
            C_Omega[child] = Cc_Omega_c

            v_J = v_Cp + A_ICp @ cross3(Cp_Omega_p, Cp_r_CpJ)

            J_omega_IJ = A_CpJ.T @ Cp_Omega_p  # J_omega_ICp + J_omega_CpJ(=0)

            v_C[child] = (
                v_J
                + A_IJ @ (J_r_JLc_dot + cross3(J_omega_IJ, J_r_JLc))
                + A_ICc @ cross3(Cc_Omega_c, Cc_r_LcCc)
            )

    return H_IC, H_IL, H_IJ, H_CV, v_C, C_Omega


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
        - ``floating`` - the v_C followed by the B_Omega values.

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
