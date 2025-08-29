"""
Notational convention for bases:
- Inertial frame: I
- Body fixed frame: B (for definition of inertial properties)
- Body fixed frame: V (for definition of visual/mesh)
- Body fixed frame: R (reference frame for root, link)
- Body fixed frame: J (joint frame, body fixed to parent)

Notational convention for points:
- Center of mass: C
- Origin visuals: V
- Origin of reference frame: R (root or link)
- Origin of joint frame: J

Modifiers:
- Child: c (e.g., reference frame child: Rc)
- Parent p (e.g., reference frame parent: Rp)
"""

from logging import root
from pathlib import Path
import numpy as np

from urdf_parser_py.urdf import URDF
from cardillo import System
from cardillo.constraints import Revolute, RigidConnection
from cardillo.discrete import RigidBody, Frame, Meshed, Sphere
from cardillo.forces import Force
from cardillo.math import cross3, norm, ax2skew, ax2skew_squared, A_IB_basic


def rpy_to_A(rpy):
    """Convert roll-pitch-yaw coordinates to a transformation matrix.

    Parameters
    ----------
    coords : (3,) float
        The roll-pitch-yaw coordinates in order (x-rot, y-rot, z-rot).

    Returns
    -------
    R : (3,3) float
        The corresponding 3x3 transformation matrix.
    """
    rpy = np.asanyarray(rpy, dtype=np.float64)
    c3, c2, c1 = np.cos(rpy)
    s3, s2, s1 = np.sin(rpy)

    return np.array(
        [
            [c1 * c2, (c1 * s2 * s3) - (c3 * s1), (s1 * s3) + (c1 * c3 * s2)],
            [c2 * s1, (c1 * c3) + (s1 * s2 * s3), (c3 * s1 * s2) - (c1 * s3)],
            [-s2, c2 * s3, c2 * c3],
        ],
        dtype=np.float64,
    )


def pose_to_r_A(pose):
    if pose is None:  # for spheres, the pose of the origin is None
        return np.zeros(3), np.eye(3)
    else:
        return np.array(pose.position), rpy_to_A(pose.rotation)


def inertia_to_matrix(inertia):
    ixx = inertia.ixx
    ixy = inertia.ixy
    ixz = inertia.ixz
    iyy = inertia.iyy
    iyz = inertia.iyz
    izz = inertia.izz
    return np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])


def axis_angle_to_A(axis, angle):
    axis = np.asanyarray(axis, dtype=np.float64)

    if norm_a := norm(axis) > 0:
        axis /= norm_a
    else:
        raise ValueError("Zero axis provided for axis-angle representation.")

    return (
        np.eye(3)
        + np.sin(angle) * ax2skew(axis)
        + (1 - np.cos(angle)) * ax2skew_squared(axis)
    )


def process_visual(link, BodyType, kwargs_body, A_RB, R_r_RC, folder_path):
    if link.visual is not None:
        if link.visual.origin is None:
            R_r_RV, A_RV = np.zeros(3), np.eye(3)
        else:
            R_r_RV, A_RV = pose_to_r_A(link.visual.origin)

        visual_type = type(link.visual.geometry).__name__
        if visual_type == "Mesh":
            BodyType = Meshed(BodyType)
            kwargs_body["mesh_obj"] = Path(folder_path, link.visual.geometry.filename)
            kwargs_body["B_r_CP"] = A_RB.T @ (R_r_RV - R_r_RC)
            kwargs_body["A_BM"] = A_RB.T @ A_RV
        elif visual_type == "Sphere":
            BodyType = Sphere(BodyType)
            kwargs_body["radius"] = link.visual.geometry.radius
            kwargs_body["B_r_CP"] = A_RB.T @ (R_r_RV - R_r_RC)
            kwargs_body["A_BM"] = A_RB.T @ A_RV
        else:
            print("Info: No visual for type {} implemented.".format(visual_type))
            # TODO: implement Box, Cylinder, and other visual types

    return BodyType


def system_from_urdf(
    file_path,
    r_OR=np.zeros(3),
    A_IR=np.eye(3),
    v_R=np.zeros(3),
    R_omega_IR=np.zeros(3),
    configuration={},
    velocities={},
    root_is_floating=False,
    gravitational_acceleration=None,
):
    system = System(origin_size=0.01)
    folder_path = Path(file_path).parent
    urdf = URDF.from_xml_file(file_path)

    # ----------
    # root to cardillo

    root = urdf.link_map[urdf.get_root()]

    # argument dictionary for call of (meshed) Frame or RigidBody
    kwargs_body = {}
    kwargs_body["name"] = root.name

    # root reference frame
    root.r_OR = r_OR
    root.A_IR = A_IR
    root.v_R = v_R
    root.R_omega_IR = R_omega_IR

    # process root
    if root_is_floating:
        if root.inertial is not None:
            if root.inertial.mass > 0:
                R_r_RC, A_RB = pose_to_r_A(root.inertial.origin)
                root.A_IB = A_IR @ A_RB
                root.r_OC = r_OR + A_IR @ R_r_RC
            else:
                raise ValueError("Root must have positive mass if it is floating.")
        else:
            raise ValueError("Root must have inertial properties if it is floating.")
        BodyType = RigidBody
        kwargs_body["mass"] = root.inertial.mass
        kwargs_body["B_Theta_C"] = inertia_to_matrix(root.inertial.inertia)
        kwargs_body["q0"] = RigidBody.pose2q(root.r_OC, root.A_IB)
        root.B_Omega = A_RB.T @ R_omega_IR
        root.v_C = v_R + A_IR @ cross3(R_omega_IR, R_r_RC)
        kwargs_body["u0"] = np.hstack([root.v_C, root.B_Omega])
    else:
        BodyType = Frame
        if root.inertial is not None:
            R_r_RC, A_RB = pose_to_r_A(root.inertial.origin)
        else:
            R_r_RC = np.zeros(3)
            A_RB = np.eye(3)

        kwargs_body["r_OP"] = r_OR + A_IR @ R_r_RC  # r_OC
        kwargs_body["A_IB"] = A_IR @ A_RB  # A_IB

    BodyType = process_visual(root, BodyType, kwargs_body, A_RB, R_r_RC, folder_path)

    # create and add root
    system.add(BodyType(**kwargs_body))

    # ----------
    # forward kinematics
    links_to_process = [root]
    while links_to_process:
        parent = links_to_process.pop(0)
        if (
            parent.name not in urdf.child_map
        ):  # the link that is processed has no children, i.e., is a leaf
            continue
        for item in urdf.child_map[parent.name]:  # for each child
            joint = urdf.joint_map[item[0]]
            child = urdf.link_map[item[1]]
            links_to_process.append(child)
            BodyType = None
            JointType = None

            # joint kinematics
            Rp_r_RpJ, A_RpJ = pose_to_r_A(joint.origin)

            JointType, kwargs_joint, J_r_JRc, A_JRc, J_v_JRc, J_omega_JRc = (
                joint_kinematics(parent, joint, configuration, velocities)
            )

            # forward kinematics (compute child state)
            child.r_OR = parent.r_OR + parent.A_IR @ (Rp_r_RpJ + A_RpJ @ J_r_JRc)
            child.A_IR = parent.A_IR @ A_RpJ @ A_JRc
            J_omega_IRc = (
                A_RpJ.T @ parent.R_omega_IR + J_omega_JRc
            )  #  Rp_omega_RpJ =0 has been used
            child.R_omega_IR = A_JRc.T @ J_omega_IRc
            child.v_R = parent.v_R + parent.A_IR @ (
                cross3(parent.R_omega_IR, Rp_r_RpJ)
                + A_RpJ @ (J_v_JRc + cross3(J_omega_IRc, J_r_JRc))
            )

            if child.inertial is not None:
                if child.inertial.mass > 0:
                    BodyType = RigidBody
                    R_r_RC, A_RB = pose_to_r_A(child.inertial.origin)
                    child.A_IB = child.A_IR @ A_RB
                    child.r_OC = child.r_OR + child.A_IR @ R_r_RC
                else:
                    if joint.type == "fixed":
                        print("Children of body with zero mass:")
                        if child.name in urdf.child_map:
                            print(urdf.child_map[child.name])
                        print("INFO: child name: {}".format(child.name))
                        continue
                        raise ValueError(
                            "Rigidly attached rigid body with zero mass detected."
                        )
                    else:
                        raise ValueError(
                            "Link {child.name} has zero mass, which will lead to a singular system."
                        )
            else:
                if joint.type == "fixed":
                    print("Children of body with zero mass:")
                    if child.name in urdf.child_map:
                        print(urdf.child_map[child.name])
                    print("INFO: child name: {}".format(child.name))
                    continue
                    raise ValueError(
                        "Rigidly attached rigid body with zero mass detected."
                    )
                else:
                    raise ValueError(
                        f"Link {child.name} has no inertia, which will lead to a singular system."
                    )

            kwargs_body = {}
            kwargs_body["name"] = child.name
            kwargs_body["mass"] = child.inertial.mass
            kwargs_body["B_Theta_C"] = inertia_to_matrix(child.inertial.inertia)
            kwargs_body["q0"] = RigidBody.pose2q(child.r_OC, child.A_IB)
            child.B_Omega = A_RB.T @ child.R_omega_IR
            child.v_C = child.v_R + child.A_IR @ cross3(child.R_omega_IR, R_r_RC)
            kwargs_body["u0"] = np.hstack([child.v_C, child.B_Omega])

            BodyType = process_visual(
                child, BodyType, kwargs_body, A_RB, R_r_RC, folder_path
            )

            # create and add link
            if BodyType is None:
                raise ValueError(
                    "BodyType could not be determined for {}. No body added.".format(
                        child.name
                    )
                )
            else:
                system.add(BodyType(**kwargs_body))

            if JointType is None:
                pass  # floating joint ;-)
            else:
                print("INFO: parent name: {}".format(parent.name))
                kwargs_joint["subsystem1"] = system.contributions_map[parent.name]
                kwargs_joint["subsystem2"] = system.contributions_map[child.name]
                system.add(JointType(**kwargs_joint))
                # print(f"Info: Added {joint.name} as {JointType} to the cardillo system.")

    # add gravity
    if gravitational_acceleration is not None:
        for link_name in urdf.link_map.keys():
            if link_name in system.contributions_map:
                link = system.contributions_map[link_name]
                if not isinstance(link, Frame):
                    system.add(
                        Force(
                            link.mass * gravitational_acceleration,
                            link,
                            name="gravity_" + link_name,
                        )
                    )

    system.assemble()

    return system


def joint_kinematics(parent, joint, configuration, velocities):
    kwargs_joint = {}
    kwargs_joint["name"] = joint.name
    Rp_r_RpJ, A_RpJ = pose_to_r_A(joint.origin)
    if joint.type == "fixed":
        JointType = RigidConnection
        J_r_JRc = np.zeros(3)
        A_JRc = np.eye(3)
        J_v_JRc = np.zeros(3)
        J_omega_JRc = np.zeros(3)

    elif joint.type in ["continuous", "revolute"]:
        JointType = Revolute
        # redefine J-frame for such that axis is its x-axis (only for constructor of Revolute joint!)
        axis = np.asanyarray(joint.axis, dtype=np.float64)
        e1 = axis / norm(axis)
        if np.abs(e1[0]) == 1:
            e2 = cross3(e1, np.array([0, 1, 0]))
        else:
            e2 = cross3(e1, np.array([1, 0, 0]))

        e2 /= norm(e2)
        e3 = cross3(e1, e2)
        A_JJ_new = np.array([e1, e2, e3]).T

        # now revolute joint is around x-axis of J-frame
        kwargs_joint["axis"] = 0
        kwargs_joint["r_OJ0"] = parent.r_OR + parent.A_IR @ Rp_r_RpJ
        kwargs_joint["A_IJ0"] = parent.A_IR @ A_RpJ @ A_JJ_new

        # use state of the joint to compute child state relative to joint
        if joint.name in configuration:
            angle = float(configuration[joint.name])
        else:
            angle = 0.0

        kwargs_joint["angle0"] = angle
        J_r_JRc = np.zeros(3)
        A_JRc = axis_angle_to_A(axis, angle)

        if joint.name in velocities:
            angle_dot = float(velocities[joint.name])
        else:
            angle_dot = 0.0
        J_v_JRc = np.zeros(3)
        J_omega_JRc = angle_dot * axis

    elif joint.type == "floating":
        JointType = None
        if joint.name in configuration:
            cfg = np.asanyarray(configuration[joint.name])
            if len(cfg) == 7:
                J_r_JRc, A_JRc = RigidBody.q2pose(cfg)
            elif len(cfg) == 6:
                J_r_JRc = cfg[0:3]
                A_JRc = rpy_to_A(cfg[3:6])
            else:
                raise ValueError(
                    "Floating joint configuration must be of length 6 (rpy) or 7 (quaternion)."
                )
        else:
            J_r_JRc = np.zeros(3)
            A_JRc = np.eye(3)

        if joint.name in velocities:
            cfg = np.asanyarray(velocities[joint.name])
            if len(cfg) == 6:
                J_v_JRc = cfg[0:3]
                J_omega_JRc = cfg[3:6]
            else:
                raise ValueError(
                    "Floating joint velocity must be of length 6 (linear + angular)."
                )
        else:
            J_v_JRc = np.zeros(3)
            J_omega_JRc = np.zeros(3)
    else:
        raise NotImplementedError(f"Joint type {joint.type} not implemented.")
    return JointType, kwargs_joint, J_r_JRc, A_JRc, J_v_JRc, J_omega_JRc
