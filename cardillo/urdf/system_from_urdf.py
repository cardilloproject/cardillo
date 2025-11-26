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

from pathlib import Path
import numpy as np

# External URDF parser
from urdf_parser_py.urdf import URDF

# Cardillo core classes
from cardillo import System
from cardillo.constraints import Revolute, RigidConnection, Prismatic
from cardillo.discrete import RigidBody, Frame, Meshed, Sphere, Box
from cardillo.forces import Force
from cardillo.math import cross3, norm, ax2skew, ax2skew_squared, A_IB_basic


def rpy_to_A(rpy):
    """
    Convert roll-pitch-yaw (RPY) coordinates to a 3x3 rotation matrix.
    Parameters
    ----------
    rpy : array-like, shape (3,)
        Roll, pitch, yaw angles (radians).
    Returns
    -------
    R : ndarray, shape (3,3)
        Rotation matrix.
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
    """
    Convert a URDF pose to position and rotation matrix.
    Returns
    -------
    r : ndarray, shape (3,)
        Position vector.
    A : ndarray, shape (3,3)
        Rotation matrix.
    """
    if pose is None:  # For spheres, the pose of the origin is None
        return np.zeros(3), np.eye(3)
    else:
        return np.array(pose.position), rpy_to_A(pose.rotation)


def inertia_to_matrix(inertia):
    """
    Convert URDF inertia object to 3x3 inertia matrix.
    """
    ixx = inertia.ixx
    ixy = inertia.ixy
    ixz = inertia.ixz
    iyy = inertia.iyy
    iyz = inertia.iyz
    izz = inertia.izz
    return np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])


def axis_angle_to_A(axis, angle):
    """
    Convert axis-angle to rotation matrix.
    """
    axis = np.asanyarray(axis, dtype=np.float64)
    norm_a = norm(axis)
    if norm_a > 0:
        axis /= norm_a
    else:
        raise ValueError("Zero axis provided for axis-angle representation.")
    return (
        np.eye(3)
        + np.sin(angle) * ax2skew(axis)
        + (1 - np.cos(angle)) * ax2skew_squared(axis)
    )


def process_visual(link, BodyType, kwargs_body, A_RB, R_r_RC, folder_path):
    """
    Add visual geometry to body if present in URDF link.
    """
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
        elif visual_type == "Box":
            BodyType = Box(BodyType)
            kwargs_body["dimensions"] = link.visual.geometry.size
            kwargs_body["B_r_CP"] = A_RB.T @ (R_r_RV - R_r_RC)
            kwargs_body["A_BM"] = A_RB.T @ A_RV
        else:
            print(f"Info: No visual for type {visual_type} implemented.")
            # TODO: implement Cylinder and other visual types
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
    """
    Parse a URDF file and construct a cardillo System instance.

    Parameters
    ----------
    file_path : str or Path
        Path to the URDF file.
    r_OR : ndarray, shape (3,)
        Position of the root reference frame in the inertial frame.
    A_IR : ndarray, shape (3,3)
        Orientation of the root reference frame in the inertial frame.
    v_R : ndarray, shape (3,)
        Linear velocity of the root reference frame.
    R_omega_IR : ndarray, shape (3,)
        Angular velocity of the root reference frame.
    configuration : dict
        Initial joint configuration values.
    velocities : dict
        Initial joint velocity values.
    root_is_floating : bool
        If True, treat the root as a floating rigid body.
    gravitational_acceleration : ndarray, shape (3,)
        Gravity vector to apply to all bodies.

    Returns
    -------
    system : System
        The assembled cardillo System instance.
    """
    system = System()
    folder_path = Path(file_path).parent

    print("-" * 80)
    print(f"Parsing URDF file: {file_path}")
    urdf = URDF.from_xml_file(file_path)
    print("-" * 80)
    print(f"Assembling cardillo system from parsed URDF.")

    # ----------
    # Parse root link and initialize its state
    root = urdf.link_map[urdf.get_root()]
    kwargs_body = {"name": root.name}

    # Set root reference frame properties
    root.r_OR = r_OR
    root.A_IR = A_IR
    root.v_R = v_R
    root.R_omega_IR = R_omega_IR

    # Handle root as floating or fixed
    if root_is_floating:
        if root.inertial is not None and root.inertial.mass > 0:
            R_r_RC, A_RB = pose_to_r_A(root.inertial.origin)
            root.A_IB = A_IR @ A_RB
            root.r_OC = r_OR + A_IR @ R_r_RC
        else:
            raise ValueError(
                "Root must have positive mass and inertial properties if it is floating."
            )
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

    # Add root body/frame to system
    system.add(BodyType(**kwargs_body))

    # ----------
    # Forward kinematics: traverse the URDF tree and add links/joints
    links_to_process = [root]
    while links_to_process:
        parent = links_to_process.pop(0)
        if parent.name not in urdf.child_map:
            continue  # Leaf node, no children
        for item in urdf.child_map[parent.name]:
            joint = urdf.joint_map[item[0]]
            child = urdf.link_map[item[1]]
            links_to_process.append(child)
            BodyType = None
            JointType = None

            # Joint kinematics
            Rp_r_RpJ, A_RpJ = pose_to_r_A(joint.origin)
            JointType, kwargs_joint, J_r_JRc, A_JRc, J_v_JRc, J_omega_JRc = (
                joint_kinematics(parent, joint, configuration, velocities)
            )

            # Compute child state from parent and joint
            child.r_OR = parent.r_OR + parent.A_IR @ (Rp_r_RpJ + A_RpJ @ J_r_JRc)
            child.A_IR = parent.A_IR @ A_RpJ @ A_JRc
            J_omega_IRc = A_RpJ.T @ parent.R_omega_IR + J_omega_JRc
            child.R_omega_IR = A_JRc.T @ J_omega_IRc
            child.v_R = parent.v_R + parent.A_IR @ (
                cross3(parent.R_omega_IR, Rp_r_RpJ)
                + A_RpJ @ (J_v_JRc + cross3(J_omega_IRc, J_r_JRc))
            )

            # Determine body type and add to system
            if child.inertial is not None and child.inertial.mass > 0:
                BodyType = RigidBody
                R_r_RC, A_RB = pose_to_r_A(child.inertial.origin)
                child.A_IB = child.A_IR @ A_RB
                child.r_OC = child.r_OR + child.A_IR @ R_r_RC
            else:
                # Only add leaf frames with no mass if joint is fixed
                if joint.type == "fixed" and child.name not in urdf.child_map:
                    continue  # Skip leaf with no mass
                raise ValueError(
                    f"Link {child.name} has zero mass or no inertia, which will lead to a singular system."
                )

            kwargs_body = {
                "name": child.name,
                "mass": child.inertial.mass,
                "B_Theta_C": inertia_to_matrix(child.inertial.inertia),
                "q0": RigidBody.pose2q(child.r_OC, child.A_IB),
            }
            child.B_Omega = A_RB.T @ child.R_omega_IR
            child.v_C = child.v_R + child.A_IR @ cross3(child.R_omega_IR, R_r_RC)
            kwargs_body["u0"] = np.hstack([child.v_C, child.B_Omega])

            # Add visual geometry if present
            BodyType = process_visual(
                child, BodyType, kwargs_body, A_RB, R_r_RC, folder_path
            )

            if BodyType is None:
                raise ValueError(
                    f"BodyType could not be determined for {child.name}. No body added."
                )
            system.add(BodyType(**kwargs_body))

            # Add joint to system
            if JointType is not None:
                kwargs_joint["subsystem1"] = system.contributions_map[parent.name]
                kwargs_joint["subsystem2"] = system.contributions_map[child.name]
                system.add(JointType(**kwargs_joint))

    # ----------
    # Add gravity forces to all bodies (except frames)
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
    print("-" * 80)
    return system


def joint_kinematics(parent, joint, configuration, velocities):
    """
    Compute the kinematics for a joint and return joint type, parameters, and state.

    Parameters
    ----------
    parent : URDF link object
        The parent link.
    joint : URDF joint object
        The joint connecting parent and child.
    configuration : dict
        Joint configuration values.
    velocities : dict
        Joint velocity values.

    Returns
    -------
    JointType : class or None
        The cardillo joint class (Revolute, Prismatic, etc.) or None for floating.
    kwargs_joint : dict
        Arguments for joint constructor.
    J_r_JRc : ndarray, shape (3,)
        Relative position vector from joint to child.
    A_JRc : ndarray, shape (3,3)
        Relative orientation from joint to child.
    J_v_JRc : ndarray, shape (3,)
        Relative linear velocity from joint to child.
    J_omega_JRc : ndarray, shape (3,)
        Relative angular velocity from joint to child.
    """
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
        A_JRc = axis_angle_to_A(e1, angle)

        if joint.name in velocities:
            angle_dot = float(velocities[joint.name])
        else:
            angle_dot = 0.0
        J_v_JRc = np.zeros(3)
        J_omega_JRc = angle_dot * e1

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

    elif joint.type == "prismatic":
        JointType = Prismatic
        # redefine J-frame for such that axis is its x-axis (only for constructor of Prismatic joint!)
        axis = np.asanyarray(joint.axis, dtype=np.float64)
        e1 = axis / norm(axis)
        if np.abs(e1[0]) == 1:
            e2 = cross3(e1, np.array([0, 1, 0]))
        else:
            e2 = cross3(e1, np.array([1, 0, 0]))

        e2 /= norm(e2)
        e3 = cross3(e1, e2)
        A_JJ_new = np.array([e1, e2, e3]).T

        # now prismatic joint is along x-axis of J-frame
        kwargs_joint["axis"] = 0
        kwargs_joint["r_OJ0"] = parent.r_OR + parent.A_IR @ Rp_r_RpJ
        kwargs_joint["A_IJ0"] = parent.A_IR @ A_RpJ @ A_JJ_new

        # use state of the joint to compute child state relative to joint
        if joint.name in configuration:
            displacement = float(configuration[joint.name])
        else:
            displacement = 0.0

        J_r_JRc = displacement * e1
        A_JRc = np.eye(3)

        if joint.name in velocities:
            velocity = float(velocities[joint.name])
        else:
            velocity = 0.0
        J_v_JRc = velocity * e1
        J_omega_JRc = np.zeros(3)

    elif joint.type == "planar":

        # redefine J-frame for such that axis is its x-axis (only for constructor of Prismatic joint!)
        kwargs_joint["axis"] = 2
        kwargs_joint["r_OJ0"] = parent.r_OR + parent.A_IR @ Rp_r_RpJ
        kwargs_joint["A_IJ0"] = parent.A_IR @ A_RpJ

        # use state of the joint to compute child state relative to joint
        if joint.name in configuration:
            x, y = np.asanyarray(configuration[joint.name], dtype=np.float64)
        else:
            x, y = np.zeros(2)

        J_r_JRc = np.array([x, y, 0.0])
        A_JRc = np.eye(3)

        if joint.name in velocities:
            vx, vy = np.asanyarray(velocities[joint.name], dtype=np.float64)
        else:
            vx, vy = np.zeros(2)
        J_v_JRc = np.array([vx, vy, 0.0])
        J_omega_JRc = np.zeros(3)

    else:
        raise NotImplementedError(f"Joint type {joint.type} not implemented.")
    return JointType, kwargs_joint, J_r_JRc, A_JRc, J_v_JRc, J_omega_JRc
