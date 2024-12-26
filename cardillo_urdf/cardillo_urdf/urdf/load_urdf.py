from cardillo import System
from urchin import URDF
import trimesh
from cardillo_urdf.urdf import link_forward_kinematics
from cardillo_urdf.discrete import RigidBodyRelKinematics
from cardillo_urdf.joints import RevoluteJoint, FloatingJoint, RigidJoint
import numpy as np
from cardillo.discrete import Meshed, Frame, RigidBody
from cardillo.constraints import Revolute, RigidConnection, Prismatic
from cardillo.forces import Force
from cardillo.math import Spurrier
from cardillo.math import cross3, norm

def add_base_link(system, base_link, S_Omega, H_IS, v_S, H_SV, grav_acc, base_link_is_floating, v_C0, C0_Omega_0):
    
    if base_link_is_floating:
        q0 = np.hstack([H_IS[base_link][:3, 3], Spurrier(H_IS[base_link][:3, :3])])
        u0 = np.hstack([v_S[base_link], S_Omega[base_link]])
        if len(base_link.visuals) != 0:
            mesh = trimesh.util.concatenate(base_link.visuals[0].geometry.meshes)
            c_base_link = Meshed(RigidBody)(
                mesh_obj=mesh,
                B_r_CP=H_SV[base_link][:3, 3],
                A_BM=H_SV[base_link][:3, :3],
                mass=base_link.inertial.mass,
                B_Theta_C=base_link.inertial.inertia,
                q0=q0,
                u0=u0,
            )
        else:
            c_base_link = RigidBody(
                mass=base_link.inertial.mass,
                B_Theta_C=base_link.inertial.inertia,
                q0=q0,
                u0=u0,
            )
        print(
                f"Added link '{base_link.name}' as cardillo body of type 'RigidBody'."
            )
    else:
        if not np.all(np.concatenate([v_C0, C0_Omega_0]) == 0):
            raise ValueError(
                "initial velocities for base_link specified, but base_link is not floating."
            )
        if len(base_link.visuals) != 0:
            mesh = trimesh.util.concatenate(base_link.visuals[0].geometry.meshes)
            
            c_base_link = Meshed(Frame)(
                mesh_obj=mesh,
                B_r_CP=H_SV[base_link][:3, 3],
                A_BM=H_SV[base_link][:3, :3],
                r_OP=H_IS[base_link][:3, 3],
                A_IB=H_IS[base_link][:3, :3],
            )
        else:
            c_base_link = Frame(
                r_OP=H_IS[base_link][:3, 3],
                A_IB=H_IS[base_link][:3, :3],
            )
        print(
                f"Added link '{base_link.name}' as cardillo body of type 'Frame'."
            )

    c_base_link.name = base_link.name
    system.add(c_base_link)

    if (grav_acc is not None) and base_link_is_floating:
        c_link = system.contributions_map[c_base_link.name]
        grav = Force(c_link.mass * grav_acc, c_link)
        grav.name = "gravity_" + c_link.name
        system.add(grav)
        print(f"Added gravity for link '{c_link.name}'.")
    
def add_joints(system, joints, H_IL, urdf_system, initial_config):
    
    for i, joint in enumerate(joints):
            A_IB_child = H_IL[urdf_system.link_map[joint.child]][:3, :3]
            r_OB_child = H_IL[urdf_system.link_map[joint.child]][:3, 3]
            parent_link = system.contributions_map[joint.parent]
            child_link = system.contributions_map[joint.child]
            if joint.joint_type in ["continuous", "revolute", "prismatic"]:
                # construct joint basis with B_child_exJ = joint.axis
                axis = 0
                e1 = joint.axis / norm(joint.axis)
                if np.abs(e1[0]) == 1:
                    e2 = cross3(e1, np.array([0, 1, 0]))
                else:
                    e2 = cross3(e1, np.array([1, 0, 0]))

                e2 /= norm(e2)
                e3 = cross3(e1, e2)
                A_B_child_J = np.array([e1, e2, e3]).T
                A_IJ = A_IB_child @ A_B_child_J
                if joint.joint_type == "revolute":
                    # For revolute joint, keep the existing implementation
                    c_joint = Revolute(
                        parent_link,
                        child_link,
                        axis=axis,
                        angle0=initial_config[joint],
                        r_OB0=r_OB_child,
                        A_IB0=A_IJ,
                    )
                    c_joint.name = joint.name
                    system.add(c_joint)

                    print(
                        f"Added joint '{joint.name}' of type '{joint.joint_type}' as cardillo constraint of type 'Revolute'."
                    )

                elif joint.joint_type == "prismatic":
                    # For prismatic joint, create an instance of the Prismatic class
                    c_joint = Prismatic(
                        parent_link,
                        child_link,
                        axis = axis,
                        r_OB0=r_OB_child,
                        A_IJ0=A_IJ,
                        
                    )
                    c_joint.name = joint.name
                    system.add(c_joint)

                    print(
                        f"Added joint '{joint.name}' of type '{joint.joint_type}' as cardillo constraint of type 'Prismatic'."
                    )
                
            elif joint.joint_type == "fixed":
                c_joint = RigidConnection(parent_link, child_link)
                c_joint.name = joint.name
                system.add(c_joint)

                print(
                    f"Added joint '{joint.name}' of type '{joint.joint_type}' as cardillo constraint of type 'RigidConnection'."
                )
            
            elif joint.joint_type == "floating":
                print(
                    f"Added joint '{joint.name}' of type '{joint.joint_type}' by not adding any cardillo constraint."
                )
            else:
                print(
                    f"Joint '{joint.name}' of type '{joint.joint_type}' could not be added."
                )
    
def add_links(system, links, H_IS, S_Omega, v_S, H_SV, grav_acc):
    for i, link in enumerate(links):
        if i == 0:
            continue
        q0 = np.hstack([H_IS[link][:3, 3], Spurrier(H_IS[link][:3, :3])])
        u0 = np.hstack([v_S[link], S_Omega[link]])
        if len(link.visuals) != 0:
                # extract mesh
                mesh = trimesh.util.concatenate(link.visuals[0].geometry.meshes)

                c_link = Meshed(RigidBody)(
                    mesh_obj=mesh,
                    B_r_CP=H_SV[link][:3, 3],
                    A_BM=H_SV[link][:3, :3],
                    mass=link.inertial.mass,
                    B_Theta_C=link.inertial.inertia,
                    q0=q0,
                    u0=u0,
                )
        else:
                c_link = RigidBody(
                    mass=link.inertial.mass,
                    B_Theta_C=link.inertial.inertia,
                    q0=q0,
                    u0=u0,
                )
        c_link.name = link.name
        system.add(c_link)
        print(
                f"Added link '{link.name}' as cardillo body of type 'RigidBody'."
            )

        if grav_acc is not None:
            c_link = system.contributions_map[c_link.name]
            grav = Force(c_link.mass * grav_acc, c_link)
            grav.name = "gravity_" + c_link.name
            system.add(grav)
            print(f"Added gravity for link '{c_link.name}'.")


def load_urdf(
    system,
    file,
    r_OC0=np.zeros(3),
    A_IC0=np.eye(3),
    v_C0=np.zeros(3),
    C0_Omega_0=np.zeros(3),
    initial_config=None,
    initial_vel=None,
    base_link_is_floating=False,
    gravitational_acceleration=None,
):
    grav_acc = gravitational_acceleration
    urdf_system = URDF.load(file)
    H_IS, H_IL, H_IJ, H_SV, v_S, S_Omega = link_forward_kinematics(
        urdf_system,
        r_OC0=r_OC0,
        A_IC0=A_IC0,
        v_C0=v_C0,
        C0_Omega_0=C0_Omega_0,
        cfg=initial_config,
        vel=initial_vel,
    )
    initial_config = urdf_system._process_cfg(initial_config)
    
    add_base_link(system, urdf_system.base_link, S_Omega, H_IS, v_S, H_SV, grav_acc, base_link_is_floating, v_C0, C0_Omega_0)
    add_links(system,urdf_system.links, H_IS, S_Omega, v_S, H_SV, grav_acc,)
    add_joints(system,urdf_system.joints,H_IL,urdf_system,initial_config)
    system.assemble()