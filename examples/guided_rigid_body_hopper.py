import numpy as np
from numpy import sin, cos, pi

from cardillo.math.algebra import A_IK_basic_z, e1, e2, e3

from cardillo.model import Model
from cardillo.model.frame import Frame
from cardillo.model.rigid_body import Rigid_body_rel_kinematics, Rigid_body_euler
from cardillo.model.bilateral_constraints.explicit import Linear_guidance, Revolute_joint
from cardillo.model.bilateral_constraints.implicit import Revolute_joint as Revolute_joint_impl
from cardillo.model.force import Force
from cardillo.model.contacts import Sphere_to_plane2D
from cardillo.model.scalar_force_interactions.force_laws import Linear_spring, Linear_damper, Linear_spring_damper
from cardillo.model.scalar_force_interactions import add_rotational_forcelaw

from cardillo.solver import Moreau, Generalized_alpha_2, Generalized_alpha_3

import matplotlib.pyplot as plt
import matplotlib.animation as animation

if __name__ == "__main__":
    #--------------------------------------------------------------------------------
    #%% PARAMETERS

    minimal_coordinates = False

    # body
    l_bdy = 0.1
    h_bdy = 0.7 * l_bdy
    m_bdy = 4

    # thigh
    m_t = 1.56
    theta_t = 8.5e-3
    l_t = 0.2

    # shank
    m_s = 0.42/2
    theta_s = 3.5e-3 /8
    l_s = 0.2 / 2

    # foot
    m_f = 0.42/2
    theta_f =  3.5e-3 /8
    l_f = 0.2 / 2

    # ankle
    k_a = 120

    # contact
    mu = 0.1
    e_N = 0
    e_T = 0

    #--------------------------------------------------------------------------------
    #%% INITIAL CONDITIONS
    
    # initial time
    t0 = 0

    # generalized position
    alpha0 = pi/3      
    beta0 = pi - 2 * alpha0
    gamma0 = 0
    hight_over_foot = l_t * cos(alpha0) + l_s * sin(beta0 - (pi/2 - alpha0)) + l_f * sin(gamma0 + beta0 + alpha0 - pi/2) + h_bdy/2
    y0 = hight_over_foot + h_bdy / 2

    # velocities
    y_dot0 = 0
    alpha_dot0 = 0
    beta_dot0 = 0
    gamma_dot0 = 0

    #--------------------------------------------------------------------------------
    #%% Excitation

    A = alpha0 / 5.8
    T = 1 / 4
    omega = 2 * pi / T

    alpha = lambda t: alpha0 - A * (1- cos(omega * t))
    beta = lambda t: pi - 2 * alpha(t)

    #--------------------------------------------------------------------------------
    #%% ASSEMBLE ROBOT

    f_gravity  = lambda t, m: m * np.array([0, -9.81, 0])

    # create assembler object
    model = Model(t0=t0)
    origin = Frame()
    model.add( origin )

    # main body
    r_OS0 = np.array([0, y0, 0])
    guidance = Linear_guidance(origin.r_OP(0), np.vstack((e2, e3, e1)).T, q0=np.array([y0]), u0=np.array([y_dot0]))
    main_body = Rigid_body_rel_kinematics(m_bdy, np.eye(3), guidance, origin, r_OS0=r_OS0)
    model.add( guidance )
    model.add( main_body )
    model.add(Force(lambda t: f_gravity(t, m=m_bdy), main_body))

    # hip and thigh
    r_OB1 = r_OS0 - np.array([0, h_bdy/2, 0])
    hip_force_law = Linear_spring_damper(200, 1)
    hip = add_rotational_forcelaw(hip_force_law, Revolute_joint)(r_OB1, np.eye(3), q0=np.array([alpha0]), u0=np.array([alpha_dot0]))
    A_IK = A_IK_basic_z(alpha0)
    r_OS0 = r_OB1 - l_t / 2 * A_IK[:, 1] 
    thigh = Rigid_body_rel_kinematics(m_t, theta_t * np.eye(3), hip, main_body, r_OS0=r_OS0, A_IK0=A_IK)
    model.add( hip )
    model.add( thigh )
    model.add(Force(lambda t: f_gravity(t, m=m_t), thigh))

    # knee and shank
    r_OB1 = r_OS0 - l_t / 2 * A_IK[:, 1] 
    knee_force_law = Linear_spring(200)
    A_IK = A_IK_basic_z(alpha0 + beta0)
    r_OS0 = r_OB1 + l_s / 2 * A_IK[:, 1] 
    if minimal_coordinates:
        knee = add_rotational_forcelaw(knee_force_law, Revolute_joint)(r_OB1, A_IK, q0=np.array([beta0]), u0=np.array([beta_dot0]))
        shank = Rigid_body_rel_kinematics(m_s, theta_s * np.eye(3), knee, thigh, r_OS0=r_OS0, A_IK0=A_IK)
    else:
        shank = Rigid_body_euler(m_s, theta_s * np.eye(3), q0=np.concatenate([r_OS0, np.array([alpha0 + beta0, 0, 0])]))
        knee = add_rotational_forcelaw(knee_force_law, Revolute_joint_impl)(thigh, shank, r_OB1, A_IK)
    
    model.add( knee )
    model.add( shank )
    model.add(Force(lambda t: f_gravity(t, m=m_s), shank))

    # ankle and foot
    r_OB1 = r_OS0 + l_s / 2 * A_IK[:, 1] 
    ankle_force_law = Linear_spring(200)
    # A_IK = A_IK_basic_z(alpha0 + beta0)
    r_OS0 = r_OB1 + l_f / 2 * A_IK[:, 1] 
    if minimal_coordinates:
        ankle = add_rotational_forcelaw(ankle_force_law, Revolute_joint)(r_OB1, A_IK, q0=np.array([gamma0]), u0=np.array([gamma_dot0]))
        foot = Rigid_body_rel_kinematics(m_f, theta_f * np.eye(3), ankle, shank, r_OS0=r_OS0, A_IK0=A_IK)
    else:
        foot = Rigid_body_euler(m_f, theta_f * np.eye(3), q0=np.concatenate([r_OS0, np.array([alpha0 + beta0, 0, 0])]))
        ankle = add_rotational_forcelaw(ankle_force_law, Revolute_joint_impl)(shank, foot, r_OB1, A_IK)
    model.add( ankle )
    model.add( foot )
    model.add(Force(lambda t: f_gravity(t, m=m_f), foot))
    
    # ground
    inclination_angle = 0
    frame = Frame(A_IK=A_IK_basic_z(inclination_angle) )
    r_N = 0.15
    K_r_SP =  l_f / 2 * e2
    ground = Sphere_to_plane2D(frame, foot, 0, mu, K_r_SP=K_r_SP, prox_r_N=r_N, prox_r_T=r_N, e_N=e_N, e_T=e_T)
    model.add( ground )

    # assemble model
    model.assemble()

    #--------------------------------------------------------------------------------
    #%% SIMULATE

    t1 = 0.6 #4*T
    dt = 5e-3

    # build solver and solve the problem
    # solver = Moreau(model, t1, dt)
    solver = Generalized_alpha_3(model, t1, dt, rho_inf=0.7, numerical_jacobian=False)
    
    sol = solver.solve()
    t = sol.t
    q = sol.q

    #--------------------------------------------------------------------------------
    #%% VISUALIZE RESULTS
    # set up figure and animation
    fig_anim = plt.figure()
    ax = fig_anim.add_subplot(111, aspect='equal', autoscale_on=True,
                        xlim=(-0.1, l_t), ylim=(-0.1, l_t + l_s + l_f + h_bdy))
    ax.grid()

    line_bdy, = ax.plot([], [], 'k-', lw=2)
    line_t, = ax.plot([], [], 'ko-', lw=2)
    line_s, = ax.plot([], [], 'k-', lw=2)
    line_f, = ax.plot([], [], 'ko-', lw=2)
    
    floor, = ax.plot([-0.1, l_t + l_s + l_f], [0,0], '-b', lw=2)

    def init():
        """initialize animation"""
        line_bdy.set_data([], [])
        line_t.set_data([], [])
        line_s.set_data([], [])
        line_f.set_data([], [])
        return line_bdy, line_t, line_s, line_f 

    def animate(i):
        """perform animation step"""
        
        # body
        r_OS = main_body.r_OP(t[i], q[i][main_body.qDOF])
        p1 = r_OS[:2]
        p3 = p1 + np.array( [l_bdy/2 , h_bdy/2] )
        p4 = p1 + np.array( [l_bdy/2 , -h_bdy/2] )
        p5 = p1 + np.array( [-l_bdy/2 , -h_bdy/2] )
        p6 = p1 + np.array( [-l_bdy/2 , h_bdy/2] )
        P2 =  np.array([p3,p4,p5,p6,p3])
        line_bdy.set_data( (P2[:,0], P2[:,1]) )

        # # thigh
        p1 = thigh.r_OP(t[i], q[i][thigh.qDOF], K_r_SP=np.array([0, l_t/2 ,0]))
        p2 = thigh.r_OP(t[i], q[i][thigh.qDOF], K_r_SP=np.array([0, -l_t/2 ,0]))
        
        P1 =  np.array([p1,p2])
        line_t.set_data( (P1[:,0], P1[:,1]) )

        # # shank
        p1 = shank.r_OP(t[i], q[i][shank.qDOF], K_r_SP=np.array([0, l_s/2 ,0]))
        p2 = shank.r_OP(t[i], q[i][shank.qDOF], K_r_SP=np.array([0, -l_s/2 ,0]))
        
        P1 =  np.array([p1,p2])
        line_s.set_data( (P1[:,0], P1[:,1]) )

        # foot
        p1 = foot.r_OP(t[i], q[i][foot.qDOF], K_r_SP=np.array([0, l_f/2 ,0]))
        p2 = foot.r_OP(t[i], q[i][foot.qDOF], K_r_SP=np.array([0, -l_f/2 ,0]))
        
        P1 =  np.array([p1,p2])
        line_f.set_data( (P1[:,0], P1[:,1]) )

        return line_bdy, line_t, line_s, line_f 

    slowmotion = 5
    fps = 50
    animation_time = slowmotion * t1
    target_frames = int(fps * animation_time)
    frac = max(1, int(len(t) / target_frames))
    if frac == 1:
        target_frames = len(t)
    interval = 1000 / fps

    frames = target_frames
    t = t[::frac]
    q = q[::frac]

    anim = animation.FuncAnimation(fig_anim, animate, frames=frames, interval=interval, blit=False)
    
    plt.show()


