import numpy as np
from numpy import sin, cos, pi

from cardillo.math.algebra import A_IK_basic_z, e1, e2, e3
from cardillo.discretization import uniform_knot_vector

from cardillo.model import Model
from cardillo.model.frame import Frame
from cardillo.model.rigid_body import Rigid_body_rel_kinematics
from cardillo.model.bilateral_constraints.explicit import Linear_guidance
from cardillo.model.bilateral_constraints.explicit import Revolute_joint as Revolute_joint_expl
from cardillo.model.bilateral_constraints.implicit import Revolute_joint2D as Revolute_joint_impl
from cardillo.model.bilateral_constraints.implicit import Rigid_connection2D
from cardillo.model.force import Force
from cardillo.model.line_force import Line_force
from cardillo.model.contacts import Sphere_to_plane2D
from cardillo.model.scalar_force_interactions.force_laws import Linear_spring, Linear_damper, Linear_spring_damper
from cardillo.model.scalar_force_interactions import add_rotational_forcelaw
from cardillo.model.classical_beams.planar import Hooke, EulerBernoulli, Inextensible_Euler_bernoulli

from cardillo.solver import Moreau, Generalized_alpha_2, Generalized_alpha_3

import matplotlib.pyplot as plt
import matplotlib.animation as animation

if __name__ == "__main__":
    #--------------------------------------------------------------------------------
    #%% PARAMETERS
    inextensible = True
    # body
    l_bdy = 0.1
    h_bdy = 0.7 * l_bdy
    m_bdy = 3

    # thigh
    m_t = 1.56
    theta_t = 8.5e-3
    l_t = 0.2

    # blade
    width = 0.03    # area width  [m]
    height = 0.003  # area height [m]
    l_bl = 0.2      # length [m]

    A = width * height
    I = width * (height ** 3) / 12
    
    E = 210e9    # elastic modulus (210 kN/mm2) [N m-2]

    EA = E*A    # bending stiffness
    EI = E*I    # bending stiffness
    if inextensible:
        EA = EI    #elongation stiffness


    A_rho0 = 7850 * A  # mass line density [kg m-1] 

    material_model = Hooke(EA, EI)
    p = 2
    nEl = 2
    assert p >= 2
    nQP = int(np.ceil((p + 1)**2 / 2))
    print(f'nQP: {nQP}')
    

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
    h = h_bdy 
    beta0 = pi - 2 * alpha0
    gamma0 = beta0 - (pi/2 - alpha0)
    hight_over_foot = l_t * cos(alpha0) + l_bl * sin(gamma0) + h_bdy / 2
    y0 = hight_over_foot + h

    # velocities
    y_dot0 = 0
    alpha_dot0 = 0
    beta_dot0 = 0

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
    # hip_force_law = Linear_spring_damper(200, 0.2)
    hip_force_law = Linear_spring(200)
    hip = add_rotational_forcelaw(hip_force_law, Revolute_joint_expl)(r_OB1, np.eye(3), q0=np.array([alpha0]), u0=np.array([alpha_dot0]))
    A_IK = A_IK_basic_z(alpha0)
    r_OS0 = r_OB1 - l_t / 2 * A_IK[:, 1] 
    thigh = Rigid_body_rel_kinematics(m_t, theta_t * np.eye(3), hip, main_body, r_OS0=r_OS0, A_IK0=A_IK)
    model.add( hip )
    model.add( thigh )
    model.add(Force(lambda t: f_gravity(t, m=m_t), thigh))

    # blade
    nNd = nEl + p
    X0 = np.linspace(0, l_bl, nNd)
    Xi = uniform_knot_vector(p, nEl)
    for i in range(nNd):
        X0[i] = np.sum(Xi[i+1:i+p+1])
    X0 = X0 * l_bl / p 
    Y0 = np.zeros_like(X0)
    Q = np.hstack((X0, Y0))
    q0 = np.hstack((X0 * cos(gamma0) , X0 * sin(gamma0) + h))
    u0 = np.zeros_like(q0)
    if inextensible:
        blade = Inextensible_Euler_bernoulli(A_rho0, material_model, p, nEl, nQP, Q=Q, q0=q0, u0=u0)
    else:
        blade = EulerBernoulli(A_rho0, material_model, p, nEl, nQP, Q=Q, q0=q0, u0=u0)

    model.add( blade )
    # TODO: gravity for blade
    model.add(Line_force(lambda xi, t: f_gravity(t, A_rho0), blade))

    # knee
    r_OB1 = r_OS0 - l_t / 2 * A_IK[:, 1] 
    knee_force_law = Linear_spring(200)
    knee = Rigid_connection2D(thigh, blade, r_OB1, A_IK, frame_ID2=(1,) )
    # knee = add_rotational_forcelaw(knee_force_law, Revolute_joint_impl)(thigh, blade, r_OB1, A_IK)
    model.add( knee )
    
    # ground
    inclination_angle = 0
    frame = Frame(A_IK=A_IK_basic_z(inclination_angle) )
    r_N = 0.15
    ground = Sphere_to_plane2D(frame, blade, 0, mu, frame_ID=(0,), prox_r_N=r_N, prox_r_T=r_N, e_N=e_N, e_T=e_T)
    model.add( ground )

    # assemble model
    model.assemble()

    #--------------------------------------------------------------------------------
    #%% SIMULATE

    t1 = 0.5 #4*T
    dt = 1e-3

    # build solver and solve the problem
    # solver = Moreau(model, t1, dt)
    # solver = Generalized_alpha_2(model, t1, dt, rho_inf=0.6, newton_tol=1e-6, numerical_jacobian=0)
    solver = Generalized_alpha_3(model, t1, dt, rho_inf=0.5, newton_tol=1e-6, numerical_jacobian=0)
    
    sol = solver.solve()
    t = sol.t
    q = sol.q

    #--------------------------------------------------------------------------------
    #%% VISUALIZE RESULTS
    # set up figure and animation
    fig_anim = plt.figure()
    ax = fig_anim.add_subplot(111, aspect='equal', autoscale_on=True,
                        xlim=(-0.1, l_t), ylim=(-0.1, l_t + l_bl + h_bdy))
    ax.grid()

    line_bdy, = ax.plot([], [], 'k-', lw=2)
    line_t, = ax.plot([], [], 'ko-', lw=2)
    line_bl, = ax.plot([], [], 'k-', lw=2)
    
    floor, = ax.plot([-0.1, l_t + l_bl], [0,0], '-b', lw=2)

    def init():
        """initialize animation"""
        line_bdy.set_data([], [])
        line_t.set_data([], [])
        line_bl.set_data([], [])
        return line_bdy, line_t, line_bl

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

        # blade
        x0, y0, _ = blade.centerline(q[i]).T
        line_bl.set_data(x0, y0)

        return line_bdy, line_t, line_bl

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


