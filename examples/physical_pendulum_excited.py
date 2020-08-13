from cardillo.model.rigid_body import rigid_body_quaternion
import numpy as np
from scipy.integrate import solve_ivp
from math import cos, sin, sqrt, tan
import matplotlib.pyplot as plt

from cardillo.math.algebra import A_IK_basic_z, axis_angle2quat, cross3, ax2skew
from cardillo.model import Model
from cardillo.model.rigid_body import Rigid_body_euler, Rigid_body_quaternion, Rigid_body_director, Rigid_body2D
from cardillo.model.frame import Frame
from cardillo.model.bilateral_constraints.implicit import Revolute_joint, Spherical_joint, Spherical_joint2D, Rigid_connection, Rigid_connection2D
from cardillo.model.force import Force
from cardillo.solver import Moreau, Moreau_sym, Euler_backward, Generalized_alpha_1, Scipy_ivp, Generalized_alpha_2, Generalized_alpha_3


if __name__ == "__main__":

    # rigid_body = 'Euler'
    # rigid_body = 'Quaternion'
    # rigid_body = 'Quaternion_connected'
    # rigid_body = 'Director'
    rigid_body = 'Director_connected'
    # rigid_body = 'Rigid_body2D'
    # rigid_body = 'Rigid_body2D_connected'
    
    #%% parameters
    m = 1
    L = 2
    theta = 1 / 12 * m * (L ** 2)
    theta_O = theta + m * (L ** 2) / 4
    theta1 = theta2 = 1 / 12 * m / 2 * (L ** 2) / 4
    K_theta_S = theta * np.eye(3)
    K_theta_S1 = K_theta_S2 = theta1 * np.eye(3)
    g = 9.81
    omega = 10
    A = L / 10

    # e = lambda t: A * np.cos(omega * t)
    # e_t = lambda t: -A * omega * np.sin(omega * t)
    # e_tt = lambda t: -A * omega * omega * np.cos(omega * t)

    e = lambda t: A * np.sin(omega * t)
    e_t = lambda t: A * omega * np.cos(omega * t)
    e_tt = lambda t: -A * omega * omega * np.sin(omega * t)

    # e = lambda t: A * t
    # e_t = lambda t: A 
    # e_tt = lambda t: 0

    # e = lambda t: 0
    # e_t = lambda t: 0 
    # e_tt = lambda t: 0

    r_OP = lambda t: np.array([e(t), 0, 0]) 
    v_P = lambda t: np.array([e_t(t), 0, 0]) 
    a_P = lambda t: np.array([e_tt(t), 0, 0]) 

    K_r_SP = np.array([0, L / 2, 0])      # center of mass single rigid body
    K_r_SP1 = np.array([0, L / 4, 0])     # center of mass half rigid body 1
    K_r_SP2 = np.array([0, 3 * L / 4, 0]) # center of mass half rigid body 2

    phi0 = 0.5
    phi_dot0 = 0
    K_omega0 = np.array([0, 0, phi_dot0])
    A_IK0 = A_IK_basic_z(phi0)

    # single rigid body
    r_OS0 = r_OP(0) - A_IK0 @ K_r_SP
    v_S0 = v_P(0) + A_IK0 @ ( cross3(K_omega0, K_r_SP) )
    
    # connected rigid bodies
    r_OS10 = r_OP(0) - A_IK0 @ K_r_SP1
    v_S10 = v_P(0) + A_IK0 @ ( cross3(K_omega0, K_r_SP1) )
    r_OS20 = r_OP(0) - A_IK0 @ K_r_SP2
    v_S20 = v_P(0) + A_IK0 @ ( cross3(K_omega0, K_r_SP2) )

    model = Model()

    frame = Frame(r_OP=r_OP, r_OP_t=v_P, r_OP_tt=a_P)
    model.add(frame)

    if rigid_body == 'Euler':
        p0 = np.array([phi0, 0, 0])
        q0 = np.concatenate((r_OS0, p0))
        u0 = np.concatenate((v_S0, K_omega0))
        RB = Rigid_body_euler(m, K_theta_S, q0=q0, u0=u0)
        model.add(RB)
        model.add(Force(np.array([0, -m*g, 0]), RB))
        model.add(Revolute_joint(frame, RB, r_OP(0), np.eye(3)))
    elif rigid_body == 'Quaternion':
        p0 = axis_angle2quat(np.array([0, 0, 1]), phi0)
        q0 = np.concatenate((r_OS0, p0))
        u0 = np.concatenate((v_S0, K_omega0))
        RB = Rigid_body_quaternion(m, K_theta_S, q0=q0, u0=u0)
        model.add(RB)
        model.add(Force(np.array([0, -m*g, 0]), RB))
        model.add(Revolute_joint(frame, RB, r_OP(0), np.eye(3)))
    elif rigid_body == 'Quaternion_connected':
        p0 = axis_angle2quat(np.array([0, 0, 1]), phi0)
        q10 = np.concatenate((r_OS10, p0))
        q20 = np.concatenate((r_OS20, p0))
        u10 = np.concatenate((v_S10, K_omega0))
        u20 = np.concatenate((v_S20, K_omega0))
        RB1 = Rigid_body_quaternion(m / 2, K_theta_S1, q0=q10, u0=u10)
        RB2 = Rigid_body_quaternion(m / 2, K_theta_S2, q0=q20, u0=u20)
        model.add(RB1)
        model.add(RB2)
        gravity1 = Force(np.array([0, -m / 2 * g, 0]), RB1)
        model.add(gravity1)
        gravity2 = Force(np.array([0, -m / 2 * g, 0]), RB2)
        model.add(gravity2)
        model.add(Revolute_joint(frame, RB1, r_OP(0), np.eye(3)))
        model.add(Rigid_connection(RB1, RB2, r_OS0))
    elif rigid_body == 'Director':
        p0 = np.concatenate((A_IK0[:, 0], A_IK0[:, 1], A_IK0[:, 2]))
        q0 = np.concatenate((r_OS0, p0))
        A_IK_dot0 = A_IK0 @ ax2skew(K_omega0)
        u0 = np.concatenate((v_S0, A_IK_dot0[:, 0], A_IK_dot0[:, 1], A_IK_dot0[:, 2]))
        I11 = K_theta_S[0,0]
        I22 = K_theta_S[1,1]
        I33 = K_theta_S[2,2]
        # Binet inertia tensor
        i11 = 0.5 * (I22 + I33 - I11)
        i22 = 0.5 * (I11 + I33 - I22)
        i33 = 0.5 * (I11 + I22 - I33)
        B_rho0 = np.zeros(3)
        C_rho0 = np.diag(np.array([i11, i22, i33]))
        RB = Rigid_body_director(m, B_rho0, C_rho0, q0=q0, u0=u0)
        model.add(RB)
        model.add(Force(np.array([0, -m*g, 0]), RB))
        model.add(Spherical_joint(frame, RB, r_OP(0)))
    elif rigid_body == 'Director_connected':
        p0 = np.concatenate((A_IK0[:, 0], A_IK0[:, 1], A_IK0[:, 2]))
        q10 = np.concatenate((r_OS10, p0))
        q20 = np.concatenate((r_OS20, p0))
        A_IK_dot0 = A_IK0 @ ax2skew(K_omega0)
        u10 = np.concatenate((v_S10, A_IK_dot0[:, 0], A_IK_dot0[:, 1], A_IK_dot0[:, 2]))
        u20 = np.concatenate((v_S20, A_IK_dot0[:, 0], A_IK_dot0[:, 1], A_IK_dot0[:, 2]))
        I11 = K_theta_S1[0,0]
        I22 = K_theta_S1[1,1]
        I33 = K_theta_S1[2,2]
        # Binet inertia tensor
        i11 = 0.5 * (I22 + I33 - I11)
        i22 = 0.5 * (I11 + I33 - I22)
        i33 = 0.5 * (I11 + I22 - I33)
        B_rho0 = np.zeros(3)
        C_rho0 = np.diag(np.array([i11, i22, i33]))
        RB1 = Rigid_body_director(m / 2, B_rho0, C_rho0, q0=q10, u0=u10)
        model.add(RB1)
        RB2 = Rigid_body_director(m / 2, B_rho0, C_rho0, q0=q20, u0=u20)
        model.add(RB2)
        gravity1 = Force(np.array([0, -m / 2 * g, 0]), RB1)
        model.add(gravity1)
        gravity2 = Force(np.array([0, -m / 2 * g, 0]), RB2)
        model.add(gravity2)
        model.add(Revolute_joint(frame, RB1, r_OP(0), np.eye(3)))
        model.add(Rigid_connection(RB1, RB2, r_OS0))
    elif rigid_body == 'Rigid_body2D':
        q0 = np.append(r_OS0[:2], phi0)
        u0 = np.append(v_S0[:2], phi_dot0)
        RB = Rigid_body2D(m, theta, q0=q0, u0=u0)
        model.add(RB)
        model.add(Force(np.array([0, -m * g, 0]), RB))
        model.add(Spherical_joint2D(frame, RB, r_OP(0)))
    elif rigid_body == 'Rigid_body2D_connected':
        q10 = np.append(r_OS10[:2], phi0)
        q20 = np.append(r_OS20[:2], phi0)
        u10 = np.append(v_S10[:2], phi_dot0)
        u20 = np.append(v_S20[:2], phi_dot0)
        RB1 = Rigid_body2D(m / 2, theta1, q0=q10, u0=u10)
        RB2 = Rigid_body2D(m / 2, theta2, q0=q20, u0=u20)
        model.add(RB1)
        model.add(RB2)
        gravity1 = Force(np.array([0, -m/2*g, 0]), RB1)
        model.add(gravity1)
        gravity2 = Force(np.array([0, -m/2*g, 0]), RB2)
        model.add(gravity2)
        model.add(Spherical_joint2D(frame, RB1, r_OP(0)))
        model.add(Rigid_connection2D(RB1, RB2, r_OS0))
        
    model.assemble()

    t0 = 0
    t1 = 2
    dt = 5e-3

    # solver = Euler_backward(model, t1, dt, numerical_jacobian=False, debug=False)
    # solver = Moreau_sym(model, t1, dt)
    # solver = Moreau(model, t1, dt)
    # solver = Generalized_alpha_1(model, t1, variable_dt=True, rtol=0, atol=2.5e-2, rho_inf=1, numerical_jacobian=False, debug=False)
    # solver = Generalized_alpha_2(model, t1, dt, rho_inf=1, numerical_jacobian=False, debug=False)
    solver = Generalized_alpha_3(model, t1, dt, rho_inf=1, numerical_jacobian=False, debug=False)

    # solver = Scipy_ivp(model, t1, dt)
    sol = solver.solve()
    t = sol.t
    q = sol.q
    u = sol.u

    fig, ax = plt.subplots(2, 1)

    x_ = []
    y_ = []   
    
    if rigid_body == 'Quaternion_connected':
        r_OS = np.zeros((3, len(q[:, 0])))
        r_OS1 = np.zeros((3, len(q[:, 0])))
        for i, ti in enumerate(t):
            r_OS = q[i, :3] - (RB1.A_IK(ti, q[i, :7]) @ K_r_SP1)
            x_.append(r_OS[0])
            y_.append(r_OS[1])
    elif rigid_body == 'Director_connected':
        for i, ti in enumerate(t):
            r_OS = q[i, :3] - (RB1.A_IK(ti, q[i, :12]) @ K_r_SP1)
            x_.append(r_OS[0])
            y_.append(r_OS[1])
    elif rigid_body == 'Rigid_body2D_connected':
        r_OS = np.zeros((2, len(q[:, 0])))
        for i, ti in enumerate(t):
            r_OS = q[i, :2] - (A_IK_basic_z(q[i, 2]) @ K_r_SP1)[:2]
            x_.append(r_OS[0])
            y_.append(r_OS[1])
    else:
        x_ = q[:,0]
        y_ = q[:,1]

    ax[0].plot(t, x_, '--gx')
    ax[1].plot(t, y_, '--gx')

    # reference solution
    def eqm(t,x):
        dx = np.zeros(2)
        dx[0] = x[1]
        dx[1] = -0.5 * m * L * (e_tt(t) * cos(x[0]) + g * sin(x[0])) / theta_O
        return dx

    dt = 0.001

    x0 = np.array([phi0, phi_dot0])
    ref = solve_ivp(eqm,[t0,t1],x0, method='RK45', t_eval=np.arange(t0,t1 + dt,dt), rtol=1e-8, atol=1e-12) # MATLAB ode45
    x = ref.y
    t = ref.t

    # plot reference solution
    ax[0].plot(t, e(t) + L/2 * np.sin(x[0]), '-r')
    ax[1].plot(t, -L/2 * np.cos(x[0]), '-r')

    plt.show()