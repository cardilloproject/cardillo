import numpy as np
from scipy.integrate import solve_ivp
from math import cos, sin, sqrt, tan
import matplotlib.pyplot as plt

from cardillo.math.algebra import A_IK_basic_z, axis_angle2quat, cross3, ax2skew
from cardillo.model import Model
from cardillo.model.rigid_body import Rigid_body_euler, Rigid_body_quaternion, Rigid_body_director
from cardillo.model.frame import Frame
from cardillo.model.bilateral_constraints import Revolute_joint, Spherical_joint
from cardillo.model.force import Force
from cardillo.solver import Moreau, Moreau_sym, Euler_backward, Generalized_alpha_1, Scipy_ivp


if __name__ == "__main__":

    # rigid_body = 'Euler'
    rigid_body = 'Quaternion'
    # rigid_body = 'Director'
    #%% parameters
    m = 1
    L = 2
    theta = 1 / 12 * m * (L ** 2)
    theta_O = theta + m * (L ** 2) / 4
    g = 9.81
    omega = 10
    A = L / 10

    # e = lambda t: A * np.cos(omega * t)
    # e_t = lambda t: -A * omega * np.sin(omega * t)
    # e_tt = lambda t: -A * omega * omega * np.cos(omega * t)

    # e = lambda t: A * np.sin(omega * t)
    # e_t = lambda t: A * omega * np.cos(omega * t)
    # e_tt = lambda t: -A * omega * omega * np.sin(omega * t)

    e = lambda t: A * t
    e_t = lambda t: A 
    e_tt = lambda t: 0

    r_OP = lambda t: np.array([e(t), 0, 0]) 
    v_P = lambda t: np.array([e_t(t), 0, 0]) 
    a_P = lambda t: np.array([e_tt(t), 0, 0]) 

    K_r_SP = np.array([0, L/2, 0]) 
    # phi0 = 0
    phi0 = 0.5
    phi_dot0 = 0
    K_omega0 = np.array([0, 0, phi_dot0])
    A_IK0 = A_IK_basic_z(phi0)
    r_OS0 = r_OP(0) - A_IK0 @ K_r_SP
    v_S0 = v_P(0) + A_IK0 @ ( cross3(K_omega0, K_r_SP) )

    model = Model()

    if rigid_body == 'Euler':
        p0 = np.array([phi0, 0, 0])
    elif rigid_body == 'Quaternion':
        p0 = axis_angle2quat(np.array([0, 0, 1]), phi0)
    elif rigid_body == 'Director':
        p0 = np.concatenate((A_IK0[:, 0], A_IK0[:, 1], A_IK0[:, 2]))

    q0 = np.concatenate((r_OS0, p0))

    if rigid_body == 'Euler' or rigid_body == 'Quaternion':
        u0 = np.concatenate((v_S0, K_omega0))
    elif rigid_body == 'Director':
        A_IK_dot0 = A_IK0 @ ax2skew(K_omega0)
        u0 = np.concatenate((v_S0, A_IK_dot0[:, 0], A_IK_dot0[:, 1], A_IK_dot0[:, 2]))

    K_theta_S = theta * np.eye(3)
    if rigid_body == 'Euler':
        RB = Rigid_body_euler(m, K_theta_S, q0=q0, u0=u0)
    elif rigid_body == 'Quaternion':
        RB = Rigid_body_quaternion(m, K_theta_S, q0=q0, u0=u0)
    elif rigid_body == 'Director':
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

    gravity = Force(np.array([0, -m*g, 0]), RB)
    model.add(gravity)

    frame = Frame(r_OP=r_OP, r_OP_t=v_P, r_OP_tt=a_P)
    model.add(frame)

    # joint = Revolute_joint(frame, RB, r_OP(0), np.eye(3))
    joint = Spherical_joint(frame, RB, r_OP(0))
    model.add(joint)

    model.assemble()

    t0 = 0
    t1 = 2
    dt = 5e-3

    # solver = Euler_backward(model, t1, dt, numerical_jacobian=False, debug=False)
    # solver = Moreau_sym(model, t1, dt, numerical_jacobian=False, debug=False)
    # solver = Moreau(model, t1, dt)
    # solver = Generalized_alpha_1(model, t1, dt, rho_inf=1, numerical_jacobian=False, debug=False)
    solver = Scipy_ivp(model, t1, dt)
    sol = solver.solve()
    t = sol.t
    q = sol.q
    u = sol.u

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(t, q[:,0], '-x')
    ax[1].plot(t, q[:,1], '-x')

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
    ax[0].plot(t, e(t) + L/2 * np.sin(x[0]))
    ax[1].plot(t, -L/2 * np.cos(x[0]))

    plt.show()