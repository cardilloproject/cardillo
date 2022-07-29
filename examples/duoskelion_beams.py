import numpy as np
import pathlib
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from cardillo.utility.post_processing_vtk import post_processing
# from cardillo.model.classical_beams.spatial.Timoshenko_beam_director import post_processing
from cardillo.math.algebra import A_IK_basic_z, A_IK_basic_x, A_IK_basic_y
from cardillo.solver.solution import load_solution
from cardillo.model.force import Force
from cardillo.model.moment import K_Moment
from cardillo.discretization import uniform_knot_vector
from cardillo.model.line_force.line_force import Line_force
from cardillo.solver import Newton, Euler_backward, Euler_forward, Generalized_alpha_1, Generalized_alpha_2, Scipy_ivp, Generalized_alpha_4_index3, Generalized_alpha_4_singular_index3
from cardillo.model import Model
from cardillo.model.classical_beams.spatial import Hooke_quadratic, Hooke
from cardillo.model.classical_beams.spatial import Timoshenko_director_dirac
from cardillo.model.classical_beams.spatial import Timoshenko_director_integral, Euler_Bernoulli_director_integral, Inextensible_Euler_Bernoulli_director_integral
from cardillo.model.classical_beams.spatial.director import straight_configuration
from cardillo.model.frame import Frame
from cardillo.model.rigid_body import Rigid_body_euler, Rigid_body_director
from cardillo.model.bilateral_constraints.implicit import (
    Rigid_connection,
    Revolute_joint,
    Linear_guidance_xyz,
    Linear_guidance_x,
    Spherical_joint,
    Rod,
    Single_position_y,
    # Saddle_joint,
    # Single_position_all_angles
)
from cardillo.model.scalar_force_interactions.force_laws import Linear_spring
from cardillo.model.scalar_force_interactions import Translational_f_pot
from cardillo.model.scalar_force_interactions import add_rotational_forcelaw as Rotational 
from collections import defaultdict


A_IK0 = np.eye(3)
A_IK1 = A_IK_basic_z(np.pi / 4)
A_IK2 = A_IK_basic_z(3 * np.pi / 4)
A_IK3 = A_IK_basic_x(3 * np.pi / 4) @ A_IK_basic_y(np.pi / 2)
# A_IK3 = A_IK_basic_z(np.pi / 4) @ A_IK_basic_y(np.pi / 2)

A_IK4 = A_IK_basic_x(3*np.pi / 4) @ A_IK_basic_y(np.pi /
                                                 2) @ A_IK_basic_z(np.pi/2)

A_IK_list = [A_IK1, A_IK2, A_IK3, A_IK4]

A_IK_y = A_IK_basic_x(np.pi/2) @ A_IK_basic_y(np.pi/2)

A_IK_x = np.eye(3)

A_IK_z = A_IK_basic_x(-np.pi/2) @ A_IK_basic_z(-np.pi/2)

A_IK_p = A_IK_basic_x(-np.pi/2)

ex = np.array([1,0,0])
ey = np.array([0,1,0])
ez = np.array([0,0,1])

def save_solution(sol, filename):
    import pickle
    with open(filename, mode='wb') as f:
        pickle.dump(sol, f)


class Beam():
    def __init__(self, p, q, nEl, L, r_OP, k, A_IK, material):
        self.r_OP = r_OP
        self.k = k
        # self.b_type = b_type
        self.A_IK = A_IK
        Q = straight_configuration(
            p, q, nEl*k, L*k, greville_abscissae=greville, r_OP=r_OP, A_IK=self.A_IK, basis=basis)
        self.q0 = Q.copy()
        self.beam = Timoshenko_director_integral(material, A_rho0,
                                                 B_rho0, C_rho0, p, q, nQP,
                                                 nEl*k, Q=Q, q0=self.q0, basis=basis)


class Cross():
    def __init__(self, r_OB, l1, l2, rigid=False, pivot=True):

        if rigid:
            NotImplementedError()
        else:
            self.beams = np.empty(6, dtype=object)
            self.joints = np.empty(5, dtype=object)
            self.rigid = np.empty(4, dtype=object)
            self.force_laws = np.empty(4, dtype=object)

            self.beams[0] = Beam(p, q, nel, l1, r_OB - 0.5*l1*ex, 1, A_IK_x, material_model)
            self.beams[1] = Beam(p, q, nel, l1, r_OB - 0.5*l1*ey, 1, A_IK_y, material_model)

            # self.joints[0] = Revolute_joint(self.beams[0].beam, self.beams[1].beam, r_OB, A_IK_x, frame_ID1=(.5,), frame_ID2=(.5,))
            self.joints[0] = Rigid_connection(self.beams[0].beam, self.beams[1].beam, r_OB, frame_ID1=(.5,), frame_ID2=(.5,))

            # beam top
            self.beams[2] = Beam(p, q, nel, l2, r_OB + 0.5*l1*ex, 1, - A_IK_y, material_model)
            self.r_OP_t = r_OB + 0.5 * l1 * ex - l2 * ey
            self.beams[3] = Beam(p, q, nel, l2, r_OB - 0.5*l1*ex, 1, A_IK_y, material_model)
            self.r_OP_b = r_OB - 0.5 * l1 * ex + l2 * ey

            self.joints[1] = Rigid_connection(self.beams[0].beam, self.beams[3].beam, r_OB - 0.5 * l1 * ex)
            self.joints[2] = Rigid_connection(self.beams[0].beam, self.beams[2].beam, r_OB + 0.5 * l1 * ex, frame_ID1=(1,))

            self.beams[4] = Beam(p, q, nel, l2, r_OB + 0.5*l1*ey, 1, A_IK_x, material_model)
            self.r_OP_r = r_OB + 0.5 * l1 * ey + l2 * ex
            self.beams[5] = Beam(p, q, nel, l2, r_OB - 0.5*l1*ey, 1, - A_IK_x, material_model)
            self.r_OP_l = r_OB - 0.5 * l1 * ey - l2 * ex

            self.joints[3] = Rigid_connection(self.beams[1].beam, self.beams[5].beam, r_OB - 0.5 * l1 * ey)
            self.joints[4] = Rigid_connection(self.beams[1].beam, self.beams[4].beam, r_OB + 0.5 * l1 * ey, frame_ID1=(1,))

            # rigid bodies on end nodes
            q0_l = np.concatenate((self.r_OP_l, np.zeros(3)))
            q0_r = np.concatenate((self.r_OP_r, np.zeros(3)))
            q0_t = np.concatenate((self.r_OP_t, np.zeros(3)))
            q0_b = np.concatenate((self.r_OP_b, np.zeros(3)))
            spring = Linear_spring(kt)
            # self.rigid[0] = Rigid_body_euler(1, np.eye(3), q0=q0_t)
            # # self.joints[5] = Revolute_joint(self.beams[2], self.rigid[0], self.r_OP_t, A_IK_x)
            # self.force_laws[0] = Rotational(spring, Revolute_joint)(self.beams[3].beam, self.rigid[0], self.r_OP_t, A_IK_x, frame_ID1=(1,))

            # self.rigid[1] = Rigid_body_euler(1, np.eye(3), q0=q0_b)
            # # self.joints[6] = Revolute_joint(self.beams[3], self.rigid[1], self.r_OP_b, A_IK_x)
            # self.force_laws[1] = Rotational(spring, Revolute_joint)(self.beams[2].beam, self.rigid[1], self.r_OP_b, A_IK_x, frame_ID1=(1,))

            # self.rigid[2] = Rigid_body_euler(1, np.eye(3), q0=q0_r)
            # # self.joints[7] = Revolute_joint(self.beams[4], self.rigid[2], self.r_OP_r, A_IK_x)
            # self.force_laws[2] = Rotational(spring, Revolute_joint)(self.beams[4].beam, self.rigid[2], self.r_OP_r, A_IK_x, frame_ID1=(1,))

            # self.rigid[3] = Rigid_body_euler(1, np.eye(3), q0=q0_l)
            # # self.joints[8] = Revolute_joint(self.beams[5], self.rigid[3], self.r_OP_l, A_IK_x)
            # self.force_laws[3] = Rotational(spring, Revolute_joint)(self.beams[5].beam, self.rigid[3], self.r_OP_l, A_IK_x, frame_ID1=(1,))
            
class Grid():
    def __init__(self, l1, l2, nEl, spring=False):
        nEl_x, nEl_y = nEl
        grid_dict = {}
        self.beams = []
        self.joints = []
        self.rigid = []
        self.force_laws = []
        lin_spring = Linear_spring(ks)
        for nx in range(nEl_x):
            for ny in range(nEl_y):
                r_OB = 2*l1*nx*ex + 2*l1*ny*ey
                cross = Cross(r_OB, l1, l2)
                grid_dict.update({(nx, ny) : {'cross': cross}})
                for beam in cross.beams:
                    self.beams.append(beam.beam)
                for rigid in cross.rigid:
                    self.rigid.append(rigid)
                for law in cross.force_laws:
                    self.force_laws.append(law)
                for joint in cross.joints:
                    self.joints.append(joint)
                    
                # rigid connection to frame left
                if nx == 0:
                    bc = Rigid_connection(frame_left, cross.beams[5].beam, cross.r_OP_l, frame_ID2=(1,))
                    self.joints.append(bc)
                # rigid connection to frame right and left unit cell
                elif nx == nEl_x - 1:
                    bc = Rigid_connection(frame_right, cross.beams[4].beam, cross.r_OP_r, frame_ID2=(1,))
                    self.joints.append(bc)
                    
                    # bc = Translational_f_pot(spring, cross.rigid[3], grid_dict[(nx-1,ny)]['cross'].rigid[2])
                    # self.force_laws.append(bc)
                    if not spring:
                        beam = Beam(p, q, nel, l1, cross.r_OP_l, 1, A_IK_y, material=soft_beam)
                        # bc1 = Rigid_connection(beam.beam, cross.rigid[3], cross.r_OP_l)
                        # bc2 = Rigid_connection(beam.beam, grid_dict[(nx-1, ny)]['cross'].rigid[2], grid_dict[(nx-1,ny)]['cross'].r_OP_r, frame_ID1=(1,))
                        bc1 = Rotational(lin_spring, Revolute_joint)(beam.beam, cross.beams[5].beam, cross.r_OP_l, A_IK_x, frame_ID2=(1,))
                        bc2 = Rotational(lin_spring, Revolute_joint)(beam.beam, grid_dict[(nx-1,ny)]['cross'].beams[4].beam, grid_dict[(nx-1,ny)]['cross'].r_OP_r, A_IK_x, frame_ID1=(1,), frame_ID2=(1,))
                        self.beams.append(beam.beam)
                        self.joints.append(bc1)
                        self.joints.append(bc2)

                else:
                    # spring = Linear_spring(ks, l1)
                    # bc = Translational_f_pot(spring, cross.rigid[3], grid_dict[(nx-1,ny)]['cross'].rigid[2])
                    # self.force_laws.append(bc)
                    if not spring:
                        beam = Beam(p, q, nel, l1, cross.r_OP_l, 1, A_IK_y, material=soft_beam)
                        # bc1 = Rigid_connection(beam.beam, cross.rigid[3], cross.r_OP_l)
                        # bc2 = Rigid_connection(beam.beam, grid_dict[(nx-1,ny)]['cross'].rigid[2], grid_dict[(nx-1,ny)]['cross'].r_OP_r, frame_ID1=(1,))
                        bc1 = Rotational(lin_spring, Revolute_joint)(beam.beam, cross.beams[5].beam, cross.r_OP_l, A_IK_x, frame_ID2=(1,))
                        bc2 = Rotational(lin_spring, Revolute_joint)(beam.beam, grid_dict[(nx-1,ny)]['cross'].beams[4].beam, grid_dict[(nx-1,ny)]['cross'].r_OP_r, A_IK_x, frame_ID1=(1,), frame_ID2=(1,))
                        self.beams.append(beam.beam)
                        self.joints.append(bc1)
                        self.joints.append(bc2)
#  create unit cells
ncells_x = 3
ncells_y = 1

ncells = (ncells_x, ncells_y)

u_l = 1. #1e-3
u_Pa = 1. #1e9
l = 70.0 * 3 * u_l  # length in mm
# Beam dimensions and parameter
L = l/np.sqrt(2)/ncells_x*2.  # beam length between pivots
r = 0.45 * u_l  #  pivot radius in mm
E_Y = 50. * u_Pa  # Young's Modulus in GPa
a = L/5 # 1. * u_l  # Beam cross section length 1 in mm
b = L/5 # 1. * u_l  # Beam cross section length 2 in mm
G = E_Y / (2 + 0.8)
I_1 = 2.25*(a/2)**4  # torsional moment for square cross-section
I_2 = a**3*b/12  # Bending moments
I_3 = a*b**3/12
# I_i = np.pi * r**4 * np.array([1/2, 1/4, 1/4])
I_P = I_1 + I_2
A = a*b
Ei = np.array([E_Y, G, G]) * A
Fi = np.array([G*I_1, E_Y*I_2, E_Y*I_3])

# force law parameters
kt = 100.
ks = 100.

# Beam parameters
rho = 2.7e-1
A_rho0 = rho * A
B_rho0 = np.zeros(3)
C_rho0 = np.array([[0, 0, 0], [0, I_3, 0], [0, 0, I_2]]) * rho

material_model = Hooke_quadratic(Ei, Fi)
soft_beam = Hooke_quadratic(Ei*0.5,Fi)

# pivot length
piv_h = 1.5 * u_l * 0# mm
rigid_pivot = True
cross = False

# discretization
basis = 'B-spline'
# basis = 'lagrange'
greville = False
p = 3
q = 3
nQP = p + 1
# nQP = int(np.ceil((p**2 + 1) / 2)) + 1
nel = 2

# dynamic solver?
dynamic = False

save_sol = True
load_sol = True

model = Model()

# boundary condittions
bc_dir = 'z'
r_OB_top0 = l / 3 * np.array([0.5, 0.5, 3.0])

# tests = ['tension', str(l),"fixed_boundary","I_1"]
tests = ['torsion', str(l),"fixed_boundary"]
if 'tension' in tests:
    def r_OP_top(t): return r_OB_top0 + t * 50.0 * np.array([0.0, 0.0, 1.0]) * u_l
    A_IK_top = np.eye(3)
    frame_top2 = Frame(r_OP_top)
    model.add(frame_top2)

if 'torsion' in tests:
    def r_OP_top(t): return r_OB_top0  # + t * 30.0 * np.array([0.0, 0.0, 1.0])
    def A_IK_top(t): return A_IK_basic_z(t * np.pi/4)
    frame_top2 = Frame(r_OP_top, A_IK=A_IK_top)
    model.add(frame_top2)


def r_OP_middle(t): return r_OB_top0 * .5 + np.sqrt(2) * \
    L/2 * np.array([0.0, 0.0, 1.0]) * t  # * 0.5 * 5e-1

# frames
l1 = 70.
l2 = 70.
frame_left = Frame(np.zeros(3))
r_OPf = lambda t: (ncells_x * 2 * l2 + l2/2 + 20. * t)*ex
frame_right = Frame(r_OPf)




if 'force' in tests:
    def force(t): return 0.05 * t * np.array([0.,0.,1.])
    frame_top2 = Rigid_body_euler(1, np.eye(3))
    frame_top = Frame(r_OB_top0)
    z_only = Linear_guidance_x(frame_top, frame_top2, r_OB_top0, A_IK_z)
    # rigid_frame_body = Rigid_connection(frame_top, frame_top2, r_OB_top0)
    T_Force = Force(force, frame_top2)
    model.add(frame_top2)
    model.add(T_Force)
    model.add(z_only)
    # model.add(rigid_frame_body)

# make
grid = Grid(l1, l2, (ncells))


# assemble
beams_all = []

model.add(frame_left)
model.add(frame_right)
# model.add(frame_middle)
# model.add(frame_top)
for beam in grid.beams:
    model.add(beam)
    beams_all.append(beam)
# for beam in grid.pivot_beams:
#     model.add(beam)
#     beams_all.append(beam)
for joint in grid.joints:
    model.add(joint)
# for rigid in grid.rigid:
#     model.add(rigid)
# for law in grid.force_laws:
#     model.add(law)
# for bc in grid.bc:
#     model.add(bc)
model.assemble()

# set initial accelarations
if dynamic:
    from scipy.sparse.linalg import spsolve
    uDOF = np.arange(model.nu)
    uDOF_algebraic = []
    uDOF_dynamic = []
    a0 = np.zeros(model.nu)
    for beam in beams_all:
        rDOF = (beam.nEl + beam.polynomial_degree_r) * 3
        dDOF = (beam.nEl + beam.polynomial_degree_di) * 3
        uDOF_algebraic.extend(beam.uDOF[rDOF:rDOF+dDOF])  # whole beam dynamic
        # uDOF_algebraic.extend(beam.uDOF[rDOF:])  # exclude director dynamics
        # beam as static force element (no beam dynamics)
        # uDOF_algebraic.extend(beam.uDOF)
        # uDOF_algebraic = beam.uDOF[tmp:4*tmp]
        # uDOF_dynamic = np.setdiff1d(beam.uDOF, uDOF_algebraic)
        # M0 = model.M(model.t0, model.q0).tocsr()[uDOF_dynamic[:, None], uDOF_dynamic]
        # rhs0 = model.h(model.t0, model.q0, model.u0) + model.W_g(model.t0,
        #                                                         model.q0) @ model.la_g0 + model.W_gamma(model.t0, model.q0) @ model.la_gamma0
        # print(2)
        # a0[uDOF_dynamic] = spsolve(M0, rhs0[uDOF_dynamic])

    uDOF_algebraic = np.array(uDOF_algebraic)
    uDOF_dynamic = np.setdiff1d(uDOF, uDOF_algebraic)
    # M0 = model.M(model.t0, model.q0).tocsr()[uDOF_dynamic[:, None], uDOF_dynamic]
    # rhs0 = model.h(model.t0, model.q0, model.u0) + model.W_g(model.t0,
    #                                                             model.q0) @ model.la_g0 + model.W_gamma(model.t0, model.q0) @ model.la_gamma0
    # from scipy.sparse.linalg import spsolve
    # a0[uDOF_dynamic] = spsolve(M0, rhs0[uDOF_dynamic])

    # solver
    t0 = 0
    t1 = 0.1
    dt = 0.005
    max_iter = 10
    tol = 1e-5
    rho_inf = 0.0
    # a0 = np.zeros(model.nu)
    # solver = Newton(model, n_load_steps=10, max_iter=10, tol=1e-5)
    # solver = Euler_forward(model, t1, dt)
    # solver = Euler_backward(model, t1, dt)
    # solver = Scipy_ivp(model, t1, dt)
    # solver = Euler_backward(model, t1, dt)
    # solver = Generalized_alpha_4_index3(
    #  model, t1, dt, newton_max_iter=max_iter, newton_tol=tol, a0=a0).solve()
    solver = Generalized_alpha_4_singular_index3(
        model, t1, dt, uDOF_algebraic=uDOF_algebraic, rho_inf=rho_inf, newton_max_iter=max_iter, newton_tol=tol).solve()

    sol = solver
    t = sol.t
    q = sol.q

else:
    solver = Newton(model, n_load_steps=10, max_iter=20, tol=1e-6)

    #sol = solver.solve()
    # t = sol.t
    # q = sol.q


file_name = pathlib.Path(__file__).stem
file_path = pathlib.Path(__file__).parent / 'results' / str(
    f"{file_name}_" + 'x'.join([str(v) for v in ncells]) + '_'.join(tests)) / file_name
file_path.parent.mkdir(parents=True, exist_ok=True)
export_path = file_path.parent / 'sol'

if save_sol:
    # import cProfile, pstats
    # pr = cProfile.Profile()
    # pr.enable()
    sol = solver.solve()
    t = sol.t
    q = sol.q

    save_solution(sol, str(export_path))
elif load_sol:
    import pickle
    sol = pickle.load(open(str(export_path), 'rb'))


# vtk export
post_processing(
    beams_all, sol.t, sol.q,
    file_path, binary=True)
