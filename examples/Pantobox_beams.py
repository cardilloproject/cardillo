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
from cardillo.solver import (
    Newton,
    Euler_backward,
    Euler_forward,
    Generalized_alpha_1,
    Generalized_alpha_2,
    Scipy_ivp,
    Generalized_alpha_4_index3,
    Generalized_alpha_4_singular_index3,
)
from cardillo.model import Model
from cardillo.model.classical_beams.spatial import Hooke_quadratic, Hooke
from cardillo.model.classical_beams.spatial import Timoshenko_director_dirac
from cardillo.model.classical_beams.spatial import (
    Timoshenko_director_integral,
    Euler_Bernoulli_director_integral,
    Inextensible_Euler_Bernoulli_director_integral,
)
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
from collections import defaultdict


A_IK0 = np.eye(3)
A_IK1 = A_IK_basic_z(np.pi / 4)
A_IK2 = A_IK_basic_z(3 * np.pi / 4)
A_IK3 = A_IK_basic_x(3 * np.pi / 4) @ A_IK_basic_y(np.pi / 2)
# A_IK3 = A_IK_basic_z(np.pi / 4) @ A_IK_basic_y(np.pi / 2)

A_IK4 = A_IK_basic_x(3 * np.pi / 4) @ A_IK_basic_y(np.pi / 2) @ A_IK_basic_z(np.pi / 2)

A_IK_list = [A_IK1, A_IK2, A_IK3, A_IK4]

A_IK_y = A_IK_basic_x(np.pi / 2) @ A_IK_basic_y(np.pi / 2)

A_IK_x = np.eye(3)

A_IK_z = A_IK_basic_x(-np.pi / 2) @ A_IK_basic_z(-np.pi / 2)

A_IK_p = A_IK_basic_x(-np.pi / 2)


def save_solution(sol, filename):
    import pickle

    with open(filename, mode="wb") as f:
        pickle.dump(sol, f)


class Panto_beam(Timoshenko_director_integral):
    def __init__(self, p, q, nEl, L, r_OP, k, b_type):
        self.r_OP0 = r_OP
        self.k = k
        self.b_type = b_type
        self.A_IK0 = A_IK_list[self.b_type - 1]
        Q = straight_configuration(
            p,
            q,
            nEl * k,
            L * k,
            greville_abscissae=greville,
            r_OP=r_OP,
            A_IK=self.A_IK0,
            basis=basis,
        )
        self.q0 = Q.copy()
        super().__init__(
            material_model,
            A_rho0,
            B_rho0,
            C_rho0,
            p,
            q,
            nQP,
            nEl * k,
            Q=Q,
            q0=self.q0,
            basis=basis,
        )

        # self.beam = Timoshenko_director_integral(material_model, A_rho0,
        #                                          B_rho0, C_rho0, p, q, nQP,
        #                                          nEl*k, Q=Q, q0=self.q0, basis=basis)


class Pivot:
    def __init__(self, r_OB, rigid=True, p_type="rigid", nEl_p=2, floppy2=True):
        self.r_OB = r_OB
        self.r_OB1 = r_OB + A_IK_z[:, 0] * piv_h / 2
        self.r_OB2 = r_OB - A_IK_z[:, 0] * piv_h / 2

        self.r_OB3 = r_OB - A_IK_x[:, 0] * piv_h / 2
        self.r_OB4 = r_OB + A_IK_x[:, 0] * piv_h / 2

        self.is_rigid = rigid

        # rigid body pivot
        if rigid:
            self.V = 2 * np.pi * r**2 * piv_h
            self.m = rho * self.V
            # single rigid body with moment of inertia of a cross
            if cross:
                q0 = np.concatenate((r_OB, np.zeros(3)))
                self.theta = 1 / 12 * self.m * piv_h**2 * np.diag([2, 1, 1])
                self.body = [Rigid_body_euler(self.m, self.theta, axis="yzx", q0=q0)]
                self.body = np.repeat(self.body, 5)
                self.f_ID = np.tile(np.zeros(3), 4)
                self.rigid = []
            # five rigid bodies
            else:
                self.body = np.empty(6, object)
                q0 = np.concatenate((r_OB, np.zeros(3)))

                q01 = np.concatenate((self.r_OB1, np.zeros(3)))
                q02 = np.concatenate((self.r_OB2, np.zeros(3)))
                q03 = np.concatenate((self.r_OB3, np.zeros(3)))
                q04 = np.concatenate((self.r_OB4, np.zeros(3)))

                In = 2 / 5 * self.m / 5 * r**2
                self.theta = np.diag([In, In, In])
                self.body[0] = Rigid_body_euler(self.m / 5, self.theta, q0=q0)
                self.body[5] = Rigid_body_euler(self.m / 5, self.theta, q0=q0)

                self.body[1] = Rigid_body_euler(
                    self.m / 5, self.theta, axis="zxy", q0=q01
                )
                self.body[2] = Rigid_body_euler(
                    self.m / 5, self.theta, axis="zxy", q0=q02
                )
                self.body[3] = Rigid_body_euler(
                    self.m / 5, self.theta, axis="xyz", q0=q03
                )
                self.body[4] = Rigid_body_euler(
                    self.m / 5, self.theta, axis="xyz", q0=q04
                )

                self.f_ID = [np.zeros(3) for i in range(4)]

                # rotational joint between pivots for second floppy mode
                if floppy2:
                    self.rigid1 = Rigid_connection(
                        self.body[0], self.body[1], self.r_OB1
                    )
                    self.rigid2 = Rigid_connection(
                        self.body[0], self.body[2], self.r_OB2
                    )
                    self.rigid3 = Rigid_connection(
                        self.body[5], self.body[3], self.r_OB3
                    )
                    self.rigid4 = Rigid_connection(
                        self.body[5], self.body[4], self.r_OB4
                    )
                    self.rigid5 = Revolute_joint(
                        self.body[0], self.body[5], r_OB, A_IK_z
                    )
                    # self.rigid5 = Spherical_joint(self.body[0], self.body[5], r_OB)
                else:
                    self.rigid1 = Rigid_connection(
                        self.body[0], self.body[1], self.r_OB1
                    )
                    self.rigid2 = Rigid_connection(
                        self.body[0], self.body[2], self.r_OB2
                    )
                    self.rigid3 = Rigid_connection(
                        self.body[5], self.body[3], self.r_OB3
                    )
                    self.rigid4 = Rigid_connection(
                        self.body[5], self.body[4], self.r_OB4
                    )
                    self.rigid5 = Rigid_connection(self.body[0], self.body[5], r_OB)

                self.rigid = [
                    self.rigid1,
                    self.rigid2,
                    self.rigid3,
                    self.rigid4,
                    self.rigid5,
                ]
        # beams as pivots
        else:
            Q1 = straight_configuration(
                p, q, nEl_p, piv_h, True, r_OP=self.r_OB1, A_IK=-A_IK_z
            )
            Q2 = straight_configuration(
                p, q, nEl_p, piv_h, True, r_OP=self.r_OB3, A_IK=A_IK_x
            )
            self.q01 = Q1.copy()
            self.q02 = Q2.copy()
            self.beam1 = Timoshenko_director_integral(
                material_model,
                A_rho0,
                B_rho0,
                C_rho0,
                p,
                q,
                nQP,
                nEl_p,
                Q=Q1,
                q0=self.q01,
                basis=basis,
            )
            self.beam2 = Timoshenko_director_integral(
                material_model,
                A_rho0,
                B_rho0,
                C_rho0,
                p,
                q,
                nQP,
                nEl_p,
                Q=Q2,
                q0=self.q02,
                basis=basis,
            )
            self.beams = [self.beam1, self.beam2]
            self.body = [None, self.beam1, self.beam1, self.beam2, self.beam2]
            self.f_ID = [(0,), (1,), (0,), (1,)]
            # joint between pivot beams
            if p_type == "rigid":
                self.rigid = [
                    Rigid_connection(
                        self.beam1, self.beam2, r_OB, frame_ID1=(0.5,), frame_ID2=(0.5,)
                    )
                ]
            # revolut joint for second floppy mode
            elif p_type == "revolute":
                self.rigid = [
                    Revolute_joint(
                        self.beam1,
                        self.beam2,
                        r_OB,
                        A_IK_p,
                        frame_ID1=(0.5,),
                        frame_ID2=(0.5,),
                    )
                ]


class Panto_grid:
    def __init__(self, n_xyz):

        self.n_xyz = n_xyz
        self.ncells_x, self.ncells_y, self.ncells_z = self.n_xyz
        # frames
        # frame_ID1 = (0,)
        # frame_ID2 = (1,)

        self.beams = []
        self.pivot_beams = []
        self.cells = []
        self.joints = []
        self.cells_dic = {}
        self.frames = []

        self.pivot_grid = defaultdict(lambda: defaultdict(dict))

        self.top_beams_y = []
        self.bottom_beams_y = []

        # create cells and beams
        # xy plane
        for nz in range(ncells_z + 1):
            i = nz % 2
            for nx in range(i, ncells_x + 1, 2):
                r_OPp_1 = np.sqrt(2) * L / 2 * np.array([nx, 0, nz]) + np.array(
                    [0, 0, piv_h / 2]
                )
                r_OPp_2 = np.sqrt(2) * L / 2 * np.array([nx, 0, nz]) - np.array(
                    [0, 0, piv_h / 2]
                )
                k_1 = min(ncells_x - nx, ncells_y)
                k_2 = min(nx, ncells_y)
                if k_1 != 0:
                    beam1 = Panto_beam(p, q, nEl, L, r_OPp_1, k_1, 1)
                    self.beams.append(beam1)
                    for k_i in range(k_1 + 1):
                        self.pivot_grid[nx + k_i, k_i, nz][1] = {
                            "beam": beam1,
                            "f_ID": (k_i / k_1,),
                            "joints": [],
                        }

                if k_2 != 0:
                    beam2 = Panto_beam(p, q, nEl, L, r_OPp_2, k_2, 2)
                    self.beams.append(beam2)
                    for k_i in range(k_2 + 1):
                        self.pivot_grid[nx - k_i, k_i, nz][2] = {
                            "beam": beam2,
                            "f_ID": (k_i / k_2,),
                            "joints": [],
                        }

            for ny in range(2 - i, ncells_y + 1, 2):
                r_OPp_1 = np.sqrt(2) * L / 2 * np.array([0, ny, nz]) + np.array(
                    [0, 0, piv_h / 2]
                )
                r_OPp_2 = np.sqrt(2) * L / 2 * np.array([ncells_x, ny, nz]) - np.array(
                    [0, 0, piv_h / 2]
                )
                k_1 = min(ncells_y - ny, ncells_x)
                k_2 = min(ncells_y - ny, ncells_x)
                if k_1 != 0:
                    beam1 = Panto_beam(p, q, nEl, L, r_OPp_1, k_1, 1)
                    self.beams.append(beam1)

                    for k_i in range(k_1 + 1):
                        self.pivot_grid[k_i, ny + k_i, nz][1] = {
                            "beam": beam1,
                            "f_ID": (k_i / k_1,),
                            "joints": [],
                        }
                if k_2 != 0:
                    beam2 = Panto_beam(p, q, nEl, L, r_OPp_2, k_2, 2)
                    self.beams.append(beam2)
                    for k_i in range(k_2 + 1):
                        self.pivot_grid[ncells_x - k_i, ny + k_i, nz][2] = {
                            "beam": beam2,
                            "f_ID": (k_i / k_2,),
                            "joints": [],
                        }
        # yz-plane
        for nx in range(ncells_x + 1):
            i = nx % 2
            for ny in range(i, ncells_y + 1, 2):
                r_OPp_1 = np.sqrt(2) * L / 2 * np.array([nx, ny, 0]) - np.array(
                    [piv_h / 2, 0, 0]
                )
                r_OPp_2 = np.sqrt(2) * L / 2 * np.array([nx, ny, 0]) + np.array(
                    [piv_h / 2, 0, 0]
                )
                k_1 = min(ncells_y - ny, ncells_z)
                k_2 = min(ny, ncells_z)
                if k_1 != 0:
                    beam1 = Panto_beam(p, q, nEl, L, r_OPp_1, k_1, 3)
                    self.beams.append(beam1)
                    for k_i in range(k_1 + 1):
                        self.pivot_grid[nx, ny + k_i, k_i][3] = {
                            "beam": beam1,
                            "f_ID": (k_i / k_1,),
                            "joints": [],
                        }

                if k_2 != 0:
                    beam2 = Panto_beam(p, q, nEl, L, r_OPp_2, k_2, 4)
                    self.beams.append(beam2)
                    for k_i in range(k_2 + 1):
                        self.pivot_grid[nx, ny - k_i, k_i][4] = {
                            "beam": beam2,
                            "f_ID": (k_i / k_2,),
                            "joints": [],
                        }

            for nz in range(2 - i, ncells_z + 1, 2):
                r_OPp_1 = np.sqrt(2) * L / 2 * np.array([nx, 0, nz]) - np.array(
                    [piv_h / 2, 0, 0]
                )
                r_OPp_2 = np.sqrt(2) * L / 2 * np.array([nx, ncells_y, nz]) + np.array(
                    [piv_h / 2, 0, 0]
                )
                k_1 = min(ncells_z - nz, ncells_y)
                k_2 = min(ncells_z - nz, ncells_y)
                if k_1 != 0:
                    beam1 = Panto_beam(p, q, nEl, L, r_OPp_1, k_1, 3)
                    self.beams.append(beam1)
                    for k_i in range(k_1 + 1):
                        self.pivot_grid[nx, k_i, nz + k_i][3] = {
                            "beam": beam1,
                            "f_ID": (k_i / k_1,),
                            "joints": [],
                        }
                if k_2 != 0:
                    beam2 = Panto_beam(p, q, nEl, L, r_OPp_2, k_2, 4)
                    self.beams.append(beam2)
                    for k_i in range(k_2 + 1):
                        self.pivot_grid[nx, ncells_y - k_i, nz + k_i][4] = {
                            "beam": beam2,
                            "f_ID": (k_i / k_2,),
                            "joints": [],
                        }

        # Rigid body pivots
        bc_dic = {"x": 0, "y": 1, "z": 2}
        bc_it = list(n_xyz)
        bc_i = bc_dic[bc_dir]
        bc_it[bc_i] = 0
        for nz in range(n_xyz[2] + 1):
            for ny in range(n_xyz[1] + 1):
                for nx in range(n_xyz[0] + 1):
                    # if [nx, ny, nz][bc_i] in [0, n_xyz[bc_i]]:
                    #    continue
                    pivot = self.pivot_grid[nx, ny, nz]
                    # and ([nx, ny, nz][bc_i] not in [0, n_xyz[bc_i]]):
                    if pivot:
                        r_OBp = np.sqrt(2) * L / 2 * np.array([nx, ny, nz])
                        pivot["pivot"] = Pivot(r_OBp, rigid=rigid_pivot)
                        rb = pivot["pivot"]
                        # self.pivot_beams.append(pivot['pivot'].beam1)
                        # self.pivot_beams.append(pivot['pivot'].beam2)
                        if rigid_pivot:
                            self.joints.extend({bod for bod in rb.body})
                        else:
                            self.beams.extend(rb.beams)
                        self.joints.extend(rb.rigid)
                        for i in [1, 2, 3, 4]:
                            if pivot[i]:
                                beam = pivot[i]["beam"]
                                f_ID = pivot[i]["f_ID"]
                                r_OB = beam.r_OP(
                                    0, beam.q0[beam.local_qDOF_P(f_ID)], f_ID
                                )

                                # joints between beams and pivot on boundaries
                                if [nx, ny, nz][bc_i] in [0, n_xyz[bc_i]]:
                                    # if bc_it == 0:
                                    #     frame = frame_bottom
                                    # else:
                                    #     frame = frame_top
                                    # # sph = Spherical_joint(beam, rb.body[i], r_OB, f_ID)
                                    sph = Rigid_connection(
                                        beam, rb.body[i], r_OB, f_ID, rb.f_ID[i - 1]
                                    )
                                    # sph = Rigid_connection(beam, frame, r_OB, f_ID)
                                # joints between beams and pivots
                                else:
                                    # sph = Saddle_joint(
                                    #     beam, rb.body[i], r_OB, A_IK_list[i-1], f_ID, rb.f_ID[i-1])

                                    # sph = Spherical_joint(beam, rb.body[i], r_OB, f_ID, rb.f_ID[i-1])
                                    # sph = Rigid_connection(beam, rb.body[i], r_OB, f_ID)
                                    sph = Revolute_joint(
                                        beam,
                                        rb.body[i],
                                        r_OB,
                                        A_IK_list[i - 1],
                                        f_ID,
                                        rb.f_ID[i - 1],
                                    )

                                self.joints.append(sph)

        # collect boundary beams and pivots
        self.bc = []
        bc_beams = {"top": [], "bottom": []}
        bc_pivots = {"top": [], "bottom": [], "middle": []}
        for nz in range(bc_it[2] + 1):
            for ny in range(bc_it[1] + 1):
                for nx in range(bc_it[0] + 1):
                    bc_bottom = [nx, ny, nz]
                    bc_top = [nx, ny, nz]
                    bc_bottom[bc_i] = 0
                    bc_top[bc_i] = n_xyz[bc_i]
                    bc_bottom = tuple(bc_bottom)
                    bc_top = tuple(bc_top)
                    beams_top = list(
                        (k, v)
                        for k, v in self.pivot_grid[bc_top].items()
                        if v and k in [1, 2, 3, 4]
                    )
                    beams_bottom = list(
                        (k, v)
                        for k, v in self.pivot_grid[bc_bottom].items()
                        if v and k in [1, 2, 3, 4]
                    )
                    # if beams_top.items():
                    bc_beams["top"].extend(beams_top)
                    bc_beams["bottom"].extend(beams_bottom)

                    if self.pivot_grid[bc_top]["pivot"]:
                        bc_pivots["top"].append(self.pivot_grid[bc_top]["pivot"])
                    if self.pivot_grid[bc_bottom]["pivot"]:
                        bc_pivots["bottom"].append(self.pivot_grid[bc_bottom]["pivot"])

                    # if self.pivot_grid[(nx, ny, 1)]['pivot'] not in bc_pivots['middle'] and self.pivot_grid[(nx, ny, 1)]['pivot']:
                    #     bc_pivots['middle'].append(
                    #         self.pivot_grid[(nx, ny, 1)]['pivot'])

        # self.bc = []
        # rigid connection between beams and frame
        if not self.pivot_grid[bc_top]["pivot"].is_rigid:
            for _, beam_bottom in bc_beams["bottom"]:
                beam = beam_bottom["beam"]
                r_OB_bottom = beam.r_OP(
                    0,
                    beam.q0[beam.local_qDOF_P(beam_bottom["f_ID"])],
                    beam_bottom["f_ID"],
                )
                rigid_bottom = Rigid_connection(
                    beam, frame_bottom, r_OB_bottom, beam_bottom["f_ID"]
                )

                self.bc.append(rigid_bottom)

            for _, beam_top in bc_beams["top"]:
                beam = beam_top["beam"]
                r_OB_top = beam.r_OP(
                    0, beam.q0[beam.local_qDOF_P(beam_top["f_ID"])], beam_top["f_ID"]
                )
                rigid_top = Rigid_connection(
                    beam, frame_top, r_OB_top, beam_top["f_ID"]
                )
                rigid_top = Rigid_connection(
                    beam, frame_top, r_OB_top, beam_top["f_ID"]
                )

                self.bc.append(rigid_top)

        # Rigid connection between pivots and frame
        # if rigid:
        else:
            for i, pivot in enumerate(bc_pivots["top"]):
                r_OB = pivot.r_OB
                if i == -1:
                    self.bc.append(Rigid_connection(pivot.body[0], frame_top, r_OB))
                else:
                    connection = Rigid_connection(pivot.body[0], frame_top2, r_OB)
                    # self.bc.append(Rigid_connection(pivot.body[5], frame_top2, r_OB))
                    # self.bc.append(Single_position_all_angles(
                    #     pivot.body[0], frame_top, r_OB, A_IK_x))
                    # self.bc.append(Single_position_y(
                    #    pivot.body[0], frame_top, r_OB, A_IK_y))
                    # self.bc.append(Linear_guidance_x(pivot.body[0], frame_top2, r_OB, A_IK_z))
                    # connection = Spherical_joint(pivot.body[0], frame_top2, r_OB)

                self.bc.append(connection)

            for i, pivot in enumerate(bc_pivots["bottom"]):
                r_OB = pivot.r_OB
                if i == -1:
                    self.bc.append(Rigid_connection(pivot.body[0], frame_bottom, r_OB))
                    # self.bc.append(Rigid_connection(
                    #     pivot.body[5], frame_bottom, r_OB))
                else:
                    # self.bc.append(Single_position_y(
                    #    pivot.body[0], frame_bottom, r_OB, A_IK_y))
                    # self.bc.append(Single_position_all_angles(
                    #     pivot.body[0], frame_bottom, r_OB, A_IK_x))
                    # self.bc.append(Linear_guidance_xyz(
                    #     pivot.body[0], frame_bottom, r_OB, A_IK_x))
                    connection = Rigid_connection(pivot.body[0], frame_bottom, r_OB)
                    # self.bc.append(Rigid_connection(
                    #     pivot.body[5], frame_bottom, r_OB))
                    # connection = Spherical_joint(pivot.body[0], frame_bottom, r_OB)
                self.bc.append(connection)

        # for pivot in bc_pivots['middle']:
        #     r_OB = pivot.r_OB
        #     # self.bc.append(Single_position_y(pivot.body[0], frame_middle, r_OB, A_IK_z))
        #     self.bc.append(Single_position_all_angles(
        #         pivot.body[0], frame_middle, r_OB, A_IK_z))


#  create unit cells
ncells_x = 4
ncells_y = 4
ncells_z = 12
ncells = (ncells_x, ncells_y, ncells_z)

# Material and Beam parameters parameters
u_l = 1.0  # 1e-3
u_Pa = 1.0  # 1e9
l = 70.0 * 3 * u_l  # length in mm
# Beam dimensions and parameter
L = l / np.sqrt(2) / ncells_z * 2.0  # beam length between pivots
r = 0.45 * u_l  # pivot radius in mm
E_Y = 50.0 * u_Pa  # Young's Modulus in GPa
a = L / 5  # 1. * u_l  # Beam cross section length 1 in mm
b = L / 5  # 1. * u_l  # Beam cross section length 2 in mm
G = E_Y / (2 + 0.8)
I_1 = 2.25 * (a / 2) ** 4  # torsional moment for square cross-section, ncells_z
I_2 = a**3 * b / 12  # Bending moments
I_3 = a * b**3 / 12
# I_i = np.pi * r**4 * np.array([1/2, 1/4, 1/4])
I_P = I_1 + I_2
A = a * b
Ei = np.array([E_Y, G, G]) * A
Fi = np.array([G * I_1, E_Y * I_2, E_Y * I_3])

rho = 2.7e-1
A_rho0 = rho * A
B_rho0 = np.zeros(3)
C_rho0 = np.array([[0, 0, 0], [0, I_3, 0], [0, 0, I_2]]) * rho

material_model = Hooke_quadratic(Ei, Fi)

# Pivot length and type
piv_h = 1.5 * u_l * 0  # mm
rigid_pivot = True
cross = False

# Discretization
basis = "B-spline"
# basis = 'lagrange'
greville = True
p = 3
q = 2
nQP = p + 1
# nQP = int(np.ceil((p**2 + 1) / 2)) + 1
nEl = 3

# dynamic solver?
dynamic = False

save_sol = True
load_sol = False

# create model
model = Model()

# boundary condittions
bc_dir = "z"
r_OB_top0 = l / 3 * np.array([0.5, 0.5, 3.0])

# test
tests = ["torsion", str(l), "fixed_boundary"]

if "tension" in tests:

    def r_OP_top(t):
        return r_OB_top0 + t * 50.0 * np.array([0.0, 0.0, 1.0]) * u_l

    A_IK_top = np.eye(3)
    frame_top2 = Frame(r_OP_top)
    model.add(frame_top2)

if "torsion" in tests:

    def r_OP_top(t):
        return r_OB_top0  # + t * 30.0 * np.array([0.0, 0.0, 1.0])

    def A_IK_top(t):
        return A_IK_basic_z(t * np.pi / 4)

    frame_top2 = Frame(r_OP_top, A_IK=A_IK_top)
    model.add(frame_top2)

if "force" in tests:

    def force(t):
        return 0.05 * t * np.array([0.0, 0.0, 1.0])

    frame_top2 = Rigid_body_euler(1, np.eye(3))
    frame_top = Frame(r_OB_top0)
    z_only = Linear_guidance_x(frame_top, frame_top2, r_OB_top0, A_IK_z)
    # rigid_frame_body = Rigid_connection(frame_top, frame_top2, r_OB_top0)
    T_Force = Force(force, frame_top2)
    model.add(frame_top2)
    model.add(T_Force)
    model.add(z_only)
    # model.add(rigid_frame_body)

# make Pantobox
n_yxz = ncells_x, ncells_y, ncells_z
grid = Panto_grid((n_yxz))

# assemble
beams_all = []

# frames
frame_bottom = Frame(np.zeros(3))

model.add(frame_bottom)
# model.add(frame_middle)
# model.add(frame_top)
for beam in grid.beams:
    model.add(beam)
    beams_all.append(beam)
for beam in grid.pivot_beams:
    model.add(beam)
    beams_all.append(beam)
for joint in grid.joints:
    model.add(joint)
# for rigid in np.concatenate([grid.rigid_top, grid.rigid_bottom]):
#    model.add(rigid)
for bc in grid.bc:
    model.add(bc)
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
        uDOF_algebraic.extend(beam.uDOF[rDOF : rDOF + dDOF])  # whole beam dynamic
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
        model,
        t1,
        dt,
        uDOF_algebraic=uDOF_algebraic,
        rho_inf=rho_inf,
        newton_max_iter=max_iter,
        newton_tol=tol,
    ).solve()

    sol = solver
    t = sol.t
    q = sol.q

else:
    solver = Newton(model, n_load_steps=10, max_iter=20, tol=1e-6)

    # sol = solver.solve()
    # t = sol.t
    # q = sol.q


file_name = pathlib.Path(__file__).stem
file_path = (
    pathlib.Path(__file__).parent
    / "results"
    / str(
        f"{file_name}_"
        + "_".join(tests)
        + "_"
        + "x".join([str(v) for v in ncells])
        + "_".join(tests)
    )
    / file_name
)
file_path.parent.mkdir(parents=True, exist_ok=True)
export_path = file_path.parent / "sol"

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

    sol = pickle.load(open(str(export_path), "rb"))


# vtk export
post_processing(beams_all, sol.t, sol.q, file_path, binary=True)
