from cardillo.constraints import RigidConnection, RevoluteJoint, SphericalJoint
from cardillo.discrete import Frame

from cardillo.forces import LinearSpring, PDRotationalJoint
from cardillo.math import A_IK_basic
from cardillo.beams import (
    TimoshenkoAxisAngleSE3,
    DirectorAxisAngle,
    TimoshenkoDirectorIntegral,
)
from cardillo.beams import Simo1986
from cardillo.beams.animate import animate_beam
from cardillo.solver import Newton
from cardillo import System

import numpy as np
from math import sin, cos, pi
from scipy.sparse.linalg import spsolve
import meshio
from pathlib import Path


# Beam = TimoshenkoAxisAngleSE3
Beam = DirectorAxisAngle
# Beam = TimoshenkoDirectorIntegral

use_revolute_joints = True
# use_revolute_joints = False

# indenter_support = "RevoluteJoint"
# indenter_support = "PointPlane"
# indenter_support = "RevoluteJointPointPlane"
# indenter_support = "PointCylinder"
indenter_support = "FixedFixed"
# indenter_support = "FixedLeft"

# TODO:
# - split of atol in position and force contributions
# - PointPlane problems with rigid body rotations
# - check ocnsistency of
# - move boundary beam initial positon and pivot xi's
# - vtk export:
#       * beam volumes
#       * vtk export contact planes
#       * vtk export contact cylinders
# - point2cylinder contacts for support/ bottom knives
# - move bottom supports instead of indenter/ use a flag
# - study torsional spring stiffness influence


class PantographicSheet:
    def __init__(
        self,
        W,
        H,
        gamma,
        n_row,
        n_col,
        n_fiber_layer,
        pivot_factor,
        fiber_factor,
        statics,
    ):
        # geometry
        self.gamma = gamma
        self.W = W
        self.H = H
        self.n_row = n_row
        self.n_col = n_col
        self.n_fiber_layer = n_fiber_layer
        self.n_pivot_layer = n_fiber_layer - 1
        self.w = W / (n_row - 1)
        self.lPivot = H / self.n_pivot_layer
        self.lBeam = self.w / sin(gamma)
        self.l = self.lBeam * cos(gamma)
        self.L = (n_col - 1) * self.l
        self.pivot_factor = pivot_factor
        self.fiber_factor = fiber_factor

        # upper left node
        xUL = 0.0
        yUL = W
        zUL = 0.0
        self.origin = np.array([xUL, yUL, zUL])

        # # # pivot geometry
        # r_Pi = 1.0e-3
        # r_Po = 2.0e-3
        # r_P = 0.5 * (r_Pi + r_Po)
        # A_P = pi * r_P**2
        # I2_P = I3_P = 0.25 * pi * r_P**4
        # IP_P = I2_P + I3_P

        # # fiber geometry
        # self.w_F = 4.4  # mm
        # self.h_F = 2.0  # mm
        # A_F = self.w_F * self.h_F
        # I2_F = (self.w_F * self.h_F ** 3) / 12.0
        # I3_F = (self.w_F ** 3 * self.h_F) / 12.0
        # IP_F = I2_F + I3_F

        # # boundary beams geometry
        # factor = 2.5
        # self.w_B = factor * self.w_F
        # self.h_B = factor * self.h_F
        # A_B = self.w_B * self.h_B
        # I2_B = (self.w_B * self.h_B ** 3) / 12.0
        # I3_B = (self.w_B ** 3 * self.h_B) / 12.0
        # IP_B = I2_B + I3_B

        # # PMMA (https://designerdata.nl/materials/plastics/thermo-plastics/poly(methyl-methacrylate))
        # rho = 1.19e3 # 1.19 g/cm3 = 1.19 * 1e-3 * 1e-6 kg / m^3.
        # E = 3e9 # Pa
        # nu = 0.5 # Possion's ratio
        # G = E / (2 * (1 + nu)) # Pa

        # pivot geometry
        r_Pi = 1.0  # mm
        r_Po = 2.0  # mm
        r_P = 0.5 * (r_Pi + r_Po)
        A_P = pi * r_P**2
        I2_P = I3_P = 0.25 * pi * r_P**4
        IP_P = I2_P + I3_P

        # fiber geometry
        self.w_F = 4.4  # mm
        self.h_F = 2.0  # mm
        A_F = self.w_F * self.h_F
        I2_F = (self.w_F * self.h_F**3) / 12.0
        I3_F = (self.w_F**3 * self.h_F) / 12.0
        IP_F = I2_F + I3_F

        # boundary beams geometry
        factor = 2.5
        self.w_B = factor * self.w_F
        self.h_B = factor * self.h_F
        A_B = self.w_B * self.h_B
        I2_B = (self.w_B * self.h_B**3) / 12.0
        I3_B = (self.w_B**3 * self.h_B) / 12.0
        IP_B = I2_B + I3_B

        # PMMA (https://designerdata.nl/materials/plastics/thermo-plastics/poly(methyl-methacrylate))
        rho = 1.19e-6  # kg / mm^3.
        # rho = 1.19e-3  # g / mm^3.
        # E = 3e6 # kg / (s^2 * mm)
        # E = 3e0  # kg / (ms^2 * mm)
        # E = 3e3  # g / (ms^2 * mm) = MPa = N / mm^2
        E = 3e-3  # GPa = kN / mm^2
        nu = 0.5  # Possion's ratio
        G = E / (2 * (1 + nu))

        k_s = 1  # shear factor

        Ei_P = np.array([E * A_P, G * A_P, G * A_P])
        Fi_P = np.array([E * IP_P, E * I2_P, E * I3_P])

        Ei_F = np.array([E * A_F, k_s * G * A_F, k_s * G * A_F])
        Fi_F = np.array([E * IP_F, E * I2_F, E * I3_F])

        Ei_B = np.array([E * A_B, k_s * G * A_B, k_s * G * A_B])
        Fi_B = np.array([E * IP_B, E * I2_B, E * I3_B])

        A_rho0_P = rho * A_P
        B_rho0_P = np.zeros(3)  # symmetric cross section!
        C_rho0_P = rho * np.diag(np.array([0, I3_P, I2_P]))
        I_rho0_P = rho * np.diag([IP_P, I2_P, I3_P])

        A_rho0_F = rho * A_F
        B_rho0_F = np.zeros(3)  # symmetric cross section!
        C_rho0_F = rho * np.diag(np.array([0, I3_F, I2_F]))
        I_rho0_F = rho * np.diag([IP_F, I2_F, I3_F])

        A_rho0_B = rho * A_B
        B_rho0_B = np.zeros(3)  # symmetric cross section!
        C_rho0_B = rho * np.diag(np.array([0, I3_B, I2_B]))
        I_rho0_B = rho * np.diag([IP_B, I2_B, I3_B])

        # discretization properties of each beam
        # p = 3
        # self.p_r = p
        # self.p_di = p - 1
        p = self.p_r = self.p_di = 1
        if statics:
            self.nQP = p + 1
        else:
            self.nQP = int(np.ceil((p + 1) ** 2 / 2))
        self.nEl = 1

        # self.basis = "B-spline"
        self.basis = "Lagrange"

        def create_director_beam(NEl, L, r_OP, A_IK, type):
            if Beam == TimoshenkoDirectorIntegral:
                Q = TimoshenkoDirectorIntegral.straight_configuration(
                    self.p_r, self.p_di, NEl, L, r_OP=r_OP, A_IK=A_IK, basis=self.basis
                )
            elif Beam == TimoshenkoAxisAngleSE3:
                Q = TimoshenkoAxisAngleSE3.straight_configuration(
                    1, NEl, L, r_OP=r_OP, A_IK=A_IK
                )
            elif Beam == DirectorAxisAngle:
                Q = DirectorAxisAngle.straight_configuration(
                    self.p_r,
                    self.p_di,
                    self.basis,
                    self.basis,
                    NEl,
                    L,
                    r_OP=r_OP,
                    A_IK=A_IK,
                )
            else:
                raise NotImplementedError

            q0 = Q.copy()
            u0 = np.zeros_like(Q)

            if type == "pivot":
                # return Beam(
                #     Simo1986(Ei_P, Fi_P),
                #     A_rho0_P,
                #     B_rho0_P,
                #     C_rho0_P,
                #     self.p_r,
                #     self.p_di,
                #     self.nQP,
                #     NEl,
                #     Q=Q.copy(),
                #     q0=q0.copy(),
                #     u0=u0.copy(),
                # )

                if Beam == TimoshenkoDirectorIntegral:
                    return TimoshenkoDirectorIntegral(
                        Simo1986(Ei_P, Fi_P),
                        A_rho0_P,
                        B_rho0_P,
                        C_rho0_P,
                        self.p_r,
                        self.p_di,
                        self.nQP,
                        NEl,
                        q0,
                        basis=self.basis,
                    )
                elif Beam == TimoshenkoAxisAngleSE3:
                    return TimoshenkoAxisAngleSE3(
                        1,
                        Simo1986(Ei_P, Fi_P),
                        A_rho0_P,
                        np.zeros(3),
                        I_rho0_P,
                        NEl,
                        Q=Q.copy(),
                        q0=q0.copy(),
                        u0=u0.copy(),
                    )
                elif Beam == DirectorAxisAngle:
                    return DirectorAxisAngle(
                        Simo1986(Ei_P, Fi_P),
                        A_rho0_P,
                        np.zeros(3),
                        I_rho0_P,
                        self.p_r,
                        self.p_di,
                        NEl,
                        Q=Q.copy(),
                        q0=q0.copy(),
                        u0=u0.copy(),
                        basis_r=self.basis,
                        basis_psi=self.basis,
                    )
                else:
                    raise NotImplementedError

            elif type == "fiber":
                # return Beam(
                #     Simo1986(Ei_F, Fi_F),
                #     A_rho0_F,
                #     B_rho0_F,
                #     C_rho0_F,
                #     self.p_r,
                #     self.p_di,
                #     self.nQP,
                #     NEl,
                #     Q=Q.copy(),
                #     q0=q0.copy(),
                #     u0=u0.copy(),
                # )
                if Beam == TimoshenkoDirectorIntegral:
                    return TimoshenkoDirectorIntegral(
                        Simo1986(Ei_F, Fi_F),
                        A_rho0_F,
                        B_rho0_F,
                        C_rho0_F,
                        self.p_r,
                        self.p_di,
                        self.nQP,
                        NEl,
                        q0,
                        basis=self.basis,
                    )
                elif Beam == TimoshenkoAxisAngleSE3:
                    return TimoshenkoAxisAngleSE3(
                        1,
                        Simo1986(Ei_F, Fi_F),
                        A_rho0_F,
                        np.zeros(3),
                        I_rho0_F,
                        NEl,
                        Q=Q.copy(),
                        q0=q0.copy(),
                        u0=u0.copy(),
                    )
                elif Beam == DirectorAxisAngle:
                    return DirectorAxisAngle(
                        Simo1986(Ei_F, Fi_F),
                        A_rho0_F,
                        np.zeros(3),
                        I_rho0_F,
                        self.p_r,
                        self.p_di,
                        NEl,
                        Q=Q.copy(),
                        q0=q0.copy(),
                        u0=u0.copy(),
                        basis_r=self.basis,
                        basis_psi=self.basis,
                    )
                else:
                    raise NotImplementedError

            elif type == "boundary":
                # return Beam(
                #     Simo1986(Ei_B, Fi_B),
                #     A_rho0_B,
                #     B_rho0_B,
                #     C_rho0_B,
                #     self.p_r,
                #     self.p_di,
                #     self.nQP,
                #     NEl,
                #     Q=Q.copy(),
                #     q0=q0.copy(),
                #     u0=u0.copy(),
                # )
                if Beam == TimoshenkoDirectorIntegral:
                    return TimoshenkoDirectorIntegral(
                        Simo1986(Ei_B, Fi_B),
                        A_rho0_B,
                        B_rho0_B,
                        C_rho0_B,
                        self.p_r,
                        self.p_di,
                        self.nQP,
                        NEl,
                        q0,
                        basis=self.basis,
                    )
                elif Beam == TimoshenkoAxisAngleSE3:
                    return TimoshenkoAxisAngleSE3(
                        1,
                        Simo1986(Ei_B, Fi_B),
                        A_rho0_B,
                        np.zeros(3),
                        I_rho0_B,
                        NEl,
                        Q=Q.copy(),
                        q0=q0.copy(),
                        u0=u0.copy(),
                    )
                elif Beam == DirectorAxisAngle:
                    return DirectorAxisAngle(
                        Simo1986(Ei_B, Fi_B),
                        A_rho0_B,
                        np.zeros(3),
                        I_rho0_B,
                        self.p_r,
                        self.p_di,
                        NEl,
                        Q=Q.copy(),
                        q0=q0.copy(),
                        u0=u0.copy(),
                        basis_r=self.basis,
                        basis_psi=self.basis,
                    )
                else:
                    raise NotImplementedError
            else:
                raise RuntimeError(f'Wrong beam type "{type}" given.')

        self.create_beam = create_director_beam


def create_pantographic_sheet(model, pantographic_sheet):
    l = pantographic_sheet.l
    w = pantographic_sheet.w
    lPivot = pantographic_sheet.lPivot
    pivot_factor = pantographic_sheet.pivot_factor
    fiber_factor = pantographic_sheet.fiber_factor
    assert pivot_factor <= 1 and pivot_factor >= 0
    assert fiber_factor <= 1 and fiber_factor >= 0

    lBeam = pantographic_sheet.lBeam
    nEl = pantographic_sheet.nEl
    n_row = pantographic_sheet.n_row
    n_col = pantographic_sheet.n_col
    n_fiber_layer = pantographic_sheet.n_fiber_layer
    assert n_row >= 2
    assert n_col >= 2
    assert n_col >= n_row

    pantographic_sheet.rigid_bodies = []
    pantographic_sheet.beams = []
    pantographic_sheet.constraints = []
    pantographic_sheet.pivots_dict = np.zeros(
        (n_row, n_col, n_fiber_layer + 1), dtype=dict
    )
    for row in range(n_row):
        for col in range(n_col):
            for layer in range(n_fiber_layer):
                pantographic_sheet.pivots_dict[row, col, layer] = dict()

    ######################################################
    # create reference configuration of individual beams #
    ######################################################

    # transformation matrix
    A_IK0m = A_IK_basic(-gamma).z()
    A_IK0p = A_IK_basic(+gamma).z()
    A_IK0pivot = A_IK_basic(-pi / 2).y()

    # build pivot beams
    for row in range(0, n_row):
        for col in range(0, n_col):
            if (row + col) % 2 == 0:
                # individual beam length and number of elements
                n_segments = n_fiber_layer - 1
                LPivot = (n_segments + 2 * pivot_factor) * lPivot
                NEl = (n_segments + 2) * nEl

                # origin of beam
                r_OP0 = np.array([col * l, row * w, -pivot_factor * lPivot])

                # create beam
                beam = pantographic_sheet.create_beam(
                    NEl, LPivot, r_OP0, A_IK0pivot, "pivot"
                )
                pantographic_sheet.beams.append(beam)

                xi_offset = (pivot_factor * lPivot) / LPivot
                Delta_xi = lPivot / LPivot
                for i in range(n_fiber_layer):
                    xi = xi_offset + i * Delta_xi
                    pantographic_sheet.pivots_dict[row, col, i]["pivot"] = beam
                    pantographic_sheet.pivots_dict[row, col, i]["pivot_xi"] = xi

    # build fiber layers
    row_range = range(0, n_row, 2)
    col_range = range(1, n_col)
    for layer in range(n_fiber_layer):
        for row in row_range:
            col = 0
            # individual beam length and number of elements
            if layer % 2 == 0:
                n_segments = n_row - 1 - row
                A_IK0 = A_IK0p
            else:
                A_IK0 = A_IK0m
                n_segments = row

            if (row + col) % 2 != 0:  # no fiber
                continue

            # if (n_segments == 0): # no fiber
            #     continue

            k = n_segments
            LFiber = (k + 2 * fiber_factor) * lBeam
            NEl = (k + 2) * nEl

            # origin of beam
            if layer % 2 == 0:
                offset = (
                    fiber_factor * lBeam * np.array([-cos(gamma), -sin(gamma), 0.0])
                )
            else:
                offset = fiber_factor * lBeam * np.array([-cos(gamma), sin(gamma), 0.0])
            r_OP0 = np.array([0.0, row * w, layer * lPivot]) + offset

            # create beam
            beam = pantographic_sheet.create_beam(NEl, LFiber, r_OP0, A_IK0, "fiber")
            pantographic_sheet.beams.append(beam)

            xi_offset = (fiber_factor * lBeam) / LFiber
            Delta_xi = lBeam / LFiber
            for i in range(k + 1):
                xi = xi_offset + i * Delta_xi
                if layer % 2 == 0:
                    pantographic_sheet.pivots_dict[row + i, col + i, layer][
                        "fiber"
                    ] = beam
                    pantographic_sheet.pivots_dict[row + i, col + i, layer][
                        "fiber_xi"
                    ] = xi
                else:
                    pantographic_sheet.pivots_dict[row - i, col + i, layer][
                        "fiber"
                    ] = beam
                    pantographic_sheet.pivots_dict[row - i, col + i, layer][
                        "fiber_xi"
                    ] = xi

        for col in col_range:
            # individual beam length and number of elements
            n_segments_complete = n_row - 1
            if layer % 2 == 0:
                row = 0
                A_IK0 = A_IK0p
            else:
                row = n_row - 1
                A_IK0 = A_IK0m
            n_segments = n_col - 1 - col

            if (row + col) % 2 != 0:  # no fiber
                continue

            # if (n_segments == 0): # no fiber
            #     continue

            k = min(n_segments_complete, n_segments)
            LFiber = (k + 2 * fiber_factor) * lBeam
            NEl = (k + 2) * nEl

            # origin of beam
            if layer % 2 == 0:
                offset = (
                    fiber_factor * lBeam * np.array([-cos(gamma), -sin(gamma), 0.0])
                )
            else:
                offset = fiber_factor * lBeam * np.array([-cos(gamma), sin(gamma), 0.0])
            r_OP0 = np.array([col * l, row * w, layer * lPivot]) + offset

            # create beam
            beam = pantographic_sheet.create_beam(NEl, LFiber, r_OP0, A_IK0, "fiber")
            pantographic_sheet.beams.append(beam)

            xi_offset = (fiber_factor * lBeam) / LFiber
            Delta_xi = lBeam / LFiber
            for i in range(k + 1):
                xi = xi_offset + i * Delta_xi
                if layer % 2 == 0:
                    pantographic_sheet.pivots_dict[row + i, col + i, layer][
                        "fiber"
                    ] = beam
                    pantographic_sheet.pivots_dict[row + i, col + i, layer][
                        "fiber_xi"
                    ] = xi
                else:
                    pantographic_sheet.pivots_dict[row - i, col + i, layer][
                        "fiber"
                    ] = beam
                    pantographic_sheet.pivots_dict[row - i, col + i, layer][
                        "fiber_xi"
                    ] = xi

    # build boundary beams
    for col in [0, n_col - 1]:
        for layer in [0, n_fiber_layer - 1]:
            n_segments = n_row - 1
            NEl = n_segments * nEl
            LBoundary = n_segments * w
            # TODO: offset of the boundary fibers
            # if layer == 0:
            #     z_offset = -0.5 * (pantographic_sheet.h_F + pantographic_sheet.h_B)
            # else:
            #     z_offset = 0.5 * (pantographic_sheet.h_F + pantographic_sheet.h_B)
            r_OP0 = np.array([col * l, 0.0, layer * lPivot])
            A_IK0 = A_IK_basic(pi / 2).z()
            beam = pantographic_sheet.create_beam(
                NEl, LBoundary, r_OP0, A_IK0, "boundary"
            )
            pantographic_sheet.beams.append(beam)

            Delta_xi = w / W
            for row in range(n_row):
                if (row + col) % 2 == 0:
                    xi = row * Delta_xi
                    pantographic_sheet.pivots_dict[row, col, layer]["boundary"] = beam
                    pantographic_sheet.pivots_dict[row, col, layer]["boundary_xi"] = xi

    # add all beams to the model
    for beam in pantographic_sheet.beams:
        model.add(beam)

    # add junctions between fibers and pivots
    for row in range(pantographic_sheet.n_row):
        for col in range(pantographic_sheet.n_col):
            if (row + col) % 2 == 0:
                for layer in range(pantographic_sheet.n_fiber_layer):

                    pivot = pantographic_sheet.pivots_dict[row, col, layer]["pivot"]
                    pivot_xi = pantographic_sheet.pivots_dict[row, col, layer][
                        "pivot_xi"
                    ]
                    fiber = pantographic_sheet.pivots_dict[row, col, layer]["fiber"]
                    fiber_xi = pantographic_sheet.pivots_dict[row, col, layer][
                        "fiber_xi"
                    ]

                    frameID_pivot = (pivot_xi,)
                    frameID_fiber = (fiber_xi,)

                    # Zipfel in even layers
                    special_case_even = (layer % 2 == 0) and (
                        (row == 0 and col == n_col - 1)
                        or (row == n_row - 1 and col == 0)
                    )

                    # Zipfel in odd layers
                    special_case_odd = (layer % 2 != 0) and (
                        (row == 0 and col == 0)
                        or (row == n_row - 1 and col == n_col - 1)
                    )
                    if special_case_even or special_case_odd:
                        junction = RigidConnection(
                            pivot,
                            fiber,
                            frame_ID1=frameID_pivot,
                            frame_ID2=frameID_fiber,
                        )
                        pantographic_sheet.constraints.append(junction)
                        model.add(junction)
                        continue

                    # RevoluteJoint with soft torsional spring for all pivots
                    # in the first layer (get a statically determined problem).
                    # For symmetry reasons we might also add a spring at the
                    # last layer.
                    # TODO:
                    boundary_cases = (layer == 0) or (n_fiber_layer - 1)
                    # boundary_cases = layer == 0
                    if boundary_cases:
                        # TODO:
                        # k = 1.0e-6
                        k = 1.0e-3
                        spring = LinearSpring(k)

                        # position and orientation of the revolute joint
                        r_OB = fiber.r_OP(
                            model.t0,
                            fiber.q0[fiber.local_qDOF_P(frameID_fiber)],
                            frameID_fiber,
                        )
                        A_IB = np.eye(3)
                        # junction = RigidConnection(
                        #     pivot,
                        #     fiber,
                        #     frame_ID1=frameID_pivot,
                        #     frame_ID2=frameID_fiber,
                        # )
                        junction = PDRotationalJoint(
                            RevoluteJoint, force_law_spring=spring
                        )(
                            # junction = add_rotational_forcelaw(spring, RevoluteJoint)(
                            pivot,
                            fiber,
                            r_OB,
                            A_IB,
                            frame_ID1=frameID_pivot,
                            frame_ID2=frameID_fiber,
                        )
                        pantographic_sheet.constraints.append(junction)
                        model.add(junction)
                        continue

                    if use_revolute_joints:
                        # position and orientation of the revolute joint
                        r_OB = fiber.r_OP(
                            model.t0,
                            fiber.q0[fiber.local_qDOF_P(frameID_fiber)],
                            frameID_fiber,
                        )
                        A_IB = np.eye(3)
                        junction = RevoluteJoint(
                            pivot,
                            fiber,
                            r_OB,
                            A_IB,
                            frame_ID1=frameID_pivot,
                            frame_ID2=frameID_fiber,
                        )
                    else:
                        junction = RigidConnection(
                            pivot,
                            fiber,
                            frame_ID1=frameID_pivot,
                            frame_ID2=frameID_fiber,
                        )

                    pantographic_sheet.constraints.append(junction)
                    model.add(junction)

    # add junctions between boundary beams and pivots
    for col in [0, n_col - 1]:
        for layer in [0, n_fiber_layer - 1]:
            for row in range(n_row):
                if (row + col) % 2 == 0:
                    pivot = pantographic_sheet.pivots_dict[row, col, layer]["pivot"]
                    pivot_xi = pantographic_sheet.pivots_dict[row, col, layer][
                        "pivot_xi"
                    ]
                    boundary = pantographic_sheet.pivots_dict[row, col, layer][
                        "boundary"
                    ]
                    boundary_xi = pantographic_sheet.pivots_dict[row, col, layer][
                        "boundary_xi"
                    ]

                    frameID_pivot = (pivot_xi,)
                    frameID_boundary = (boundary_xi,)

                    if use_revolute_joints:
                        # position and orientation of the revolute joint
                        r_OB = boundary.r_OP(
                            model.t0,
                            boundary.q0[boundary.local_qDOF_P(frameID_boundary)],
                            frameID_boundary,
                        )
                        A_IB = np.eye(3)
                        junction = RevoluteJoint(
                            pivot,
                            boundary,
                            r_OB,
                            A_IB,
                            frame_ID1=frameID_pivot,
                            frame_ID2=frameID_boundary,
                        )
                    else:
                        junction = RigidConnection(
                            pivot,
                            boundary,
                            frame_ID1=frameID_pivot,
                            frame_ID2=frameID_boundary,
                        )

                    pantographic_sheet.constraints.append(junction)
                    model.add(junction)


def export_pivot_points(filename, pantographic_sheet, q):
    for k in range(len(q)):
        pivot_positions = []
        for row in range(pantographic_sheet.n_row):
            for col in range(pantographic_sheet.n_col):
                if (row + col) % 2 == 0:
                    for layer in range(pantographic_sheet.n_fiber_layer):
                        fiber = pantographic_sheet.pivots_dict[row, col, layer]["fiber"]
                        fiber_xi = pantographic_sheet.pivots_dict[row, col, layer][
                            "fiber_xi"
                        ]

                        frameID_fiber = (fiber_xi,)
                        r_OP = fiber.r_OP(
                            0,
                            q[k, fiber.qDOF[fiber.local_qDOF_P(frameID_fiber)]],
                            frameID_fiber,
                        )
                        pivot_positions.append(
                            [col, row, layer, r_OP[0], r_OP[1], r_OP[2]]
                        )
        filename_txt = filename + f"_{k}.txt"
        export_data = np.array(pivot_positions)
        header = "idx_x, idx_y, idx_z, x [mm], y [mm], z [mm]"
        np.savetxt(
            filename_txt,
            export_data,
            fmt="%i,%i,%i,%10.1f,%10.1f,%10.1f",
            delimiter=", ",
            header=header,
            comments="",
        )


def write_connecting_points(filename, pantographic_sheet):
    points = []
    cell_connectivity = []
    idx = 0
    for row in range(pantographic_sheet.n_row):
        for col in range(pantographic_sheet.n_col):
            if (row + col) % 2 == 0:
                for layer in range(pantographic_sheet.n_fiber_layer):
                    pivot = pantographic_sheet.pivots_dict[row, col, layer]["pivot"]
                    pivot_xi = pantographic_sheet.pivots_dict[row, col, layer][
                        "pivot_xi"
                    ]
                    fiber = pantographic_sheet.pivots_dict[row, col, layer]["fiber"]
                    fiber_xi = pantographic_sheet.pivots_dict[row, col, layer][
                        "fiber_xi"
                    ]

                    frameID_pivot = (pivot_xi,)
                    r_OP = pivot.r_OP(
                        0, pivot.q0[pivot.local_qDOF_P(frameID_pivot)], frameID_pivot
                    )
                    frameID_fiber = (fiber_xi,)
                    r_OQ = fiber.r_OP(
                        0, fiber.q0[fiber.local_qDOF_P(frameID_fiber)], frameID_fiber
                    )

                    points.extend([r_OP, r_OQ])
                    cell_connectivity.append([2 * idx, 2 * idx + 1])
                    idx = idx + 1

    print(f"len(points): {len(points)}")
    print(f"len(cell_connectivity): {len(cell_connectivity)}")
    cells = [("line", cell_connectivity)]

    meshio.write_points_cells(filename, points, cells, binary=True)


def write_xis_fibers(filename, pantographic_sheet):
    points = []
    for row in range(pantographic_sheet.n_row):
        for col in range(pantographic_sheet.n_col):
            if (row + col) % 2 == 0:
                for layer in range(pantographic_sheet.n_fiber_layer):
                    fiber = pantographic_sheet.pivots_dict[row, col, layer]["fiber"]
                    fiber_xi = pantographic_sheet.pivots_dict[row, col, layer][
                        "fiber_xi"
                    ]

                    frameID_fiber = (fiber_xi,)
                    r_OP = fiber.r_OP(
                        0, fiber.q0[fiber.local_qDOF_P(frameID_fiber)], frameID_fiber
                    )
                    points.extend([r_OP])

    cells = [("vertex", [[i] for i in range(len(points))])]
    meshio.write_points_cells(filename, points, cells, binary=True)


def write_xis_pivots(filename, pantographic_sheet):
    points = []
    for row in range(pantographic_sheet.n_row):
        for col in range(pantographic_sheet.n_col):
            if (row + col) % 2 == 0:
                for layer in range(pantographic_sheet.n_fiber_layer):
                    pivot = pantographic_sheet.pivots_dict[row, col, layer]["pivot"]
                    pivot_xi = pantographic_sheet.pivots_dict[row, col, layer][
                        "pivot_xi"
                    ]

                    frameID_pivot = (pivot_xi,)
                    r_OP = pivot.r_OP(
                        0, pivot.q0[pivot.local_qDOF_P(frameID_pivot)], frameID_pivot
                    )
                    points.extend([r_OP])

    cells = [("vertex", [[i] for i in range(len(points))])]
    meshio.write_points_cells(filename, points, cells, binary=False)


def write_xis_boundary(filename, pantographic_sheet):
    points = []
    for col in [0, n_col - 1]:
        for layer in [0, n_fiber_layer - 1]:
            for row in range(n_row):
                if (row + col) % 2 == 0:
                    boundary = pantographic_sheet.pivots_dict[row, col, layer][
                        "boundary"
                    ]
                    boundary_xi = pantographic_sheet.pivots_dict[row, col, layer][
                        "boundary_xi"
                    ]
                    frameID_boundary = (boundary_xi,)

                    r_OP = boundary.r_OP(
                        0,
                        boundary.q0[boundary.local_qDOF_P(frameID_boundary)],
                        frameID_boundary,
                    )
                    points.extend([r_OP])

    cells = [("vertex", [[i] for i in range(len(points))])]
    meshio.write_points_cells(filename, points, cells, binary=True)


def boundaries_clamped_clamped(
    model, pantographic_sheet, r_OP, A_IK, left=True, right=True
):
    W = pantographic_sheet.W
    L = pantographic_sheet.L
    H = pantographic_sheet.H
    n_row = pantographic_sheet.n_row
    n_col = pantographic_sheet.n_col
    n_fiber_layer = pantographic_sheet.n_fiber_layer

    r_OPL = np.array([0, W / 2, H / 2])
    frameL = Frame(r_OPL)
    model.add(frameL)

    if right:

        def r_OPR(t):
            return np.array([L, W / 2, H / 2]) + r_OP(t)

        frameR = Frame(r_OP=r_OPR, A_IK=A_IK)
        model.add(frameR)

    for layer in range(n_fiber_layer):
        for row in range(0, n_row):
            # constraints for all fibers on the left hand side
            col = 0

            # # even layer
            # if layer % 2 == 0:
            #     if row == 0 and col == n_col - 1: # bottom right
            #         continue
            #     if row == n_row - 1 and col == 0: # top left
            #         continue

            # # odd layer
            # else:
            #     if row == 0 and col == 0: # bottom left
            #         continue
            #     if row == n_row - 1 and col == n_col - 1: # top right
            #         continue

            if left:
                if (row + col) % 2 == 0:
                    fiber_left = pantographic_sheet.pivots_dict[row, col, layer][
                        "fiber"
                    ]
                    model.add(
                        RigidConnection(
                            frameL, fiber_left, frame_ID1=None, frame_ID2=(0.0,)
                        )
                    )

            # constraints for all fibers on the right hand side
            col = n_col - 1

            # # even layer
            # if layer % 2 == 0:
            #     if row == 0 and col == n_col - 1: # bottom right
            #         continue
            #     if row == n_row - 1 and col == 0: # top left
            #         continue

            # # odd layer
            # else:
            #     if row == 0 and col == 0: # bottom left
            #         continue
            #     if row == n_row - 1 and col == n_col - 1: # top right
            #         continue

            if right:
                if (row + col) % 2 == 0:
                    fiber_right = pantographic_sheet.pivots_dict[row, col, layer][
                        "fiber"
                    ]
                    model.add(
                        RigidConnection(
                            frameR, fiber_right, frame_ID1=None, frame_ID2=(1.0,)
                        )
                    )


def revolute_joint_indenter(model, pantographic_sheet, r_OP):
    W = pantographic_sheet.W
    L = pantographic_sheet.L
    H = pantographic_sheet.H
    n_row = pantographic_sheet.n_row
    n_col = pantographic_sheet.n_col
    n_fiber_layer = pantographic_sheet.n_fiber_layer

    assert n_row % 2 != 0
    assert n_col % 2 != 0

    col = int((n_col - 1) / 2)
    if col % 2 == 0:
        row = n_row - 1
    else:
        row = n_row - 2

    def r_OP0(t):
        return np.array([L / 2, W, H / 2]) + r_OP(t)

    A_IK0 = np.eye(3)
    frame = Frame(r_OP0, A_IK=A_IK0)
    model.add(frame)

    for layer in range(n_fiber_layer):
        pivot = pantographic_sheet.pivots_dict[row, col, layer]["pivot"]
        pivot_xi = pantographic_sheet.pivots_dict[row, col, layer]["pivot_xi"]

        frameID_pivot = (pivot_xi,)

        # position and orientation of the revolute joint
        r_OB = pivot.r_OP(
            model.t0,
            pivot.q0[pivot.local_qDOF_P(frameID_pivot)],
            frameID_pivot,
        )
        junction = RevoluteJoint(
            frame,
            pivot,
            r_OB,
            A_IK0,
            frame_ID1=None,
            frame_ID2=frameID_pivot,
        )

        pantographic_sheet.constraints.append(junction)
        model.add(junction)


def revolute_joints_support(model, pantographic_sheet):
    n_row = pantographic_sheet.n_row
    n_col = pantographic_sheet.n_col
    n_fiber_layer = pantographic_sheet.n_fiber_layer

    assert n_row % 2 != 0
    assert n_col % 2 != 0
    assert n_col >= 7

    frame = Frame()
    model.add(frame)

    row = 0
    for layer in range(n_fiber_layer):
        for col in [2, n_col - 3]:
            pivot = pantographic_sheet.pivots_dict[row, col, layer]["pivot"]
            pivot_xi = pantographic_sheet.pivots_dict[row, col, layer]["pivot_xi"]

            frameID_pivot = (pivot_xi,)

            r_OB0 = pivot.r_OP(
                model.t0,
                pivot.q0[pivot.local_qDOF_P(frameID_pivot)],
                frameID_pivot,
            )
            junction = RevoluteJoint(
                frame,
                pivot,
                r_OB0,
                np.eye(3),
                frame_ID1=None,
                frame_ID2=frameID_pivot,
            )

            pantographic_sheet.constraints.append(junction)
            model.add(junction)


def contact_indenter(model, pantographic_sheet, r_OP):
    n_row = pantographic_sheet.n_row
    n_col = pantographic_sheet.n_col
    n_fiber_layer = pantographic_sheet.n_fiber_layer

    assert n_row % 2 != 0
    assert n_col % 2 != 0

    middle_col = int((n_col - 1) / 2)
    row = n_row - 1
    if middle_col % 2 == 0:
        col_p = middle_col
        col_m = middle_col
    else:
        col_p = middle_col - 1
        col_m = middle_col + 1

    # n = -e_y^I
    A_IK0 = A_IK_basic(pi / 2).x()
    frame = Frame(r_OP, A_IK=A_IK0)
    model.add(frame)

    fiber = pantographic_sheet.pivots_dict[row, col_p, 0]["fiber"]
    frameID_fiber = (1,)
    r_OB = fiber.r_OP(
        model.t0,
        fiber.q0[fiber.local_qDOF_P(frameID_fiber)],
        frameID_fiber,
    )
    A_IB = np.eye(3)
    # joint = SphericalJoint(frame, fiber, r_OB, frame_ID2=frameID_fiber)
    joint = RevoluteJoint(frame, fiber, r_OB, A_IB, frame_ID2=frameID_fiber)
    model.add(joint)

    prox_r_N = 1.0e-1  # note: this is never used!
    for layer in range(2, n_fiber_layer, 2):
        fiber = pantographic_sheet.pivots_dict[row, col_p, layer]["fiber"]
        frameID_fiber = (1,)
        contact = Point2Plane(frame, fiber, prox_r_N, frame_ID=frameID_fiber)
        model.add(contact)

    for layer in range(1, n_fiber_layer, 2):
        fiber = pantographic_sheet.pivots_dict[row, col_m, layer]["fiber"]
        frameID_fiber = (0,)
        contact = Point2Plane(frame, fiber, prox_r_N, frame_ID=frameID_fiber)
        model.add(contact)


def contact_supports(model, pantographic_sheet):
    n_row = pantographic_sheet.n_row
    n_col = pantographic_sheet.n_col
    n_fiber_layer = pantographic_sheet.n_fiber_layer

    w_offset = pantographic_sheet.w * pantographic_sheet.fiber_factor
    r_OP = np.array([0.0, -w_offset, 0.0])

    # n = -e_y^I
    A_IK0 = A_IK_basic(-pi / 2).x()
    frame = Frame(r_OP, A_IK=A_IK0)
    model.add(frame)

    prox_r_N = 1.0e-1  # note: this is never used!

    # right contact
    row = 0
    for col in [2, n_col - 3]:
        for layer in range(0, n_fiber_layer, 2):
            fiber = pantographic_sheet.pivots_dict[row, col, layer]["fiber"]
            frameID_fiber = (0,)
            contact = Point2Plane(frame, fiber, prox_r_N, frame_ID=frameID_fiber)
            model.add(contact)

        for layer in range(1, n_fiber_layer, 2):
            fiber = pantographic_sheet.pivots_dict[row, col, layer]["fiber"]
            frameID_fiber = (1,)
            contact = Point2Plane(frame, fiber, prox_r_N, frame_ID=frameID_fiber)
            model.add(contact)


def contact_indenter_cylinder(model, pantographic_sheet, r_OP, radius):
    l = pantographic_sheet.l
    w = pantographic_sheet.w
    n_row = pantographic_sheet.n_row
    n_col = pantographic_sheet.n_col
    n_fiber_layer = pantographic_sheet.n_fiber_layer

    assert n_row % 2 != 0
    assert n_col % 2 != 0

    middle_col = int((n_col - 1) / 2)
    row = n_row - 1
    if middle_col % 2 == 0:
        col_p = middle_col
        col_m = middle_col
    else:
        col_p = middle_col - 1
        col_m = middle_col + 1

    # d3 = e_z^I
    A_IK0 = np.eye(3)
    frame = Frame(r_OP, A_IK=A_IK0)
    model.add(frame)

    fiber = pantographic_sheet.pivots_dict[row, col_p, 0]["fiber"]
    frameID_fiber = (1,)
    r_OB = fiber.r_OP(
        model.t0,
        fiber.q0[fiber.local_qDOF_P(frameID_fiber)],
        frameID_fiber,
    )
    A_IB = np.eye(3)
    # joint = SphericalJoint(frame, fiber, r_OB, frame_ID2=frameID_fiber)
    joint = RevoluteJoint(frame, fiber, r_OB, A_IB, frame_ID2=frameID_fiber)
    model.add(joint)

    prox_r_N = 1.0e-1  # note: this is never used!
    for layer in range(2, n_fiber_layer, 2):
        fiber = pantographic_sheet.pivots_dict[row, col_p, layer]["fiber"]
        frameID_fiber = (1,)
        contact = Point2Cylinder(frame, fiber, radius, prox_r_N, frame_ID=frameID_fiber)
        model.add(contact)

    for layer in range(1, n_fiber_layer, 2):
        fiber = pantographic_sheet.pivots_dict[row, col_m, layer]["fiber"]
        frameID_fiber = (0,)
        contact = Point2Cylinder(frame, fiber, radius, prox_r_N, frame_ID=frameID_fiber)
        model.add(contact)


if __name__ == "__main__":
    path = Path(__file__)
    path = path.parent / path.stem
    path.mkdir(parents=True, exist_ok=True)  # create directory

    statics = True
    # statics = False

    # initialize_with_statics = True
    initialize_with_statics = False

    # geometric parameters
    gamma = pi / 4  # angle between fiber families
    n_row0 = 5
    n_col0 = 15
    n_fiber_layer0 = 8

    # # original pantograph
    # n_row = 5
    # n_col = 15
    # n_fiber_layer = 8

    # # medium pantograph
    # n_row = 3
    # n_col = 9
    # n_fiber_layer = 2

    # small pantograph
    n_row = 2
    n_col = 2
    n_fiber_layer = 2

    pivot_factor = 1.0
    fiber_factor = 0.75

    # W0 = 40e-3
    # H0 = 70e-3
    W0 = 40  # [mm]
    H0 = 70  # [mm]
    w = W0 / n_row0
    h = H0 / (n_fiber_layer0 - 1)
    W = w * n_row
    H = h * (n_fiber_layer - 1)

    pantographic_sheet = PantographicSheet(
        W, H, gamma, n_row, n_col, n_fiber_layer, pivot_factor, fiber_factor, statics
    )
    model = System()
    create_pantographic_sheet(model, pantographic_sheet)

    # # add gravity
    # for beam in pantographic_sheet.beams:
    #     fg = -np.array([0, 0, 1.0e0])
    #     if statics:
    #         line_force = Line_force(
    #             lambda xi, t: t * fg, beam)
    #     else:
    #         line_force = Line_force(
    #             lambda xi, t: fg, beam)
    #     model.add(line_force)

    # # add clamped-clamped bundary condtions
    # # r_OP = lambda t: t * np.array([0.25 * pantographic_sheet.L, 0, 0])
    # r_OP = lambda t: np.zeros(3)
    # A_IK = lambda t: np.eye(3)

    # r_OP = lambda t: np.zeros(3)
    # A_IK = lambda t: A_IK_basic(t * pi / 2).x()

    # boundaries_clamped_clamped(model, pantographic_sheet, r_OP, A_IK)

    if indenter_support == "RevoluteJoint":
        #############################################
        # indenter using a sequence of RevoluteJoints
        #############################################
        r_OP = lambda t: t * np.array([0.0, -0.5 * pantographic_sheet.W, 0.0])
        revolute_joint_indenter(model, pantographic_sheet, r_OP)

        ###############################################
        # support using two sequences of RevoluteJoints
        ###############################################
        revolute_joints_support(model, pantographic_sheet)
    elif indenter_support == "PointPlane":
        ####################################################
        # indenter using a sequences of Point2Plane contacts
        ####################################################
        w_offset = pantographic_sheet.w * pantographic_sheet.fiber_factor
        r_OP = lambda t: np.array(
            [0.0, pantographic_sheet.W + w_offset, 0.0]
        ) + t * np.array([0.0, -w_offset - 0.8 * pantographic_sheet.W, 0.0])
        contact_indenter(model, pantographic_sheet, r_OP)

        #####################################################
        # support using two sequences of Point2Plane contacts
        #####################################################
        contact_supports(model, pantographic_sheet)
    elif indenter_support == "RevoluteJointPointPlane":
        #############################################
        # indenter using a sequence of RevoluteJoints
        #############################################
        r_OP = lambda t: t * np.array([0.0, -0.8 * pantographic_sheet.W, 0.0])
        revolute_joint_indenter(model, pantographic_sheet, r_OP)

        #####################################################
        # support using two sequences of Point2Plane contacts
        #####################################################
        contact_supports(model, pantographic_sheet)

    elif indenter_support == "PointCylinder":
        #############################################
        # indenter using a sequence of Point2Cylinder
        #############################################
        radius = 0.25 * pantographic_sheet.L
        w_offset = pantographic_sheet.w * pantographic_sheet.fiber_factor + radius
        r_OP = lambda t: np.array(
            [0.5 * pantographic_sheet.L, pantographic_sheet.W + w_offset, 0.0]
        ) + t * np.array([0.0, -w_offset - 0.5 * pantographic_sheet.W, 0.0])
        contact_indenter_cylinder(model, pantographic_sheet, r_OP, radius)

        #####################################################
        # support using two sequences of Point2Plane contacts
        #####################################################
        # TODO:
        contact_supports(model, pantographic_sheet)
    elif indenter_support == "FixedFixed":
        r_OP = lambda t: np.zeros(3)
        # phi_max = 0
        phi_max = 0.1 * pi
        # phi_max = pi
        A_IK = lambda t: A_IK_basic(t * phi_max).x()
        boundaries_clamped_clamped(model, pantographic_sheet, r_OP, A_IK)
    elif indenter_support == "FixedLeft":
        r_OP = lambda t: np.zeros(3)
        A_IK = lambda t: np.eye(3)
        boundaries_clamped_clamped(
            model, pantographic_sheet, r_OP, A_IK, left=True, right=False
        )
    elif indenter_support == "None":
        print(f"None support chosen!")
    else:
        raise NotImplementedError("")

    model.assemble()

    # # show initial configuration
    # animate_beam([model.t0], [model.q0], pantographic_sheet.beams, scale=H0)
    # exit()

    # # TODO: Debugging initial configuration!
    # # post_processing(pantographic_sheet.beams, [model.t0], [model.q0], f"pantograph")
    # # write_connecting_points("connecting_points_q0.vtu", pantographic_sheet)
    # write_xis_fibers("xis_fibers.vtu", pantographic_sheet)
    # write_xis_pivots("xis_pivots.vtu", pantographic_sheet)
    # write_xis_boundary("xis_boundaries.vtu", pantographic_sheet)

    # exit()

    if statics:
        n_load_steps = 20
        # n_load_steps = 10
        atol = 1.0e-6
        # atol = 1.0e-4
        sol = Newton(model, n_load_steps=n_load_steps, atol=atol).solve()

        # save static solution
        filename = path / (
            ("pantograph_revolute_" if use_revolute_joints else "pantograph_rigid_")
            + indenter_support
            + "_static_sol.npy"
        )
        np.save(filename, sol.q, allow_pickle=True)
    else:
        if initialize_with_statics:
            # TODO:
            # raise RuntimeWarning("This is hard coded for FixedFixed clamping")
            filename = path / (
                ("pantograph_revolute_" if use_revolute_joints else "pantograph_rigid_")
                + "FixedFixed_static_sol.npy"
            )
            q = np.load(filename, allow_pickle=True)
            q0 = q[-1]
            model.q0 = q0
            for contribution in model.contributions:
                contribution.q0 = model.q0[contribution.qDOF]
        # compute initial accelerations for dynamic DOF's
        M0 = model.M(model.t0, model.q0).tocsr()
        rhs0 = (
            model.h(model.t0, model.q0, model.u0)
            + model.W_g(model.t0, model.q0) @ model.la_g0
            + model.W_gamma(model.t0, model.q0) @ model.la_gamma0
        )
        a0 = spsolve(M0, rhs0)

        t1 = 500  # ms
        dt = 1e-1
        rho_inf = 0.85
        sol = GenAlphaDAEAcc(model, t1, dt, rho_inf, a0=a0).solve()

    filename = path / (
        ("pantograph_revolute_" if use_revolute_joints else "pantograph_rigid_")
        + indenter_support
    )

    animate_beam(sol.t, sol.q, pantographic_sheet.beams, scale=H0)

    # export_pivot_points(str(filename), pantographic_sheet, sol.q)

    # post_processing(pantographic_sheet.beams, sol.t, sol.q, str(filename))
