import numpy as np
from abc import ABC, abstractmethod
from cardillo.math import (
    pi,
    e1,
    norm,
    cross3,
    ax2skew,
    Exp_SO3,
    Log_SO3,
    T_SO3_inv,
    # tangent_map_s,
    approx_fprime,
)

from cardillo.discretization.bezier import L2_projection_Bezier_curve
from cardillo.beams._cross_section import (
    CrossSection,
    CircularCrossSection,
    RectangularCrossSection,
)

from cardillo.utility.coo import Coo
from cardillo.discretization.lagrange import LagrangeKnotVector
from cardillo.discretization.b_spline import BSplineKnotVector
from cardillo.discretization.hermite import HermiteNodeVector
from cardillo.discretization.mesh1D import Mesh1D


class RodExportBase(ABC):
    def __init__(self, cross_section: CrossSection):
        self.cross_section = cross_section

    @abstractmethod
    def r_OP(self, t, q, frame_ID, K_r_SP):
        ...

    @abstractmethod
    def A_IK(self, t, q, frame_ID):
        ...

    def frames(self, q, num=10):
        q_body = q[self.qDOF]
        r = []
        d1 = []
        d2 = []
        d3 = []

        for xi in np.linspace(0, 1, num=num):
            frame_ID = (xi,)
            qp = q_body[self.local_qDOF_P(frame_ID)]
            r.append(self.r_OP(1, qp, frame_ID))

            d1i, d2i, d3i = self.A_IK(1, qp, frame_ID).T
            d1.extend([d1i])
            d2.extend([d2i])
            d3.extend([d3i])

        return np.array(r).T, np.array(d1).T, np.array(d2).T, np.array(d3).T

    def export(self, sol_i, circle_as_wedge=True, **kwargs):
        q = sol_i.q

        level = kwargs["level"]

        if "num" in kwargs:
            num = kwargs["num"]
        else:
            num = self.nelement * 4

        r_OPs, d1s, d2s, d3s = self.frames(q, num=num)

        if level == "centerline + directors":
            #######################################
            # simple export of points and directors
            #######################################
            r_OPs, d1s, d2s, d3s = self.frames(q, num=num)

            vtk_points = r_OPs.T

            cells = [("line", [[i, i + 1] for i in range(num - 1)])]

            cell_data = {}

            point_data = {
                "d1": d1s.T,
                "d2": d2s.T,
                "d3": d3s.T,
            }

        elif level == "volume":
            assert isinstance(
                self.cross_section, (CircularCrossSection, RectangularCrossSection)
            ), "Volume export is only implemented for CircularCrossSection and RectangularCrossSection."

            ################################
            # project on cubic Bezier volume
            ################################
            if "n_segments" in kwargs:
                n_segments = kwargs["n_segments"]
            else:
                n_segments = self.nelement

            r_OPs, d1s, d2s, d3s = self.frames(q, num=num)
            target_points_centerline = r_OPs.T

            # create points of the target curves (three characteristic points
            # of the cross section)
            if isinstance(self.cross_section, CircularCrossSection):
                if circle_as_wedge:
                    ri = self.cross_section.radius
                    ru = 2 * ri
                    a = 2 * np.sqrt(3) * ri

                    target_points_0 = np.array(
                        [r_OP - ri * d3 for (r_OP, d3) in zip(r_OPs.T, d3s.T)]
                    )

                    target_points_1 = np.array(
                        [
                            r_OP + d2 * a / 2 - ri * d3
                            for (r_OP, d2, d3) in zip(r_OPs.T, d2s.T, d3s.T)
                        ]
                    )

                    target_points_2 = np.array(
                        [r_OP + d3 * ru for (r_OP, d3) in zip(r_OPs.T, d3s.T)]
                    )
                else:
                    target_points_0 = target_points_centerline
                    target_points_1 = np.array(
                        [
                            r_OP + d2 * self.cross_section.radius
                            for (r_OP, d2) in zip(r_OPs.T, d2s.T)
                        ]
                    )
                    target_points_2 = np.array(
                        [
                            r_OP + d3 * self.cross_section.radius
                            for (r_OP, d3) in zip(r_OPs.T, d3s.T)
                        ]
                    )
            elif isinstance(self.cross_section, RectangularCrossSection):
                target_points_0 = target_points_centerline
                target_points_1 = np.array(
                    [
                        r_OP + d2 * self.cross_section.width
                        for (r_OP, d2) in zip(r_OPs.T, d2s.T)
                    ]
                )
                target_points_2 = np.array(
                    [
                        r_OP + d3 * self.cross_section.height
                        for (r_OP, d3) in zip(r_OPs.T, d3s.T)
                    ]
                )
            else:
                raise NotImplementedError

            # project target points on cubic C1 Bézier curve
            _, _, points_segments_0 = L2_projection_Bezier_curve(
                target_points_0, n_segments, case="C1"
            )
            _, _, points_segments_1 = L2_projection_Bezier_curve(
                target_points_1, n_segments, case="C1"
            )
            _, _, points_segments_2 = L2_projection_Bezier_curve(
                target_points_2, n_segments, case="C1"
            )

            if isinstance(self.cross_section, CircularCrossSection):
                if circle_as_wedge:

                    def compute_missing_points(segment, layer):
                        P0 = points_segments_0[segment, layer]
                        P3 = points_segments_1[segment, layer]
                        P4 = points_segments_2[segment, layer]

                        P5 = 2 * P0 - P3
                        P1 = 0.5 * (P3 + P4)
                        P0 = 0.5 * (P5 + P3)
                        P2 = 0.5 * (P4 + P5)

                        dim = len(P0)
                        points_weights = np.zeros((6, dim + 1))
                        points_weights[0] = np.array([*P0, 1])
                        points_weights[1] = np.array([*P1, 1])
                        points_weights[2] = np.array([*P2, 1])
                        points_weights[3] = np.array([*P3, 1 / 2])
                        points_weights[4] = np.array([*P4, 1 / 2])
                        points_weights[5] = np.array([*P5, 1 / 2])

                        return points_weights

                    # create correct VTK ordering, see
                    # https://coreform.com/papers/implementation-of-rational-bezier-cells-into-VTK-report.pdf:
                    vtk_points_weights = []
                    for i in range(n_segments):
                        # compute all missing points of the layer
                        points_layer0 = compute_missing_points(i, 0)
                        points_layer1 = compute_missing_points(i, 1)
                        points_layer2 = compute_missing_points(i, 2)
                        points_layer3 = compute_missing_points(i, 3)

                        #######################
                        # 1. vertices (corners)
                        #######################

                        # bottom
                        for j in range(3):
                            vtk_points_weights.append(points_layer0[j])

                        # top
                        for j in range(3):
                            vtk_points_weights.append(points_layer3[j])

                        ##########
                        # 2. edges
                        ##########

                        # bottom
                        for j in range(3, 6):
                            vtk_points_weights.append(points_layer0[j])

                        # top
                        for j in range(3, 6):
                            vtk_points_weights.append(points_layer3[j])

                        # first and second
                        for j in range(3):
                            vtk_points_weights.append(points_layer1[j])
                            vtk_points_weights.append(points_layer2[j])

                        ##########
                        # 3. faces
                        ##########

                        # first and second
                        for j in range(3, 6):
                            vtk_points_weights.append(points_layer1[j])
                            vtk_points_weights.append(points_layer2[j])
                else:

                    def compute_missing_points(segment, layer):
                        P2 = points_segments_2[segment, layer]
                        P1 = points_segments_1[segment, layer]
                        P8 = points_segments_0[segment, layer]
                        P0 = 2 * P8 - P2
                        P3 = 2 * P8 - P1
                        P4 = (P0 + P1) - P8
                        P5 = (P1 + P2) - P8
                        P6 = (P2 + P3) - P8
                        P7 = (P3 + P0) - P8

                        dim = len(P0)
                        s22 = np.sqrt(2) / 2
                        points_weights = np.zeros((9, dim + 1))
                        points_weights[0] = np.array([*P0, 1])
                        points_weights[1] = np.array([*P1, 1])
                        points_weights[2] = np.array([*P2, 1])
                        points_weights[3] = np.array([*P3, 1])
                        points_weights[4] = np.array([*P4, s22])
                        points_weights[5] = np.array([*P5, s22])
                        points_weights[6] = np.array([*P6, s22])
                        points_weights[7] = np.array([*P7, s22])
                        points_weights[8] = np.array([*P8, 1])

                        return points_weights

                    # create correct VTK ordering, see
                    # https://coreform.com/papers/implementation-of-rational-bezier-cells-into-VTK-report.pdf:
                    vtk_points_weights = []
                    for i in range(n_segments):
                        # compute all missing points of the layer
                        points_layer0 = compute_missing_points(i, 0)
                        points_layer1 = compute_missing_points(i, 1)
                        points_layer2 = compute_missing_points(i, 2)
                        points_layer3 = compute_missing_points(i, 3)

                        #######################
                        # 1. vertices (corners)
                        #######################

                        # bottom
                        for j in range(4):
                            vtk_points_weights.append(points_layer0[j])

                        # top
                        for j in range(4):
                            vtk_points_weights.append(points_layer3[j])

                        ##########
                        # 2. edges
                        ##########

                        # bottom
                        for j in range(4, 8):
                            vtk_points_weights.append(points_layer0[j])

                        # top
                        for j in range(4, 8):
                            vtk_points_weights.append(points_layer3[j])

                        # first and second
                        for j in [0, 1, 3, 2]:
                            vtk_points_weights.append(points_layer1[j])
                            vtk_points_weights.append(points_layer2[j])

                        ##########
                        # 3. faces
                        ##########

                        # first and second
                        for j in [7, 5, 4, 6]:
                            vtk_points_weights.append(points_layer1[j])
                            vtk_points_weights.append(points_layer2[j])

                        # bottom and top
                        vtk_points_weights.append(points_layer0[0])
                        vtk_points_weights.append(points_layer3[-1])

                        ############
                        # 3. volumes
                        ############

                        # first and second
                        vtk_points_weights.append(points_layer1[0])
                        vtk_points_weights.append(points_layer2[-1])

            elif isinstance(self.cross_section, RectangularCrossSection):

                def compute_missing_points(segment, layer):
                    Q0 = points_segments_0[segment, layer]
                    Q1 = points_segments_1[segment, layer]
                    Q2 = points_segments_2[segment, layer]
                    P0 = Q0 - (Q2 - Q0) - (Q1 - Q0)
                    P1 = Q0 - (Q2 - Q0) + (Q1 - Q0)
                    P2 = Q0 + (Q2 - Q0) + (Q1 - Q0)
                    P3 = Q0 + (Q2 - Q0) - (Q1 - Q0)

                    dim = len(P0)
                    points_weights = np.zeros((4, dim + 1))
                    points_weights[0] = np.array([*P0, 1])
                    points_weights[1] = np.array([*P1, 1])
                    points_weights[2] = np.array([*P2, 1])
                    points_weights[3] = np.array([*P3, 1])

                    return points_weights

                # create correct VTK ordering, see
                # https://coreform.com/papers/implementation-of-rational-bezier-cells-into-VTK-report.pdf:
                vtk_points_weights = []
                for i in range(n_segments):
                    # compute all missing points of the layer
                    points_layer0 = compute_missing_points(i, 0)
                    points_layer1 = compute_missing_points(i, 1)
                    points_layer2 = compute_missing_points(i, 2)
                    points_layer3 = compute_missing_points(i, 3)

                    #######################
                    # 1. vertices (corners)
                    #######################

                    # bottom
                    for j in range(4):
                        vtk_points_weights.append(points_layer0[j])

                    # top
                    for j in range(4):
                        vtk_points_weights.append(points_layer3[j])

                    ##########
                    # 2. edges
                    ##########

                    # first and second
                    for j in [0, 1, 3, 2]:
                        vtk_points_weights.append(points_layer1[j])
                        vtk_points_weights.append(points_layer2[j])

            p_zeta = 3
            if isinstance(self.cross_section, CircularCrossSection):
                if circle_as_wedge:
                    n_layer = 6
                    n_cell = (p_zeta + 1) * n_layer

                    higher_order_degrees = [
                        (np.array([2, 2, p_zeta]),) for _ in range(n_segments)
                    ]

                    cells = [
                        (
                            "VTK_BEZIER_WEDGE",
                            np.arange(i * n_cell, (i + 1) * n_cell)[None],
                        )
                        for i in range(n_segments)
                    ]
                else:
                    n_layer = 9
                    n_cell = (p_zeta + 1) * n_layer

                    higher_order_degrees = [
                        (np.array([2, 2, p_zeta]),) for _ in range(n_segments)
                    ]

                    cells = [
                        (
                            "VTK_BEZIER_HEXAHEDRON",
                            np.arange(i * n_cell, (i + 1) * n_cell)[None],
                        )
                        for i in range(n_segments)
                    ]
            if isinstance(self.cross_section, RectangularCrossSection):
                n_layer = 4
                n_cell = (p_zeta + 1) * n_layer

                higher_order_degrees = [
                    (np.array([1, 1, p_zeta]),) for _ in range(n_segments)
                ]

                cells = [
                    (
                        "VTK_BEZIER_HEXAHEDRON",
                        np.arange(i * n_cell, (i + 1) * n_cell)[None],
                    )
                    for i in range(n_segments)
                ]

            vtk_points_weights = np.array(vtk_points_weights)
            vtk_points = vtk_points_weights[:, :3]

            point_data = {
                "RationalWeights": vtk_points_weights[:, 3],
            }

            cell_data = {
                "HigherOrderDegrees": higher_order_degrees,
            }

        else:
            raise NotImplementedError

        return vtk_points, cells, point_data, cell_data


class TimoshenkoPetrovGalerkinBase(RodExportBase, ABC):
    def __init__(
        self,
        cross_section,
        material_model,
        A_rho0,
        K_S_rho0,
        K_I_rho0,
        polynomial_degree_r,
        polynomial_degree_psi,
        nelement,
        Q,
        q0=None,
        u0=None,
        basis_r="Lagrange",
        basis_psi="Lagrange",
    ):
        """Base class for Petrov-Galerkin spatial Timoshenko rod formulations.

        Up to now we restrict oursleves to rotations vector parametrization,
        but quanternions can added without too much work."""

        super().__init__(cross_section)

        # beam properties
        self.material_model = material_model  # material model
        self.A_rho0 = A_rho0
        self.K_S_rho0 = K_S_rho0
        self.K_I_rho0 = K_I_rho0

        # can we use a constant mass matrix
        if np.allclose(K_S_rho0, np.zeros_like(K_S_rho0)):
            self.constant_mass_matrix = True
        else:
            self.constant_mass_matrix = False

        # discretization parameters
        self.polynomial_degree_r = polynomial_degree_r
        self.polynomial_degree_psi = polynomial_degree_psi

        # TODO: Distinguish between mass matrix quadrature and other quaratures!
        p = max(polynomial_degree_r, polynomial_degree_psi)
        # self.nquadrature = nquadrature = int(np.ceil((p + 1) ** 2 / 2))
        self.nquadrature = nquadrature = p
        self.nelement = nelement

        # chose basis functions
        self.basis_r = basis_r
        self.basis_psi = basis_psi

        if basis_r == "Lagrange":
            self.knot_vector_r = LagrangeKnotVector(polynomial_degree_r, nelement)
        elif basis_r == "B-spline":
            self.knot_vector_r = BSplineKnotVector(polynomial_degree_r, nelement)
        elif basis_r == "Hermite":
            assert (
                polynomial_degree_r == 3
            ), "only cubic Hermite splines are implemented!"
            self.knot_vector_r = HermiteNodeVector(polynomial_degree_r, nelement)
        else:
            raise RuntimeError(f'wrong basis_r: "{basis_r}" was chosen')

        if basis_psi == "Lagrange":
            self.knot_vector_psi = LagrangeKnotVector(polynomial_degree_psi, nelement)
        elif basis_psi == "B-spline":
            self.knot_vector_psi = BSplineKnotVector(polynomial_degree_psi, nelement)
        elif basis_psi == "Hermite":
            assert (
                polynomial_degree_psi == 3
            ), "only cubic Hermite splines are implemented!"
            self.knot_vector_psi = HermiteNodeVector(polynomial_degree_psi, nelement)
        else:
            raise RuntimeError(f'wrong basis_psi: "{basis_psi}" was chosen')

        # build mesh objects
        self.mesh_r = Mesh1D(
            self.knot_vector_r,
            nquadrature,
            dim_q=3,
            derivative_order=1,
            basis=basis_r,
            quadrature="Gauss",
        )

        self.mesh_psi = Mesh1D(
            self.knot_vector_psi,
            nquadrature,
            dim_q=3,
            derivative_order=1,
            basis=basis_psi,
            quadrature="Gauss",
        )

        # total number of nodes
        self.nnodes_r = self.mesh_r.nnodes
        self.nnodes_psi = self.mesh_psi.nnodes

        # number of nodes per element
        self.nnodes_element_r = self.mesh_r.nnodes_per_element
        self.nnodes_element_psi = self.mesh_psi.nnodes_per_element

        # total number of generalized coordinates and velocities
        self.nq_r = self.nu_r = self.mesh_r.nq
        self.nq_psi = self.nu_psi = self.mesh_psi.nq
        self.nq = self.nu = self.nq_r + self.nq_psi

        # number of generalized coordiantes and velocities per element
        self.nq_element_r = self.nu_element_r = self.mesh_r.nq_per_element
        self.nq_element_psi = self.nu_element_psi = self.mesh_psi.nq_per_element
        self.nq_element = self.nu_element = self.nq_element_r + self.nq_element_psi

        # global element connectivity
        self.elDOF_r = self.mesh_r.elDOF
        self.elDOF_psi = self.mesh_psi.elDOF + self.nq_r
        # qe = q[elDOF[e]] "q^e = C_e,q q"

        # global nodal
        self.nodalDOF_r = self.mesh_r.nodalDOF
        self.nodalDOF_psi = self.mesh_psi.nodalDOF + self.nq_r

        # nodal connectivity on element level
        self.nodalDOF_element_r = self.mesh_r.nodalDOF_element
        self.nodalDOF_element_psi = self.mesh_psi.nodalDOF_element + self.nq_element_r
        # r_OP_i^e = C_r,i^e * C_e,q q = C_r,i^e * q^e
        # r_OPi = qe[nodelDOF_element_r[i]]

        # build global elDOF connectivity matrix
        self.elDOF = np.zeros((nelement, self.nq_element), dtype=int)
        for el in range(nelement):
            self.elDOF[el, : self.nq_element_r] = self.elDOF_r[el]
            self.elDOF[el, self.nq_element_r :] = self.elDOF_psi[el]

        # shape functions and their first derivatives
        self.N_r = self.mesh_r.N
        self.N_r_xi = self.mesh_r.N_xi
        self.N_psi = self.mesh_psi.N
        self.N_psi_xi = self.mesh_psi.N_xi

        # quadrature points
        self.qp = self.mesh_r.qp  # quadrature points
        self.qw = self.mesh_r.wp  # quadrature weights

        # reference generalized coordinates, initial coordinates and initial velocities
        self.Q = Q
        self.q0 = Q.copy() if q0 is None else q0
        self.u0 = np.zeros(self.nu, dtype=float) if u0 is None else u0

        # evaluate shape functions at specific xi
        self.basis_functions_r = self.mesh_r.eval_basis
        self.basis_functions_psi = self.mesh_psi.eval_basis

        # reference generalized coordinates, initial coordinates and initial velocities
        self.Q = Q  # reference configuration
        self.q0 = Q.copy() if q0 is None else q0  # initial configuration
        self.u0 = (
            np.zeros(self.nu, dtype=float) if u0 is None else u0
        )  # initial velocities

        # precompute values of the reference configuration in order to save computation time
        # J in Harsch2020b (5)
        self.J = np.zeros((nelement, nquadrature), dtype=float)
        # dilatation and shear strains of the reference configuration
        self.K_Gamma0 = np.zeros((nelement, nquadrature, 3), dtype=float)
        # curvature of the reference configuration
        self.K_Kappa0 = np.zeros((nelement, nquadrature, 3), dtype=float)

        for el in range(nelement):
            qe = self.Q[self.elDOF[el]]

            for i in range(nquadrature):
                # current quadrature point
                qpi = self.qp[el, i]

                # evaluate required quantities
                _, _, K_Gamma_bar, K_Kappa_bar = self._eval(qe, qpi)

                # length of reference tangential vector
                J = norm(K_Gamma_bar)

                # axial and shear strains
                K_Gamma = K_Gamma_bar / J

                # torsional and flexural strains
                K_Kappa = K_Kappa_bar / J

                # safe precomputed quantities for later
                self.J[el, i] = J
                self.K_Gamma0[el, i] = K_Gamma
                self.K_Kappa0[el, i] = K_Kappa

    def frames(self, q, num=10):
        q_body = q[self.qDOF]
        r = []
        d1 = []
        d2 = []
        d3 = []

        for xi in np.linspace(0, 1, num=num):
            frame_ID = (xi,)
            qp = q_body[self.local_qDOF_P(frame_ID)]
            r.append(self.r_OP(1, qp, frame_ID))

            d1i, d2i, d3i = self.A_IK(1, qp, frame_ID).T
            d1.extend([d1i])
            d2.extend([d2i])
            d3.extend([d3i])

        return np.array(r).T, np.array(d1).T, np.array(d2).T, np.array(d3).T

    def export(self, sol_i, circle_as_wedge=True, **kwargs):
        q = sol_i.q

        level = kwargs["level"]

        if "num" in kwargs:
            num = kwargs["num"]
        else:
            num = self.nelement * 4

        r_OPs, d1s, d2s, d3s = self.frames(q, num=num)

        if level == "centerline + directors":
            #######################################
            # simple export of points and directors
            #######################################
            r_OPs, d1s, d2s, d3s = self.frames(q, num=num)

            vtk_points = r_OPs.T

            cells = [("line", [[i, i + 1] for i in range(num - 1)])]

            cell_data = {}

            point_data = {
                "d1": d1s.T,
                "d2": d2s.T,
                "d3": d3s.T,
            }

        elif level == "volume":
            assert isinstance(
                self.cross_section, (CircularCrossSection, RectangularCrossSection)
            ), "Volume export is only implemented for CircularCrossSection and RectangularCrossSection."

            ################################
            # project on cubic Bezier volume
            ################################
            if "n_segments" in kwargs:
                n_segments = kwargs["n_segments"]
            else:
                n_segments = self.nelement

            r_OPs, d1s, d2s, d3s = self.frames(q, num=num)
            target_points_centerline = r_OPs.T

            # create points of the target curves (three characteristic points
            # of the cross section)
            if isinstance(self.cross_section, CircularCrossSection):
                if circle_as_wedge:
                    ri = self.cross_section.radius
                    ru = 2 * ri
                    a = 2 * np.sqrt(3) * ri

                    target_points_0 = np.array(
                        [r_OP - ri * d3 for (r_OP, d3) in zip(r_OPs.T, d3s.T)]
                    )

                    target_points_1 = np.array(
                        [
                            r_OP + d2 * a / 2 - ri * d3
                            for (r_OP, d2, d3) in zip(r_OPs.T, d2s.T, d3s.T)
                        ]
                    )

                    target_points_2 = np.array(
                        [r_OP + d3 * ru for (r_OP, d3) in zip(r_OPs.T, d3s.T)]
                    )
                else:
                    target_points_0 = target_points_centerline
                    target_points_1 = np.array(
                        [
                            r_OP + d2 * self.cross_section.radius
                            for (r_OP, d2) in zip(r_OPs.T, d2s.T)
                        ]
                    )
                    target_points_2 = np.array(
                        [
                            r_OP + d3 * self.cross_section.radius
                            for (r_OP, d3) in zip(r_OPs.T, d3s.T)
                        ]
                    )
            elif isinstance(self.cross_section, RectangularCrossSection):
                target_points_0 = target_points_centerline
                target_points_1 = np.array(
                    [
                        r_OP + d2 * self.cross_section.width
                        for (r_OP, d2) in zip(r_OPs.T, d2s.T)
                    ]
                )
                target_points_2 = np.array(
                    [
                        r_OP + d3 * self.cross_section.height
                        for (r_OP, d3) in zip(r_OPs.T, d3s.T)
                    ]
                )
            else:
                raise NotImplementedError

            # project target points on cubic C1 Bézier curve
            _, _, points_segments_0 = L2_projection_Bezier_curve(
                target_points_0, n_segments, case="C1"
            )
            _, _, points_segments_1 = L2_projection_Bezier_curve(
                target_points_1, n_segments, case="C1"
            )
            _, _, points_segments_2 = L2_projection_Bezier_curve(
                target_points_2, n_segments, case="C1"
            )

            if isinstance(self.cross_section, CircularCrossSection):
                if circle_as_wedge:

                    def compute_missing_points(segment, layer):
                        P0 = points_segments_0[segment, layer]
                        P3 = points_segments_1[segment, layer]
                        P4 = points_segments_2[segment, layer]

                        P5 = 2 * P0 - P3
                        P1 = 0.5 * (P3 + P4)
                        P0 = 0.5 * (P5 + P3)
                        P2 = 0.5 * (P4 + P5)

                        dim = len(P0)
                        points_weights = np.zeros((6, dim + 1))
                        points_weights[0] = np.array([*P0, 1])
                        points_weights[1] = np.array([*P1, 1])
                        points_weights[2] = np.array([*P2, 1])
                        points_weights[3] = np.array([*P3, 1 / 2])
                        points_weights[4] = np.array([*P4, 1 / 2])
                        points_weights[5] = np.array([*P5, 1 / 2])

                        return points_weights

                    # create correct VTK ordering, see
                    # https://coreform.com/papers/implementation-of-rational-bezier-cells-into-VTK-report.pdf:
                    vtk_points_weights = []
                    for i in range(n_segments):
                        # compute all missing points of the layer
                        points_layer0 = compute_missing_points(i, 0)
                        points_layer1 = compute_missing_points(i, 1)
                        points_layer2 = compute_missing_points(i, 2)
                        points_layer3 = compute_missing_points(i, 3)

                        #######################
                        # 1. vertices (corners)
                        #######################

                        # bottom
                        for j in range(3):
                            vtk_points_weights.append(points_layer0[j])

                        # top
                        for j in range(3):
                            vtk_points_weights.append(points_layer3[j])

                        ##########
                        # 2. edges
                        ##########

                        # bottom
                        for j in range(3, 6):
                            vtk_points_weights.append(points_layer0[j])

                        # top
                        for j in range(3, 6):
                            vtk_points_weights.append(points_layer3[j])

                        # first and second
                        for j in range(3):
                            vtk_points_weights.append(points_layer1[j])
                            vtk_points_weights.append(points_layer2[j])

                        ##########
                        # 3. faces
                        ##########

                        # first and second
                        for j in range(3, 6):
                            vtk_points_weights.append(points_layer1[j])
                            vtk_points_weights.append(points_layer2[j])
                else:

                    def compute_missing_points(segment, layer):
                        P2 = points_segments_2[segment, layer]
                        P1 = points_segments_1[segment, layer]
                        P8 = points_segments_0[segment, layer]
                        P0 = 2 * P8 - P2
                        P3 = 2 * P8 - P1
                        P4 = (P0 + P1) - P8
                        P5 = (P1 + P2) - P8
                        P6 = (P2 + P3) - P8
                        P7 = (P3 + P0) - P8

                        dim = len(P0)
                        s22 = np.sqrt(2) / 2
                        points_weights = np.zeros((9, dim + 1))
                        points_weights[0] = np.array([*P0, 1])
                        points_weights[1] = np.array([*P1, 1])
                        points_weights[2] = np.array([*P2, 1])
                        points_weights[3] = np.array([*P3, 1])
                        points_weights[4] = np.array([*P4, s22])
                        points_weights[5] = np.array([*P5, s22])
                        points_weights[6] = np.array([*P6, s22])
                        points_weights[7] = np.array([*P7, s22])
                        points_weights[8] = np.array([*P8, 1])

                        return points_weights

                    # create correct VTK ordering, see
                    # https://coreform.com/papers/implementation-of-rational-bezier-cells-into-VTK-report.pdf:
                    vtk_points_weights = []
                    for i in range(n_segments):
                        # compute all missing points of the layer
                        points_layer0 = compute_missing_points(i, 0)
                        points_layer1 = compute_missing_points(i, 1)
                        points_layer2 = compute_missing_points(i, 2)
                        points_layer3 = compute_missing_points(i, 3)

                        #######################
                        # 1. vertices (corners)
                        #######################

                        # bottom
                        for j in range(4):
                            vtk_points_weights.append(points_layer0[j])

                        # top
                        for j in range(4):
                            vtk_points_weights.append(points_layer3[j])

                        ##########
                        # 2. edges
                        ##########

                        # bottom
                        for j in range(4, 8):
                            vtk_points_weights.append(points_layer0[j])

                        # top
                        for j in range(4, 8):
                            vtk_points_weights.append(points_layer3[j])

                        # first and second
                        for j in [0, 1, 3, 2]:
                            vtk_points_weights.append(points_layer1[j])
                            vtk_points_weights.append(points_layer2[j])

                        ##########
                        # 3. faces
                        ##########

                        # first and second
                        for j in [7, 5, 4, 6]:
                            vtk_points_weights.append(points_layer1[j])
                            vtk_points_weights.append(points_layer2[j])

                        # bottom and top
                        vtk_points_weights.append(points_layer0[0])
                        vtk_points_weights.append(points_layer3[-1])

                        ############
                        # 3. volumes
                        ############

                        # first and second
                        vtk_points_weights.append(points_layer1[0])
                        vtk_points_weights.append(points_layer2[-1])

            elif isinstance(self.cross_section, RectangularCrossSection):

                def compute_missing_points(segment, layer):
                    Q0 = points_segments_0[segment, layer]
                    Q1 = points_segments_1[segment, layer]
                    Q2 = points_segments_2[segment, layer]
                    P0 = Q0 - (Q2 - Q0) - (Q1 - Q0)
                    P1 = Q0 - (Q2 - Q0) + (Q1 - Q0)
                    P2 = Q0 + (Q2 - Q0) + (Q1 - Q0)
                    P3 = Q0 + (Q2 - Q0) - (Q1 - Q0)

                    dim = len(P0)
                    points_weights = np.zeros((4, dim + 1))
                    points_weights[0] = np.array([*P0, 1])
                    points_weights[1] = np.array([*P1, 1])
                    points_weights[2] = np.array([*P2, 1])
                    points_weights[3] = np.array([*P3, 1])

                    return points_weights

                # create correct VTK ordering, see
                # https://coreform.com/papers/implementation-of-rational-bezier-cells-into-VTK-report.pdf:
                vtk_points_weights = []
                for i in range(n_segments):
                    # compute all missing points of the layer
                    points_layer0 = compute_missing_points(i, 0)
                    points_layer1 = compute_missing_points(i, 1)
                    points_layer2 = compute_missing_points(i, 2)
                    points_layer3 = compute_missing_points(i, 3)

                    #######################
                    # 1. vertices (corners)
                    #######################

                    # bottom
                    for j in range(4):
                        vtk_points_weights.append(points_layer0[j])

                    # top
                    for j in range(4):
                        vtk_points_weights.append(points_layer3[j])

                    ##########
                    # 2. edges
                    ##########

                    # first and second
                    for j in [0, 1, 3, 2]:
                        vtk_points_weights.append(points_layer1[j])
                        vtk_points_weights.append(points_layer2[j])

            p_zeta = 3
            if isinstance(self.cross_section, CircularCrossSection):
                if circle_as_wedge:
                    n_layer = 6
                    n_cell = (p_zeta + 1) * n_layer

                    higher_order_degrees = [
                        (np.array([2, 2, p_zeta]),) for _ in range(n_segments)
                    ]

                    cells = [
                        (
                            "VTK_BEZIER_WEDGE",
                            np.arange(i * n_cell, (i + 1) * n_cell)[None],
                        )
                        for i in range(n_segments)
                    ]
                else:
                    n_layer = 9
                    n_cell = (p_zeta + 1) * n_layer

                    higher_order_degrees = [
                        (np.array([2, 2, p_zeta]),) for _ in range(n_segments)
                    ]

                    cells = [
                        (
                            "VTK_BEZIER_HEXAHEDRON",
                            np.arange(i * n_cell, (i + 1) * n_cell)[None],
                        )
                        for i in range(n_segments)
                    ]
            if isinstance(self.cross_section, RectangularCrossSection):
                n_layer = 4
                n_cell = (p_zeta + 1) * n_layer

                higher_order_degrees = [
                    (np.array([1, 1, p_zeta]),) for _ in range(n_segments)
                ]

                cells = [
                    (
                        "VTK_BEZIER_HEXAHEDRON",
                        np.arange(i * n_cell, (i + 1) * n_cell)[None],
                    )
                    for i in range(n_segments)
                ]

            vtk_points_weights = np.array(vtk_points_weights)
            vtk_points = vtk_points_weights[:, :3]

            point_data = {
                "RationalWeights": vtk_points_weights[:, 3],
            }

            cell_data = {
                "HigherOrderDegrees": higher_order_degrees,
            }

        else:
            raise NotImplementedError

        return vtk_points, cells, point_data, cell_data

    @staticmethod
    def straight_configuration(
        polynomial_degree_r,
        polynomial_degree_psi,
        basis_r,
        basis_psi,
        nelement,
        L,
        r_OP=np.zeros(3, dtype=float),
        A_IK=np.eye(3, dtype=float),
    ):
        if basis_r == "Lagrange":
            nnodes_r = polynomial_degree_r * nelement + 1
        elif basis_r == "B-spline":
            nnodes_r = polynomial_degree_r + nelement
        elif basis_r == "Hermite":
            nnodes_r = nelement + 1
        else:
            raise RuntimeError(f'wrong basis_r: "{basis_r}" was chosen')

        if basis_psi == "Lagrange":
            nnodes_psi = polynomial_degree_psi * nelement + 1
        elif basis_psi == "B-spline":
            nnodes_psi = polynomial_degree_psi + nelement
        elif basis_psi == "Hermite":
            nnodes_psi = nelement + 1
        else:
            raise RuntimeError(f'wrong basis_psi: "{basis_psi}" was chosen')

        if basis_r == "B-spline" or basis_r == "Lagrange":
            x0 = np.linspace(0, L, num=nnodes_r)
            y0 = np.zeros(nnodes_r)
            z0 = np.zeros(nnodes_r)
            if basis_r == "B-spline":
                # build Greville abscissae for B-spline basis
                kv = BSplineKnotVector.uniform(polynomial_degree_r, nelement)
                for i in range(nnodes_r):
                    x0[i] = np.sum(kv[i + 1 : i + polynomial_degree_r + 1])
                x0 = x0 * L / polynomial_degree_r

            r0 = np.vstack((x0, y0, z0))
            for i in range(nnodes_r):
                r0[:, i] = r_OP + A_IK @ r0[:, i]

        elif basis_r == "Hermite":
            xis = np.linspace(0, 1, num=nnodes_r)
            r0 = np.zeros((6, nnodes_r))
            t0 = A_IK @ (L * e1)
            for i, xi in enumerate(xis):
                ri = r_OP + xi * t0
                r0[:3, i] = ri
                r0[3:, i] = t0

        # reshape generalized coordinates to nodal ordering
        q_r = r0.reshape(-1, order="C")

        # we have to extract the rotation vector from the given rotation matrix
        # and set its value for each node
        if basis_psi == "Hermite":
            raise NotImplementedError
        psi = Log_SO3(A_IK)
        q_psi = np.repeat(psi, nnodes_psi)

        return np.concatenate([q_r, q_psi])

    def element_number(self, xi):
        """Compute element number from given xi."""
        return self.knot_vector_r.element_number(xi)[0]

    ##################
    # abstract methods
    ##################
    @abstractmethod
    def _eval(self, qe, xi):
        """Compute (r_OP, A_IK, K_Gamma_bar, K_Kappa_bar)."""
        ...

    # @abstractmethod
    # def _deval(self, qe, xi):
    #     """Compute
    #         * r_OP
    #         * A_IK
    #         * K_Gamma_bar
    #         * K_Kappa_bar
    #         * r_OP_qe
    #         * A_IK_qe
    #         * K_Gamma_bar_qe
    #         * K_Kappa_bar_qe
    #     """
    #     ...

    @abstractmethod
    def A_IK(self, t, q, frame_ID):
        ...

    @abstractmethod
    def A_IK_q(self, t, q, frame_ID):
        ...

    #########################################
    # kinematic equation
    #########################################
    def q_dot(self, t, q, u):
        # centerline part
        q_dot = u.copy()

        # correct axis angle vector part
        for node in range(self.nnodes_psi):
            nodalDOF_psi = self.nodalDOF_psi[node]
            psi = q[nodalDOF_psi]
            K_omega_IK = u[nodalDOF_psi]

            psi_dot = T_SO3_inv(psi) @ K_omega_IK
            q_dot[nodalDOF_psi] = psi_dot

        return q_dot

    def B(self, t, q, coo):
        # trivial kinematic equation for centerline
        coo.extend_diag(
            np.ones(self.nq_r), (self.qDOF[: self.nq_r], self.uDOF[: self.nq_r])
        )

        # axis angle vector part
        for node in range(self.nnodes_psi):
            nodalDOF_psi = self.nodalDOF_psi[node]

            psi = q[nodalDOF_psi]
            coo.extend(
                T_SO3_inv(psi),
                (self.qDOF[nodalDOF_psi], self.uDOF[nodalDOF_psi]),
            )

    def q_ddot(self, t, q, u, u_dot):
        # centerline part
        q_ddot = u_dot

        # correct axis angle vector part
        for node in range(self.nnodes_psi):
            nodalDOF_psi = self.nodalDOF_psi[node]

            psi = q[nodalDOF_psi]
            K_omega_IK = u[nodalDOF_psi]
            K_omega_IK_dot = u_dot[nodalDOF_psi]

            T_inv = T_SO3_inv(psi)
            psi_dot = T_inv @ K_omega_IK

            # TODO:
            T_dot = tangent_map_s(psi, psi_dot)
            Tinv_dot = -T_inv @ T_dot @ T_inv
            psi_ddot = T_inv @ K_omega_IK_dot + Tinv_dot @ K_omega_IK

            # psi_ddot = (
            #     T_inv @ K_omega_IK_dot
            #     + np.einsum("ijk,j,k",
            #         approx_fprime(psi, T_SO3_inv, eps=1.0e-10, method="cs"),
            #         K_omega_IK,
            #         psi_dot
            #     )
            # )

            q_ddot[nodalDOF_psi] = psi_ddot

        return q_ddot

    # change between rotation vector and its complement in order to circumvent
    # singularities of the rotation vector
    def psi_C(self, psi):
        angle = norm(psi)
        if angle < pi:
            return psi
        else:
            # Ibrahimbegovic1995 after (62)
            psi_C = (1.0 - 2.0 * pi / angle) * psi
            return psi_C

    def step_callback(self, t, q, u):
        for node in range(self.nnodes_psi):
            psi = q[self.nodalDOF_psi[node]]
            q[self.nodalDOF_psi[node]] = self.psi_C(psi)
        return q, u

    ###############################
    # potential and internal forces
    ###############################
    def E_pot(self, t, q):
        E_pot = 0
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            E_pot += self.E_pot_el(q[elDOF], el)
        return E_pot

    def E_pot_el(self, qe, el):
        E_pot_el = 0.0

        for i in range(self.nquadrature):
            # extract reference state variables
            qpi = self.qp[el, i]
            qwi = self.qw[el, i]
            J = self.J[el, i]
            K_Gamma0 = self.K_Gamma0[el, i]
            K_Kappa0 = self.K_Kappa0[el, i]

            # evaluate required quantities
            _, _, K_Gamma_bar, K_Kappa_bar = self._eval(qe, qpi)

            # axial and shear strains
            K_Gamma = K_Gamma_bar / J

            # torsional and flexural strains
            K_Kappa = K_Kappa_bar / J

            # evaluate strain energy function
            E_pot_el += (
                self.material_model.potential(K_Gamma, K_Gamma0, K_Kappa, K_Kappa0)
                * J
                * qwi
            )

        return E_pot_el

    def f_pot_el(self, qe, el):
        f_pot_el = np.zeros(self.nq_element, dtype=qe.dtype)

        for i in range(self.nquadrature):
            # extract reference state variables
            qpi = self.qp[el, i]
            qwi = self.qw[el, i]
            J = self.J[el, i]
            K_Gamma0 = self.K_Gamma0[el, i]
            K_Kappa0 = self.K_Kappa0[el, i]

            # evaluate required quantities
            _, A_IK, K_Gamma_bar, K_Kappa_bar = self._eval(qe, qpi)

            # axial and shear strains
            K_Gamma = K_Gamma_bar / J

            # torsional and flexural strains
            K_Kappa = K_Kappa_bar / J

            # compute contact forces and couples from partial derivatives of
            # the strain energy function w.r.t. strain measures
            K_n = self.material_model.K_n(K_Gamma, K_Gamma0, K_Kappa, K_Kappa0)
            K_m = self.material_model.K_m(K_Gamma, K_Gamma0, K_Kappa, K_Kappa0)

            ############################
            # virtual work contributions
            ############################
            for node in range(self.nnodes_element_r):
                f_pot_el[self.nodalDOF_element_r[node]] -= (
                    self.N_r_xi[el, i, node] * A_IK @ K_n * qwi
                )

            for node in range(self.nnodes_element_psi):
                f_pot_el[self.nodalDOF_element_psi[node]] -= (
                    self.N_psi_xi[el, i, node] * K_m * qwi
                )

                f_pot_el[self.nodalDOF_element_psi[node]] += (
                    self.N_psi[el, i, node]
                    * (cross3(K_Gamma_bar, K_n) + cross3(K_Kappa_bar, K_m))
                    * qwi
                )

        return f_pot_el

        # f_pot_el_num = approx_fprime(
        #     qe, lambda qe: -self.E_pot_el(qe, el), eps=1.0e-10, method="cs"
        #     # qe, lambda qe: -self.E_pot_el(qe, el), eps=1.0e-6, method="3-point"
        # )
        # diff = f_pot_el - f_pot_el_num
        # error = np.linalg.norm(diff)
        # print(f"error: {error}")
        # # return f_pot_el
        # return f_pot_el_num

    # TODO:
    def f_pot_el_q(self, qe, el):
        return approx_fprime(
            # qe, lambda qe: self.f_pot_el(qe, el), eps=1.0e-6, method="3-point"
            qe,
            lambda qe: self.f_pot_el(qe, el),
            eps=1.0e-10,
            method="cs",
        )

        f_pot_q_el = np.zeros((self.nq_element, self.nq_element), dtype=float)

        for i in range(self.nquadrature):
            # extract reference state variables
            qpi = self.qp[el, i]
            qwi = self.qw[el, i]
            J = self.J[el, i]
            K_Gamma0 = self.K_Gamma0[el, i]
            K_Kappa0 = self.K_Kappa0[el, i]

            # evaluate required quantities
            (
                r_OP,
                A_IK,
                K_Gamma_bar,
                K_Kappa_bar,
                r_OP_qe,
                A_IK_qe,
                K_Gamma_bar_qe,
                K_Kappa_bar_qe,
            ) = self._deval(qe, qpi)

            # axial and shear strains
            K_Gamma = K_Gamma_bar / J
            K_Gamma_qe = K_Gamma_bar_qe / J

            # torsional and flexural strains
            K_Kappa = K_Kappa_bar / J
            K_Kappa_qe = K_Kappa_bar_qe / J

            # compute contact forces and couples from partial derivatives of
            # the strain energy function w.r.t. strain measures
            K_n = self.material_model.K_n(K_Gamma, K_Gamma0, K_Kappa, K_Kappa0)
            K_n_K_Gamma = self.material_model.K_n_K_Gamma(
                K_Gamma, K_Gamma0, K_Kappa, K_Kappa0
            )
            K_n_K_Kappa = self.material_model.K_n_K_Kappa(
                K_Gamma, K_Gamma0, K_Kappa, K_Kappa0
            )
            K_n_qe = K_n_K_Gamma @ K_Gamma_qe + K_n_K_Kappa @ K_Kappa_qe

            K_m = self.material_model.K_m(K_Gamma, K_Gamma0, K_Kappa, K_Kappa0)
            K_m_K_Gamma = self.material_model.K_m_K_Gamma(
                K_Gamma, K_Gamma0, K_Kappa, K_Kappa0
            )
            K_m_K_Kappa = self.material_model.K_m_K_Kappa(
                K_Gamma, K_Gamma0, K_Kappa, K_Kappa0
            )
            K_m_qe = K_m_K_Gamma @ K_Gamma_qe + K_m_K_Kappa @ K_Kappa_qe

            ############################
            # virtual work contributions
            ############################
            for node in range(self.nnodes_element_r):
                f_pot_q_el[self.nodalDOF_element_r[node], :] -= (
                    self.N_r_xi[el, i, node]
                    * qwi
                    * (np.einsum("ikj,k->ij", A_IK_qe, K_n) + A_IK @ K_n_qe)
                )

            for node in range(self.nnodes_element_psi):
                f_pot_q_el[self.nodalDOF_element_psi[node], :] += (
                    self.N_psi[el, i, node]
                    * qwi
                    * (ax2skew(K_Gamma_bar) @ K_n_qe - ax2skew(K_n) @ K_Gamma_bar_qe)
                )

                f_pot_q_el[self.nodalDOF_element_psi[node], :] -= (
                    self.N_psi_xi[el, i, node] * qwi * K_m_qe
                )

                f_pot_q_el[self.nodalDOF_element_psi[node], :] += (
                    self.N_psi[el, i, node]
                    * qwi
                    * (ax2skew(K_Kappa_bar) @ K_m_qe - ax2skew(K_m) @ K_Kappa_bar_qe)
                )

        return f_pot_q_el

        # f_pot_q_el_num = approx_fprime(
        #     qe, lambda qe: self.f_pot_el(qe, el), eps=1.0e-10, method="cs"
        # )
        # # f_pot_q_el_num = approx_fprime(
        # #     qe, lambda qe: self.f_pot_el(qe, el), eps=5.0e-6, method="2-point"
        # # )
        # diff = f_pot_q_el - f_pot_q_el_num
        # error = np.linalg.norm(diff)
        # print(f"error f_pot_q_el: {error}")
        # return f_pot_q_el_num

    #########################################
    # equations of motion
    #########################################
    def assembler_callback(self):
        if self.constant_mass_matrix:
            self._M_coo()

    def M_el_constant(self, el):
        M_el = np.zeros((self.nq_element, self.nq_element), dtype=float)

        for i in range(self.nquadrature):
            # extract reference state variables
            qwi = self.qw[el, i]
            Ji = self.J[el, i]

            # delta_r A_rho0 r_ddot part
            M_el_r_r = np.eye(3) * self.A_rho0 * Ji * qwi
            for node_a in range(self.nnodes_element_r):
                nodalDOF_a = self.nodalDOF_element_r[node_a]
                for node_b in range(self.nnodes_element_r):
                    nodalDOF_b = self.nodalDOF_element_r[node_b]
                    M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_r_r * (
                        self.N_r[el, i, node_a] * self.N_r[el, i, node_b]
                    )

            # first part delta_phi (I_rho0 omega_dot + omega_tilde I_rho0 omega)
            M_el_psi_psi = self.K_I_rho0 * Ji * qwi
            for node_a in range(self.nnodes_element_psi):
                nodalDOF_a = self.nodalDOF_element_psi[node_a]
                for node_b in range(self.nnodes_element_psi):
                    nodalDOF_b = self.nodalDOF_element_psi[node_b]
                    M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_psi_psi * (
                        self.N_psi[el, i, node_a] * self.N_psi[el, i, node_b]
                    )

        return M_el

    def M_el(self, qe, el):
        M_el = np.zeros((self.nq_element, self.nq_element), dtype=float)

        for i in range(self.nquadrature):
            # extract reference state variables
            qwi = self.qw[el, i]
            Ji = self.J[el, i]

            # delta_r A_rho0 r_ddot part
            M_el_r_r = np.eye(3) * self.A_rho0 * Ji * qwi
            for node_a in range(self.nnodes_element_r):
                nodalDOF_a = self.nodalDOF_element_r[node_a]
                for node_b in range(self.nnodes_element_r):
                    nodalDOF_b = self.nodalDOF_element_r[node_b]
                    M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_r_r * (
                        self.N_r[el, i, node_a] * self.N_r[el, i, node_b]
                    )

            # first part delta_phi (I_rho0 omega_dot + omega_tilde I_rho0 omega)
            M_el_psi_psi = self.K_I_rho0 * Ji * qwi
            for node_a in range(self.nnodes_element_psi):
                nodalDOF_a = self.nodalDOF_element_psi[node_a]
                for node_b in range(self.nnodes_element_psi):
                    nodalDOF_b = self.nodalDOF_element_psi[node_b]
                    M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_psi_psi * (
                        self.N_psi[el, i, node_a] * self.N_psi[el, i, node_b]
                    )

            # For non symmetric cross sections there are also other parts
            # involved in the mass matrix. These parts are configuration
            # dependent and lead to configuration dependent mass matrix.
            _, A_IK, _, _ = self.eval(qe, self.qp[el, i])
            M_el_r_psi = A_IK @ self.K_S_rho0 * Ji * qwi
            M_el_psi_r = A_IK @ self.K_S_rho0 * Ji * qwi

            for node_a in range(self.nnodes_element_r):
                nodalDOF_a = self.nodalDOF_element_r[node_a]
                for node_b in range(self.nnodes_element_psi):
                    nodalDOF_b = self.nodalDOF_element_psi[node_b]
                    M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_r_psi * (
                        self.N_r[el, i, node_a] * self.N_psi[el, i, node_b]
                    )
            for node_a in range(self.nnodes_element_psi):
                nodalDOF_a = self.nodalDOF_element_psi[node_a]
                for node_b in range(self.nnodes_element_r):
                    nodalDOF_b = self.nodalDOF_element_r[node_b]
                    M_el[nodalDOF_a[:, None], nodalDOF_b] += M_el_psi_r * (
                        self.N_psi[el, i, node_a] * self.N_r[el, i, node_b]
                    )

        return M_el

    def _M_coo(self):
        self.__M = Coo((self.nu, self.nu))
        for el in range(self.nelement):
            # extract element degrees of freedom
            elDOF = self.elDOF[el]

            # sparse assemble element mass matrix
            self.__M.extend(
                self.M_el_constant(el), (self.uDOF[elDOF], self.uDOF[elDOF])
            )

    def M(self, t, q, coo):
        if self.constant_mass_matrix:
            coo.extend_sparse(self.__M)
        else:
            for el in range(self.nelement):
                # extract element degrees of freedom
                elDOF = self.elDOF[el]

                # sparse assemble element mass matrix
                coo.extend(
                    self.M_el(q[elDOF], el), (self.uDOF[elDOF], self.uDOF[elDOF])
                )

    def f_gyr_el(self, t, qe, ue, el):
        f_gyr_el = np.zeros(self.nq_element, dtype=float)

        for i in range(self.nquadrature):
            # interpoalte angular velocity
            K_Omega = np.zeros(3, dtype=float)
            for node in range(self.nnodes_element_psi):
                K_Omega += self.N_psi[el, i, node] * ue[self.nodalDOF_element_psi[node]]

            # vector of gyroscopic forces
            f_gyr_el_psi = (
                cross3(K_Omega, self.K_I_rho0 @ K_Omega)
                * self.J[el, i]
                * self.qw[el, i]
            )

            # multiply vector of gyroscopic forces with nodal virtual rotations
            for node in range(self.nnodes_element_psi):
                f_gyr_el[self.nodalDOF_element_psi[node]] += (
                    self.N_psi[el, i, node] * f_gyr_el_psi
                )

        return f_gyr_el

    def f_gyr_u_el(self, t, qe, ue, el):
        f_gyr_u_el = np.zeros((self.nq_element, self.nq_element), dtype=float)

        for i in range(self.nquadrature):
            # interpoalte angular velocity
            K_Omega = np.zeros(3, dtype=float)
            for node in range(self.nnodes_element_psi):
                K_Omega += self.N_psi[el, i, node] * ue[self.nodalDOF_element_psi[node]]

            # derivative of vector of gyroscopic forces
            f_gyr_u_el_psi = (
                ((ax2skew(K_Omega) @ self.K_I_rho0 - ax2skew(self.K_I_rho0 @ K_Omega)))
                * self.J[el, i]
                * self.qw[el, i]
            )

            # multiply derivative of gyroscopic force vector with nodal virtual rotations
            for node_a in range(self.nnodes_element_psi):
                nodalDOF_a = self.nodalDOF_element_psi[node_a]
                for node_b in range(self.nnodes_element_psi):
                    nodalDOF_b = self.nodalDOF_element_psi[node_b]
                    f_gyr_u_el[nodalDOF_a[:, None], nodalDOF_b] += f_gyr_u_el_psi * (
                        self.N_psi[el, i, node_a] * self.N_psi[el, i, node_b]
                    )

        return f_gyr_u_el

    def h(self, t, q, u):
        h = np.zeros(self.nu, dtype=q.dtype)
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            h[elDOF] += self.f_pot_el(q[elDOF], el) - self.f_gyr_el(
                t, q[elDOF], u[elDOF], el
            )
        return h

    def h_q(self, t, q, u, coo):
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            h_q_el = self.f_pot_el_q(q[elDOF], el)
            coo.extend(h_q_el, (self.uDOF[elDOF], self.qDOF[elDOF]))

    def h_u(self, t, q, u, coo):
        for el in range(self.nelement):
            elDOF = self.elDOF[el]
            h_u_el = -self.f_gyr_u_el(t, q[elDOF], u[elDOF], el)
            coo.extend(h_u_el, (self.uDOF[elDOF], self.uDOF[elDOF]))

    ####################################################
    # interactions with other bodies and the environment
    ####################################################
    def elDOF_P(self, frame_ID):
        xi = frame_ID[0]
        el = self.element_number(xi)
        return self.elDOF[el]

    def local_qDOF_P(self, frame_ID):
        return self.elDOF_P(frame_ID)

    def local_uDOF_P(self, frame_ID):
        return self.elDOF_P(frame_ID)

    #########################
    # r_OP/ A_IK contribution
    #########################
    def r_OP(self, t, q, frame_ID, K_r_SP=np.zeros(3, dtype=float)):
        # evaluate shape functions
        N_r, _ = self.basis_functions_r(frame_ID[0])

        # interpolate centerline
        r = np.zeros(3, dtype=q.dtype)
        for node in range(self.nnodes_element_r):
            r += N_r[node] * q[self.nodalDOF_element_r[node]]

        # interpolate orientation
        A_IK = self.A_IK(t, q, frame_ID)

        return r + A_IK @ K_r_SP

    def r_OP_q(self, t, q, frame_ID, K_r_SP=np.zeros(3, dtype=float)):
        # evaluate shape functions
        N_r, _ = self.basis_functions_r(frame_ID[0])

        # interpolate centerline position
        r_q = np.zeros((3, self.nq_element), dtype=q.dtype)
        for node in range(self.nnodes_element_r):
            nodalDOF_r = self.nodalDOF_element_r[node]
            r_q[:, nodalDOF_r] += N_r[node] * np.eye(3, dtype=float)

        # interpolate orientation
        A_IK_q = self.A_IK_q(t, q, frame_ID)

        r_OP_q = r_q + np.einsum("k,kij", K_r_SP, A_IK_q)
        return r_OP_q

        # r_OP_q_num = approx_fprime(
        #     q, lambda q: self.r_OP(t, q, frame_ID, K_r_SP), eps=1.0e-10, method="cs"
        # )
        # diff = r_OP_q - r_OP_q_num
        # error = np.linalg.norm(diff)
        # print(f"error r_OP_q: {error}")
        # return r_OP_q_num

    def v_P(self, t, q, u, frame_ID, K_r_SP=np.zeros(3, dtype=float)):
        N_r, _ = self.basis_functions_r(frame_ID[0])
        N_psi, _ = self.basis_functions_psi(frame_ID[0])

        # interpolate A_IK and angular velocity in K-frame
        A_IK = self.A_IK(t, q, frame_ID)
        K_Omega = np.zeros(3, dtype=np.common_type(q, u))
        for node in range(self.nnodes_element_psi):
            K_Omega += N_psi[node] * u[self.nodalDOF_element_psi[node]]

        # angular velocity in K-frame
        K_Omega = np.zeros(3, dtype=np.common_type(q, u))
        for node in range(self.nnodes_element_psi):
            K_Omega += N_psi[node] * u[self.nodalDOF_element_psi[node]]

        # centerline velocity
        v = np.zeros(3, dtype=np.common_type(q, u))
        for node in range(self.nnodes_element_r):
            v += N_r[node] * u[self.nodalDOF_element_r[node]]

        return v + A_IK @ cross3(K_Omega, K_r_SP)

    def v_P_q(self, t, q, u, frame_ID, K_r_SP=np.zeros(3, dtype=float)):
        # evaluate shape functions
        N_psi, _ = self.basis_functions_psi(frame_ID[0])

        # interpolate derivative of A_IK and angular velocity in K-frame
        A_IK_q = self.A_IK_q(t, q, frame_ID)
        K_Omega = np.zeros(3, dtype=np.common_type(q, u))
        for node in range(self.nnodes_element_psi):
            K_Omega += N_psi[node] * u[self.nodalDOF_element_psi[node]]

        v_P_q = np.einsum(
            "ijk,j->ik",
            A_IK_q,
            cross3(K_Omega, K_r_SP),
        )
        return v_P_q

        # v_P_q_num = approx_fprime(
        #     q, lambda q: self.v_P(t, q, u, frame_ID, K_r_SP), method="3-point"
        # )
        # diff = v_P_q - v_P_q_num
        # error = np.linalg.norm(diff)
        # print(f"error v_P_q: {error}")
        # return v_P_q_num

    def J_P(self, t, q, frame_ID, K_r_SP=np.zeros(3, dtype=float)):
        # evaluate required nodal shape functions
        N_r, _ = self.basis_functions_r(frame_ID[0])
        N_psi, _ = self.basis_functions_psi(frame_ID[0])

        # transformation matrix
        A_IK = self.A_IK(t, q, frame_ID)

        # skew symmetric matrix of K_r_SP
        K_r_SP_tilde = ax2skew(K_r_SP)

        # interpolate centerline and axis angle contributions
        J_P = np.zeros((3, self.nq_element))
        for node in range(self.nnodes_element_r):
            J_P[:, self.nodalDOF_element_r[node]] += N_r[node] * np.eye(3)
        for node in range(self.nnodes_element_psi):
            J_P[:, self.nodalDOF_element_psi[node]] -= N_psi[node] * A_IK @ K_r_SP_tilde

        return J_P

        # J_P_num = approx_fprime(
        #     np.zeros(self.nq_element, dtype=float),
        #     lambda u: self.v_P(t, q, u, frame_ID, K_r_SP),
        # )
        # diff = J_P_num - J_P
        # error = np.linalg.norm(diff)
        # print(f"error J_P: {error}")
        # return J_P_num

    # TODO:
    def J_P_q(self, t, q, frame_ID, K_r_SP=np.zeros(3, dtype=float)):
        return approx_fprime(
            q, lambda q: self.J_P(t, q, frame_ID, K_r_SP), method="2-point"
        )

        # evaluate required nodal shape functions
        N_psi, _ = self.basis_functions_psi(frame_ID[0])

        K_r_SP_tilde = ax2skew(K_r_SP)
        A_IK_q = self.A_IK_q(t, q, frame_ID)
        prod = np.einsum("ijl,jk", A_IK_q, K_r_SP_tilde)

        # interpolate axis angle contributions since centerline contributon is
        # zero
        J_P_q = np.zeros((3, self.nq_element, self.nq_element), dtype=float)
        for node in range(self.nnodes_element_psi):
            nodalDOF_psi = self.nodalDOF_element_psi[node]
            J_P_q[:, nodalDOF_psi] -= N_psi[node] * prod

        return J_P_q

        # J_P_q_num = approx_fprime(
        #     q, lambda q: self.J_P(t, q, frame_ID, K_r_SP), method="3-point"
        # )
        # diff = J_P_q_num - J_P_q
        # error = np.linalg.norm(diff)
        # print(f"error J_P_q: {error}")
        # return J_P_q_num

    def a_P(self, t, q, u, u_dot, frame_ID, K_r_SP=np.zeros(3, dtype=float)):
        N_r, _ = self.basis_functions_r(frame_ID[0])

        # interpolate orientation
        A_IK = self.A_IK(t, q, frame_ID)

        # centerline acceleration
        a = np.zeros(3, dtype=float)
        for node in range(self.nnodes_element_r):
            a += N_r[node] * u_dot[self.nodalDOF_element_r[node]]

        # angular velocity and acceleration in K-frame
        K_Omega = self.K_Omega(t, q, u, frame_ID)
        K_Psi = self.K_Psi(t, q, u, u_dot, frame_ID)

        # rigid body formular
        return a + A_IK @ (
            cross3(K_Psi, K_r_SP) + cross3(K_Omega, cross3(K_Omega, K_r_SP))
        )

    # TODO:
    def a_P_q(self, t, q, u, u_dot, frame_ID, K_r_SP=None):
        raise NotImplementedError
        K_Omega = self.K_Omega(t, q, u, frame_ID)
        K_Psi = self.K_Psi(t, q, u, u_dot, frame_ID)
        a_P_q = np.einsum(
            "ijk,j->ik",
            self.A_IK_q(t, q, frame_ID),
            cross3(K_Psi, K_r_SP) + cross3(K_Omega, cross3(K_Omega, K_r_SP)),
        )
        # return a_P_q

        a_P_q_num = approx_fprime(
            q, lambda q: self.a_P(t, q, u, u_dot, frame_ID, K_r_SP), method="3-point"
        )
        diff = a_P_q_num - a_P_q
        error = np.linalg.norm(diff)
        print(f"error a_P_q: {error}")
        return a_P_q_num

    # TODO:
    def a_P_u(self, t, q, u, u_dot, frame_ID, K_r_SP=None):
        raise NotImplementedError
        K_Omega = self.K_Omega(t, q, u, frame_ID)
        local = -self.A_IK(t, q, frame_ID) @ (
            ax2skew(cross3(K_Omega, K_r_SP)) + ax2skew(K_Omega) @ ax2skew(K_r_SP)
        )

        N, _ = self.basis_functions_r(frame_ID[0])
        a_P_u = np.zeros((3, self.nq_element), dtype=float)
        for node in range(self.nnodes_element_r):
            a_P_u[:, self.nodalDOF_element_psi[node]] += N[node] * local

        # return a_P_u

        a_P_u_num = approx_fprime(
            u, lambda u: self.a_P(t, q, u, u_dot, frame_ID, K_r_SP), method="3-point"
        )
        diff = a_P_u_num - a_P_u
        error = np.linalg.norm(diff)
        print(f"error a_P_u: {error}")
        return a_P_u_num

    def K_Omega(self, t, q, u, frame_ID):
        """Since we use Petrov-Galerkin method we only interpoalte the nodal
        angular velocities in the K-frame.
        """
        N_psi, _ = self.basis_functions_psi(frame_ID[0])
        K_Omega = np.zeros(3, dtype=np.common_type(q, u))
        for node in range(self.nnodes_element_psi):
            K_Omega += N_psi[node] * u[self.nodalDOF_element_psi[node]]
        return K_Omega

    def K_Omega_q(self, t, q, u, frame_ID):
        return np.zeros((3, self.nq_element), dtype=float)

    def K_J_R(self, t, q, frame_ID):
        N_psi, _ = self.basis_functions_psi(frame_ID[0])
        K_J_R = np.zeros((3, self.nu_element), dtype=float)
        for node in range(self.nnodes_element_psi):
            K_J_R[:, self.nodalDOF_element_psi[node]] += N_psi[node] * np.eye(
                3, dtype=float
            )
        return K_J_R

    def K_J_R_q(self, t, q, frame_ID):
        return np.zeros((3, self.nu_element, self.nq_element), dtype=float)

    def K_Psi(self, t, q, u, u_dot, frame_ID):
        """Since we use Petrov-Galerkin method we only interpoalte the nodal
        time derivative of the angular velocities in the K-frame.
        """
        N_psi, _ = self.basis_functions_psi(frame_ID[0])
        K_Psi = np.zeros(3, dtype=float)
        for node in range(self.nnodes_element_psi):
            K_Psi += N_psi[node] * u_dot[self.nodalDOF_element_psi[node]]
        return K_Psi

    def K_Psi_q(self, t, q, u, u_dot, frame_ID):
        return np.zeros((3, self.nq_element), dtype=float)

    def K_Psi_u(self, t, q, u, u_dot, frame_ID):
        return np.zeros((3, self.nu_element), dtype=float)

    ####################################################
    # body force
    ####################################################
    def distributed_force1D_pot_el(self, force, t, qe, el):
        Ve = 0
        for i in range(self.nquadrature):
            # extract reference state variables
            qwi = self.qw[el, i]
            Ji = self.J[el, i]

            # interpolate centerline position
            r_C = np.zeros(3, dtype=float)
            for node in range(self.nnodes_element_r):
                r_C += self.N_r[el, i, node] * qe[self.nodalDOF_element_r[node]]

            # compute potential value at given quadrature point
            Ve += (r_C @ force(t, qwi)) * Ji * qwi

        return Ve

    def distributed_force1D_pot(self, t, q, force):
        V = 0
        for el in range(self.nelement):
            qe = q[self.elDOF[el]]
            V += self.distributed_force1D_pot_el(force, t, qe, el)
        return V

    def distributed_force1D_el(self, force, t, el):
        fe = np.zeros(self.nq_element, dtype=float)
        for i in range(self.nquadrature):
            # extract reference state variables
            qwi = self.qw[el, i]
            Ji = self.J[el, i]

            # compute local force vector
            fe_r = force(t, qwi) * Ji * qwi

            # multiply local force vector with variation of centerline
            for node in range(self.nnodes_element_r):
                fe[self.nodalDOF_element_r[node]] += self.N_r[el, i, node] * fe_r

        return fe

    def distributed_force1D(self, t, q, force):
        f = np.zeros(self.nq, dtype=float)
        for el in range(self.nelement):
            f[self.elDOF[el]] += self.distributed_force1D_el(force, t, el)
        return f

    def distributed_force1D_q(self, t, q, coo, force):
        pass
