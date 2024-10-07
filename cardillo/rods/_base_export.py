from abc import ABC, abstractmethod
import numpy as np
from vtk import VTK_BEZIER_WEDGE, VTK_BEZIER_HEXAHEDRON, VTK_LAGRANGE_CURVE
from warnings import warn

from cardillo.utility.bezier import L2_projection_Bezier_curve
from cardillo.math import cross3

from ._cross_section import (
    CrossSection,
    CircularCrossSection,
    RectangularCrossSection,
)

"""
very usefull for the ordering: 
https://coreform.com/papers/implementation-of-rational-bezier-cells-into-VTK-report.pdf
"""


class RodExportBase(ABC):
    def __init__(self, cross_section: CrossSection):
        self.cross_section = cross_section

        self._export_dict = {
            # "level": "centerline + directors",
            "level": "volume",
            "num_per_cell": "Auto",
            "ncells": "Auto",
            "stresses": False,
            "volume_directors": False,
            "surface_normals": False,
        }

        self.preprocessed_export = False

    @abstractmethod
    def r_OP(self, t, q, xi, B_r_CP): ...

    @abstractmethod
    def A_IB(self, t, q, xi): ...

    def centerline(self, q, num=100):
        q_body = q[self.qDOF]
        r = []
        for xi in np.linspace(0, 1, num):
            qp = q_body[self.local_qDOF_P(xi)]
            r.append(self.r_OP(1, qp, xi))
        return np.array(r).T

    def frames(self, q, num=10):
        q_body = q[self.qDOF]
        r = []
        d1 = []
        d2 = []
        d3 = []

        for xi in np.linspace(0, 1, num=num):
            qp = q_body[self.local_qDOF_P(xi)]
            r.append(self.r_OP(1, qp, xi))

            d1i, d2i, d3i = self.A_IB(1, qp, xi).T
            d1.extend([d1i])
            d2.extend([d2i])
            d3.extend([d3i])

        return np.array(r).T, np.array(d1).T, np.array(d2).T, np.array(d3).T

    def preprocess_export(self):
        ########################
        # evaluate export dict #
        ########################
        # number if cells to be exported
        if self._export_dict["ncells"] == "Auto":
            self._export_dict["ncells"] = self.nelement

        assert isinstance(self._export_dict["ncells"], int)
        assert self._export_dict["ncells"] >= 1
        ncells = self._export_dict["ncells"]

        # continuity for L2 projection
        self._export_dict["continuity"] = "C0"

        # export options for circle
        # TODO: maybe this should be part of the cross section
        self._export_dict["circle_as_wedge"] = True

        # what shall be exported
        assert self._export_dict["level"] in [
            "centerline + directors",
            "volume",
            "None",
            None,
        ]

        ########################################################
        # get number of frames for export and make cell-arrays #
        ########################################################
        if self._export_dict["level"] == "centerline + directors":
            # TODO: maybe export directors with respect to their polynomialdegree (to see them at their nodes)
            # for polynomial shape functions this should be the best.
            # But for other interpolations (SE3) this is not good!

            num_per_cell = self._export_dict["num_per_cell"]
            if num_per_cell == "Auto":
                p = self.polynomial_degree_r
            else:
                p = num_per_cell - 1

            # compute number of frames
            self._export_dict["num_frames"] = p * ncells + 1

            # create cells
            self._export_dict["cells"] = [
                (
                    VTK_LAGRANGE_CURVE,
                    np.concatenate([[0], [p], np.arange(1, p)]) + i * p,
                )
                for i in range(ncells)
            ]

        elif self._export_dict["level"] == "volume":
            assert isinstance(
                self.cross_section, (CircularCrossSection, RectangularCrossSection)
            ), "Volume export is only implemented for CircularCrossSection and RectangularCrossSection."

            # always use 4 here
            # this is only the number of points we use for the projection
            self._export_dict["num_frames"] = 4 * ncells + 1

            # make cells
            p_zeta = 3  # polynomial_degree of the cell along the rod. Always 3, this is hardcoded in L2_projection when BernsteinBasis is called!
            # determines also the number of layers per cell (p_zeta + 1 = 4), which is also very hardcoded here with 'points_layer_0' ... 'points_layer3'
            if isinstance(self.cross_section, CircularCrossSection):
                if self._export_dict["circle_as_wedge"]:
                    self._export_dict["hasCap"] = True
                    # TODO: document this, when it is completed with the possibility for discontinuous stresses
                    points_per_layer = 6
                    points_per_cell = (p_zeta + 1) * points_per_layer

                    # BiQuadratic(p_zeta) cells alternating with BiQuadraticLinear (flat) cells for the end faces and to remove the internal faces in the rod
                    self._export_dict["higher_order_degrees"] = np.vstack(
                        [[2, 2, 1], [[2, 2, p_zeta], [2, 2, 1]] * ncells]
                    )

                    # Note: if there is a closed rod (ring):
                    #   we have to set the ids for the top of the last cell
                    #   to the ids of the bottom from the first cell:
                    #   last_cell[0:3] = [0, 1, 2]
                    #   last_cell[6:9] = [3, 4, 5]
                    #   what should be also done: do not add them into "vtk_points_weights"
                    #   in this case, the first and last cell must not be used
                    # fmt: off
                    connectivity_main = np.array(
                        [
                             0,  1,  2, # vertices bottom   l0
                            18, 19, 20, # vertices top      l3
                             3,  4,  5, # edges bottom      l0
                            21, 22, 23, # edges top         l3
                             6, 12,     # edge1 middle      l1&l2
                             7, 13,     # edge2 middle      l1&l2
                             8, 14,     # edge3 middle      l1&l2
                             9, 15,     # faces1 middle     l1&l2
                            10, 16,     # faces2 middle     l1&l2
                            11, 17,     # faces3 middle     l1&l2
                        # l1|^, |^l2
                        ], 
                        dtype=int
                    )
                    connectivity_flat = np.array(
                        [
                            18, 19, 20, # vertiecs bottom
                            24, 25, 26, # vertices top
                            21, 22, 23, # edges bottom
                            27, 28, 29, # edges top
                        ],
                        dtype=int
                    )
                    # fmt: on
                    cells = [
                        (
                            VTK_BEZIER_WEDGE,
                            connectivity_flat - points_per_cell + points_per_layer,
                        )
                    ]
                    for i in range(ncells):
                        for c in [connectivity_main, connectivity_flat]:
                            cells.append(
                                (
                                    VTK_BEZIER_WEDGE,
                                    c + i * points_per_cell + points_per_layer,
                                )
                            )

                else:
                    self._export_dict["hasCap"] = False
                    points_per_layer = 9
                    points_per_cell = (p_zeta + 1) * points_per_layer

                    # TODO: add here the BiQuadraticLinear cells with 0-width to remove inner walls when opacity is reduced

                    # BiQuadratic(p_zeta) cells
                    self._export_dict["higher_order_degrees"] = np.array(
                        [[2, 2, p_zeta]] * ncells
                    )

                    # connectivity with points
                    cells = [
                        (
                            VTK_BEZIER_HEXAHEDRON,
                            np.arange(i * points_per_cell, (i + 1) * points_per_cell),
                        )
                        for i in range(ncells)
                    ]
            elif isinstance(self.cross_section, RectangularCrossSection):
                self._export_dict["hasCap"] = False
                points_per_layer = 4
                points_per_cell = (p_zeta + 1) * points_per_layer

                # BiLinear(p_zeta) cells, alternattely with BiLinearLinear cells with 0-width to remove inner walls when opacity is reduced and to keep posibility for dicontiuous stresses
                self._export_dict["higher_order_degrees"] = np.array(
                    [[1, 1, p_zeta], [1, 1, 1]] * ncells
                )[:-1]

                # connectivities
                # ordering for vtu file version>=2.0
                # ordering for vtu file version<2.0, e.g. 0.1 changes order of edges! like [0, 1, 3, 2]
                # fmt: off
                connectivity_main = np.array(
                    [
                         0,  1,  2,  3,     # vertices bottom
                        12, 13, 14, 15,     # vertices top
                         4,  8,             # edge1
                         5,  9,             # edge2
                         6, 10,             # edge3
                         7, 11,             # edge4
                    ],
                    dtype=int
                )
                connectivity_flat = np.array(
                    [
                        12, 13, 14, 15,
                        16, 17, 18, 19
                    ],
                    dtype=int
                )
                # fmt: on

                # connectivity with points
                cells = []
                for i in range(ncells):
                    for c in [connectivity_main, connectivity_flat]:
                        cells.append((VTK_BEZIER_HEXAHEDRON, c + i * points_per_cell))
                # remove last (internal) cell
                cells = cells[:-1]

            self._export_dict["cells"] = cells
            self._export_dict["points_per_layer"] = points_per_layer
            self._export_dict["p_zeta"] = p_zeta

            ###############################
            # assertion for stress export #
            ###############################
            # TODO: remove this assertion, when we compute the stresses on element level
            # stresses are evaluated at xis
            # stresses are very likely to jump at xis_e_int
            # therfore, we shouldn't evaluate them at these points
            num = self._export_dict["num_frames"] - 1
            xis = np.linspace(0, 1, num=num)
            xis_e_int = np.linspace(0, 1, self.nelement + 1)[1:-1]
            for xi_e in xis_e_int:
                assert (
                    xi_e not in xis
                ), f"xis for fitting may not contain internal element boundaries to represent discontinuities in stresses. \nInternal boundaries at {xis_e_int}, \nxis_fitting={xis}."

            #################################
            # assertion for surface normals #
            #################################
            if self._export_dict["surface_normals"]:
                assert isinstance(self.cross_section, CircularCrossSection)
                warn(
                    "surface normals are not implmented correctly! They are wrong when shear is present!"
                )

        elif self._export_dict["level"] == "None" or self._export_dict["level"] == None:
            self._export_dict["cells"] = []

        # set flag value to True
        self.preprocessed_export = True

    def export(self, sol_i, **kwargs):
        q = sol_i.q

        if not self.preprocessed_export:
            self.preprocess_export()

        # get values that very computed in preprocess into local scope
        ncells = self._export_dict["ncells"]
        continuity = self._export_dict["continuity"]
        circle_as_wedge = self._export_dict["circle_as_wedge"]
        level = self._export_dict["level"]
        num = self._export_dict["num_frames"]

        # get frames
        r_OPs, d1s, d2s, d3s = self.frames(q, num=num)

        if level == "centerline + directors":
            #######################################
            # simple export of points and directors
            #######################################
            # fill structs for centerline and directors
            vtk_points = r_OPs.T

            point_data = {
                "d1": d1s.T,
                "d2": d2s.T,
                "d3": d3s.T,
            }
            # TODO: add stresses here?
            # here is a bit more work to do, as the points allo for now only C0 continuity in stresses!
            cell_data = {}

            return vtk_points, self._export_dict["cells"], point_data, cell_data

        elif level == "volume":
            ################################
            # project on cubic Bezier volume
            ################################
            # project directors on cubic C1 bezier curve
            # TODO: if that is the case, export them also for circular cross sections
            # TODO: I think this is a but useless:
            #       Will anyone color the mesh based on the components of the directors?
            #       Or place glyphs at all vertices?
            # TODO: I think there is time-saving potential in the L2_projection_Bezier_curve (e.g., the matrix A there is constant, but created and filled everytime)
            # TODO: can we get rid of the d1 projection?
            r_OP_segments = L2_projection_Bezier_curve(
                r_OPs.T, ncells, case=continuity
            )[2]
            d1_segments = L2_projection_Bezier_curve(d1s.T, ncells, case=continuity)[2]
            d2_segments = L2_projection_Bezier_curve(d2s.T, ncells, case=continuity)[2]
            d3_segments = L2_projection_Bezier_curve(d3s.T, ncells, case=continuity)[2]

            # define functions to get points, normals, ...
            ppl = self._export_dict["points_per_layer"]
            p_zeta = self._export_dict["p_zeta"]
            if isinstance(self.cross_section, CircularCrossSection):
                if circle_as_wedge:

                    # points:
                    #         ^ d3
                    #         |
                    #         x  P4
                    #       / | \
                    #      /-----\
                    # P2  x   |   x  P1
                    #    /|---o---|\--> d2
                    #   / \   |   / \
                    #  x---\--x--/---x
                    #  P5     P0     P3
                    #
                    # P0-P1-P2 form the inner equilateral triangle. Later refered to as points for layer 0 and 3 or edges for layer 1 and 2. They are weighted with 1
                    # P3-P4-P5 from the outer equilateral traingle. Later refered to as edges for layer 0 and 3 or faces for layer 1 and 2. They are weighted with 1/2
                    #

                    # TODO: maybe preprocess these or maybe add them to Cross-Section?

                    # alpha measures the angle to d2
                    alpha0 = np.pi * 3 / 2
                    alpha1 = np.pi / 6
                    alpha2 = np.pi * 5 / 6
                    alpha3 = np.pi * 11 / 6
                    alpha4 = np.pi / 2
                    alpha5 = np.pi * 7 / 6

                    alphas = [alpha0, alpha1, alpha2, alpha3, alpha4, alpha5]
                    trig_alphas = [(np.sin(alpha), np.cos(alpha)) for alpha in alphas]

                    # radii
                    ri = self.cross_section.radius
                    ru = 2 * ri
                    a = 2 * np.sqrt(3) * ri

                    def compute_missing_points(segment, layer):
                        r_OP = r_OP_segments[segment, layer]
                        d2 = d2_segments[segment, layer]
                        d3 = d3_segments[segment, layer]

                        P0 = r_OP - ri * d3
                        P3 = r_OP + a / 2 * d2 - ri * d3
                        P4 = r_OP + d3 * ru

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

                    # TODO: think about how to proceed with the normals
                    def compute_normals(segment, layer):
                        # TODO: these are not the normals if there is shear!
                        d2ii = d2_segments[segment, layer]
                        d3ii = d3_segments[segment, layer]

                        normals = np.zeros([6, 3])
                        for i in range(6):
                            salpha, calpha = trig_alphas[i]
                            normals[i, :] = calpha * d2ii + salpha * d3ii

                        return normals

                else:
                    warn(
                        f"Test rod export: circle as VTK_BEZIER_HEXAHEDRON may leads to unexpected results!"
                    )

                    def compute_missing_points(segment, layer):
                        r_OP = r_OP_segments[segment, layer]
                        d2 = d2_segments[segment, layer]
                        d3 = d3_segments[segment, layer]

                        P2 = r_OP + d3 * self.cross_section.radius
                        P1 = r_OP + d2 * self.cross_section.radius
                        P8 = r_OP

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
                    for i in range(ncells):
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
                        # for j in [0, 1, 3, 2]:  # ordering for vtu file version<2.0, e.g. 0.1
                        for j in range(4):  # ordering for vtu file version>=2.0
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
                    r_OP = r_OP_segments[segment, layer]
                    d2 = d2_segments[segment, layer]
                    d3 = d3_segments[segment, layer]

                    #       ^ d3
                    # P3    |     P2
                    # x-----Q2----x
                    # |     |     |
                    # |-----Q0----Q1--> d2
                    # |           |
                    # x-----------x
                    # P0          P1

                    r_PP2 = (
                        d2 * self.cross_section.width / 2
                        + d3 * self.cross_section.height / 2
                    )
                    r_PP1 = (
                        d2 * self.cross_section.width / 2
                        - d3 * self.cross_section.height / 2
                    )

                    P0 = r_OP - r_PP2
                    P1 = r_OP + r_PP1
                    P2 = r_OP + r_PP2
                    P3 = r_OP - r_PP1

                    dim = len(P0)
                    points_weights = np.zeros((4, dim + 1))
                    points_weights[0] = np.array([*P0, 1])
                    points_weights[1] = np.array([*P1, 1])
                    points_weights[2] = np.array([*P2, 1])
                    points_weights[3] = np.array([*P3, 1])

                    return points_weights

            ##################
            # compute points #
            ##################
            vtk_points_weights = []
            # cap at xi=0
            if self._export_dict["hasCap"]:
                vtk_points_weights.extend(compute_missing_points(0, 0))

            # iterate all cells
            for i in range(ncells):
                # iterate all layers
                for layer in range(p_zeta + 1):
                    vtk_points_weights.extend(compute_missing_points(i, layer))

            # cap at xi=1
            if self._export_dict["hasCap"]:
                vtk_points_weights.extend(compute_missing_points(-1, -1))

            # points to export is just the R^3 part
            vtk_points_weights = np.array(vtk_points_weights)
            vtk_points = vtk_points_weights[:, :3]

            # the 4th component are the rational weights
            point_data = {
                "RationalWeights": vtk_points_weights[:, 3, None],
            }

            ################
            # add stresses #
            ################
            if self._export_dict["stresses"]:
                # TODO: do this on element basis when eval_stresses accepts el as argument
                num = self._export_dict["num_frames"] - 1
                xis = np.linspace(0, 1, num=num)
                B_ns = np.zeros([3, num])
                B_ms = np.zeros([3, num])

                t = sol_i.t
                la_c = sol_i.la_c
                la_g = sol_i.la_g
                for j in range(num):
                    B_ns[:, j], B_ms[:, j] = self.eval_stresses(
                        t, q, la_c, la_g, xis[j]
                    )

                # project contact forces and moments on cubic C1 bezier curve
                B_n_segments = L2_projection_Bezier_curve(
                    B_ns.T,
                    ncells,
                    case="C-1",
                )[2]
                B_m_segments = L2_projection_Bezier_curve(
                    B_ms.T,
                    ncells,
                    case="C-1",
                )[2]

                # fill lists
                vtk_B_n = []
                vtk_B_m = []
                # cap at xi=0
                if self._export_dict["hasCap"]:
                    vtk_B_n.extend(np.repeat([B_n_segments[0, 0]], ppl, axis=0))
                    vtk_B_m.extend(np.repeat([B_m_segments[0, 0]], ppl, axis=0))

                # iterate all cells
                for i in range(ncells):
                    # iterate all layers
                    for layer in range(p_zeta + 1):
                        vtk_B_n.extend(np.repeat([B_n_segments[i, layer]], ppl, axis=0))
                        vtk_B_m.extend(np.repeat([B_m_segments[i, layer]], ppl, axis=0))

                # cap at xi=1
                if self._export_dict["hasCap"]:
                    vtk_B_n.extend(np.repeat([B_n_segments[-1, -1]], ppl, axis=0))
                    vtk_B_m.extend(np.repeat([B_m_segments[-1, -1]], ppl, axis=0))

                # add them to dictionary with point data
                point_data["B_n"] = vtk_B_n
                point_data["B_m"] = vtk_B_m

            ########################
            # add volume directors #
            ########################
            if self._export_dict["volume_directors"]:
                vtk_d1 = []
                vtk_d2 = []
                vtk_d3 = []
                # cap at xi=0
                if self._export_dict["hasCap"]:
                    vtk_d1.extend(np.repeat([d1_segments[0, 0]], ppl, axis=0))
                    vtk_d2.extend(np.repeat([d2_segments[0, 0]], ppl, axis=0))
                    vtk_d3.extend(np.repeat([d3_segments[0, 0]], ppl, axis=0))

                # iterate all cells
                for i in range(ncells):
                    # iterate all layers
                    for layer in range(p_zeta + 1):
                        vtk_d1.extend(np.repeat([d1_segments[i, layer]], ppl, axis=0))
                        vtk_d2.extend(np.repeat([d2_segments[i, layer]], ppl, axis=0))
                        vtk_d3.extend(np.repeat([d3_segments[i, layer]], ppl, axis=0))

                # cap at xi=1
                if self._export_dict["hasCap"]:
                    vtk_d1.extend(np.repeat([d1_segments[-1, -1]], ppl, axis=0))
                    vtk_d2.extend(np.repeat([d2_segments[-1, -1]], ppl, axis=0))
                    vtk_d3.extend(np.repeat([d3_segments[-1, -1]], ppl, axis=0))

                # add them to dictionary with point data
                point_data["d1"] = vtk_d1
                point_data["d2"] = vtk_d2
                point_data["d3"] = vtk_d3

            #######################
            # add surface normals #
            #######################
            if self._export_dict["surface_normals"]:
                vtk_surface_normals = []
                # cap at xi=0
                if self._export_dict["hasCap"]:
                    vtk_surface_normals.extend(
                        np.repeat([-d1_segments[0, 0]], ppl, axis=0)
                    )

                # iterate all cells
                for i in range(ncells):
                    # iterate all layers
                    for layer in range(p_zeta + 1):
                        vtk_surface_normals.extend(compute_normals(i, layer))

                # cap at xi=1
                if self._export_dict["hasCap"]:
                    vtk_points_weights.extend(compute_normals(-1, -1))

                # add them to dictionary with point data
                point_data["surface_normal"] = vtk_surface_normals

            #################
            # add cell data #
            #################
            cell_data = {
                "HigherOrderDegrees": self._export_dict["higher_order_degrees"],
            }

        elif level == "None" or level == None:
            vtk_points = []
            point_data = {}
            cell_data = {}

        return vtk_points, self._export_dict["cells"], point_data, cell_data
