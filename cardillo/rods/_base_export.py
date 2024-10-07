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


class RodExportBase(ABC):
    def __init__(self, cross_section: CrossSection):
        self.cross_section = cross_section

        self._export_dict = {
            # "level": "centerline + directors",
            "level": "volume",
            "num_per_cell": "Auto",
            "ncells": "Auto",
            "stresses": False,
            "surface_normals": True,
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
            # always use 4 here
            # this is only the number of points we use for the projection
            self._export_dict["num_frames"] = 4 * ncells + 1

            # make cells
            p_zeta = 3  # polynomial_degree of the cell along the rod. Always 3, this is hardcoded in L2_projection when BernsteinBasis is called!
            # determines also the number of layers per cell (p_zeta + 1 = 4), which is also very hardcoded here with 'points_layer_0' ... 'points_layer3'
            if isinstance(self.cross_section, CircularCrossSection):
                if self._export_dict["circle_as_wedge"]:
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
                points_per_layer = 4
                points_per_cell = (p_zeta + 1) * points_per_layer

                # BiLinear(p_zeta) cells, alternattely with BiLinearLinear cells with 0-width to remove inner walls when opacity is reduced and to keep posibility for dicontiuous stresses
                self._export_dict["higher_order_degrees"] = np.array(
                    [[1, 1, p_zeta], [1, 1, 1]] * ncells
                )[:-1]

                # connectivities
                connectivity_main = np.arange(points_per_cell)
                connectivity_flat = np.array([4, 5, 6, 7, 16, 17, 18, 19])

                # connectivity with points
                cells = []
                for i in range(ncells):
                    for c in [connectivity_main, connectivity_flat]:
                        cells.append((VTK_BEZIER_HEXAHEDRON, c + i * points_per_cell))
                # remove last (internal) cell
                cells = cells[:-1]

            self._export_dict["cells"] = cells
            self._export_dict["points_per_layer"] = points_per_layer

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
            assert isinstance(
                self.cross_section, (CircularCrossSection, RectangularCrossSection)
            ), "Volume export is only implemented for CircularCrossSection and RectangularCrossSection."

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

            # compute points
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
                        points_weights[3] = np.array([*P3, 0])  # 1 / 2])
                        points_weights[4] = np.array([*P4, 0])  # 1 / 2])
                        points_weights[5] = np.array([*P5, 0])  # 1 / 2])

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

                    # create correct VTK ordering, see
                    # https://coreform.com/papers/implementation-of-rational-bezier-cells-into-VTK-report.pdf:
                    vtk_points_weights = []
                    vtk_surface_normals = []
                    # iterate two more to get the end faces clean
                    for i in range(-1, ncells + 1):

                        if i == -1:
                            # take points of 1st layer in 1st cell
                            points = compute_missing_points(0, 0)
                            d1i_neg = -d1_segments[0, 0]

                            # points and edges
                            for j in range(6):
                                vtk_points_weights.append(points[j])
                                vtk_surface_normals.append(d1i_neg)

                            continue

                        if i == ncells:
                            # take points of last layer in last cell
                            points = compute_missing_points(-1, -1)
                            d1i = d1_segments[-1, -1]

                            # points and edges
                            for j in range(6):
                                vtk_points_weights.append(points[j])
                                vtk_surface_normals.append(d1i)

                            continue

                        # compute all missing points of the layer
                        points_layer0 = compute_missing_points(i, 0)
                        points_layer1 = compute_missing_points(i, 1)
                        points_layer2 = compute_missing_points(i, 2)
                        points_layer3 = compute_missing_points(i, 3)

                        # set all values the same per layer for directors
                        normal_layer0 = compute_normals(i, 0)
                        normal_layer1 = compute_normals(i, 1)
                        normal_layer2 = compute_normals(i, 2)
                        normal_layer3 = compute_normals(i, 3)

                        # bottom
                        # points and edges
                        vtk_points_weights.extend(points_layer0)
                        vtk_surface_normals.extend(normal_layer0)

                        # first and second
                        # edges and faces
                        vtk_points_weights.extend(points_layer1)
                        vtk_points_weights.extend(points_layer2)
                        vtk_surface_normals.extend(normal_layer1)
                        vtk_surface_normals.extend(normal_layer2)

                        # top
                        # points and edges
                        vtk_points_weights.extend(points_layer3)
                        vtk_surface_normals.extend(normal_layer3)

                else:
                    from warnings import warn

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

                # create correct VTK ordering, see
                # https://coreform.com/papers/implementation-of-rational-bezier-cells-into-VTK-report.pdf:
                vtk_points_weights = []
                vtk_d1_weights = []
                vtk_d2_weights = []
                vtk_d3_weights = []
                for i in range(ncells):
                    # compute all missing points of the layer
                    points_layer0 = compute_missing_points(i, 0)
                    points_layer1 = compute_missing_points(i, 1)
                    points_layer2 = compute_missing_points(i, 2)
                    points_layer3 = compute_missing_points(i, 3)

                    # set all values the same per layer for directors
                    d1_layer0 = np.repeat([d1_segments[i, 0]], 4, axis=0)
                    d1_layer1 = np.repeat([d1_segments[i, 1]], 4, axis=0)
                    d1_layer2 = np.repeat([d1_segments[i, 2]], 4, axis=0)
                    d1_layer3 = np.repeat([d1_segments[i, 3]], 4, axis=0)
                    d2_layer0 = np.repeat([d2_segments[i, 0]], 4, axis=0)
                    d2_layer1 = np.repeat([d2_segments[i, 1]], 4, axis=0)
                    d2_layer2 = np.repeat([d2_segments[i, 2]], 4, axis=0)
                    d2_layer3 = np.repeat([d2_segments[i, 3]], 4, axis=0)
                    d3_layer0 = np.repeat([d3_segments[i, 0]], 4, axis=0)
                    d3_layer1 = np.repeat([d3_segments[i, 1]], 4, axis=0)
                    d3_layer2 = np.repeat([d3_segments[i, 2]], 4, axis=0)
                    d3_layer3 = np.repeat([d3_segments[i, 3]], 4, axis=0)

                    #######################
                    # 1. vertices (corners)
                    #######################

                    # bottom
                    for j in range(4):
                        vtk_points_weights.append(points_layer0[j])
                        vtk_d1_weights.append(d1_layer0[j])
                        vtk_d2_weights.append(d2_layer0[j])
                        vtk_d3_weights.append(d3_layer0[j])

                    # top
                    for j in range(4):
                        vtk_points_weights.append(points_layer3[j])
                        vtk_d1_weights.append(d1_layer3[j])
                        vtk_d2_weights.append(d2_layer3[j])
                        vtk_d3_weights.append(d3_layer3[j])

                    ##########
                    # 2. edges
                    ##########
                    # first and second
                    # for j in [0, 1, 3, 2]:  # ordering for vtu file version<2.0, e.g. 0.1
                    for j in range(4):  # ordering for vtu file version>=2.0
                        vtk_points_weights.append(points_layer1[j])
                        vtk_points_weights.append(points_layer2[j])
                        vtk_d1_weights.append(d1_layer1[j])
                        vtk_d1_weights.append(d1_layer2[j])
                        vtk_d2_weights.append(d2_layer1[j])
                        vtk_d2_weights.append(d2_layer2[j])
                        vtk_d3_weights.append(d3_layer1[j])
                        vtk_d3_weights.append(d3_layer2[j])

            # points to export is just the R^3 part
            vtk_points_weights = np.array(vtk_points_weights)
            vtk_points = vtk_points_weights[:, :3]

            # the 4th component are the rational weights
            point_data = {
                "RationalWeights": vtk_points_weights[:, 3, None],
            }

            # add other fields for point data
            if isinstance(self.cross_section, CircularCrossSection):
                if circle_as_wedge:
                    point_data["surface_normal"] = vtk_surface_normals

            elif isinstance(self.cross_section, RectangularCrossSection):
                point_data["d1"] = vtk_d1_weights
                point_data["d2"] = vtk_d2_weights
                point_data["d3"] = vtk_d3_weights

            if self._export_dict["stresses"]:
                # TODO: do this on element basis when eval_stresses accepts el as argument
                # TODO: put this then into preprocessor of export
                num = self._export_dict["num_frames"] - 1
                xis = np.linspace(0, 1, num=num)
                xis_e_int = np.linspace(0, 1, self.nelement + 1)[1:-1]
                for xi_e in xis_e_int:
                    assert (
                        xi_e not in xis
                    ), f"xis for fitting may not contain internal element boundaries to represent discontinuities in stresses. \nInternal boundaries at {xis_e_int}, \nxis_fitting={xis}."

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

                ppl = self._export_dict["points_per_layer"]
                # duplicate points for end cap
                vtk_B_n = [B_n_segments[0, 0] for _ in range(ppl)]
                vtk_B_m = [B_m_segments[0, 0] for _ in range(ppl)]

                for i in range(ncells):
                    for layer in range(4):
                        vtk_B_n.extend(
                            np.repeat(
                                [B_n_segments[i, layer]],
                                ppl,
                                axis=0,
                            )
                        )
                        vtk_B_m.extend(
                            np.repeat(
                                [B_m_segments[i, layer]],
                                ppl,
                                axis=0,
                            )
                        )

                # duplicate points for end cap
                vtk_B_n.extend([B_n_segments[-1, -1] for _ in range(ppl)])
                vtk_B_m.extend([B_m_segments[-1, -1] for _ in range(ppl)])

                point_data["B_n"] = vtk_B_n
                point_data["B_m"] = vtk_B_m

            # add cell data
            cell_data = {
                "HigherOrderDegrees": self._export_dict["higher_order_degrees"],
            }

        elif level == "None" or level == None:
            vtk_points = []
            point_data = {}
            cell_data = {}

        return vtk_points, self._export_dict["cells"], point_data, cell_data


class RodExportBaseStress(RodExportBase):

    @abstractmethod
    def eval_stresses(self, t, q, la_c, la_g, xi): ...

    def export(
        self, sol_i, continuity="C1", circle_as_wedge=True, level="volume", **kwargs
    ):

        vtk_points, cells, point_data, cell_data = super().export(
            sol_i,
            continuity=continuity,
            circle_as_wedge=circle_as_wedge,
            level=level,
            **kwargs,
        )

        export_stress = isinstance(self.cross_section, RectangularCrossSection)
        if export_stress:
            B_n_weights, B_m_weights = self.export_stresses(
                sol_i, level=level, circle_as_wedge=circle_as_wedge, **kwargs
            )
            point_data["B_n_old"] = B_n_weights
            point_data["B_m_old"] = B_m_weights
        else:
            warn("Stresses will not be exported: No rectangular cross section.")

        return vtk_points, cells, point_data, cell_data

    def export_stresses(self, sol_i, level="volume", circle_as_wedge=True, **kwargs):

        assert (
            level == "volume"
        ), "Stress export is only implemented for level='volume'."
        assert isinstance(
            self.cross_section, RectangularCrossSection
        ), "Stress export is only implemented for RectangularCrossSection."

        t = sol_i.t
        q = sol_i.q
        la_c = sol_i.la_c
        la_g = sol_i.la_g

        if "num" in kwargs:
            num = kwargs["num"]
        else:
            num = self.nelement * 4

        # TODO: to represent discontinuities in contact forces and moment, xis may not be located at the element boundaries
        B_ns = np.empty((3, num))
        B_ms = np.empty((3, num))
        xis = np.linspace(0, 1, num=num)
        for j in range(num):
            B_ns[:, j], B_ms[:, j] = self.eval_stresses(t, q, la_c, la_g, xis[j])

        if level == "volume":
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

            xis_e_int = np.linspace(0, 1, self.nelement + 1)[1:-1]
            for xi_e in xis_e_int:
                assert (
                    xi_e not in xis
                ), f"xis for fitting may not contain internal element boundaries to represent discontinuities in stresses. \nInternal boundaries at {xis_e_int}, \nxis_fitting={xis}."

            assert num >= 2 * n_segments

            n_cont = "C-1"
            if hasattr(self, "G1_continuity"):
                if not self.G1_continuity:
                    n_cont = "C0"

            # project contact forces and moments on cubic C1 bezier curve
            _, _, B_n_segments = L2_projection_Bezier_curve(
                B_ns.T,
                n_segments,
                case=n_cont,
            )
            _, _, B_m_segments = L2_projection_Bezier_curve(
                B_ms.T,
                n_segments,
                case="C-1",
            )

            if isinstance(self.cross_section, RectangularCrossSection):

                # create correct VTK ordering, see
                # https://coreform.com/papers/implementation-of-rational-bezier-cells-into-VTK-report.pdf:
                vtk_B_n_weights = []
                vtk_B_m_weights = []
                for i in range(n_segments):
                    # set all values the same per layer for contact forces and couples
                    B_n_layer0 = np.repeat([B_n_segments[i, 0]], 4, axis=0)
                    B_n_layer1 = np.repeat([B_n_segments[i, 1]], 4, axis=0)
                    B_n_layer2 = np.repeat([B_n_segments[i, 2]], 4, axis=0)
                    B_n_layer3 = np.repeat([B_n_segments[i, 3]], 4, axis=0)
                    B_m_layer0 = np.repeat([B_m_segments[i, 0]], 4, axis=0)
                    B_m_layer1 = np.repeat([B_m_segments[i, 1]], 4, axis=0)
                    B_m_layer2 = np.repeat([B_m_segments[i, 2]], 4, axis=0)
                    B_m_layer3 = np.repeat([B_m_segments[i, 3]], 4, axis=0)

                    #######################
                    # 1. vertices (corners)
                    #######################

                    # bottom
                    for j in range(4):
                        vtk_B_n_weights.append(B_n_layer0[j])
                        vtk_B_m_weights.append(B_m_layer0[j])

                    # top
                    for j in range(4):
                        vtk_B_n_weights.append(B_n_layer3[j])
                        vtk_B_m_weights.append(B_m_layer3[j])

                    ##########
                    # 2. edges
                    ##########
                    # first and second
                    # for j in [0, 1, 3, 2]:  # ordering for vtu file version<2.0, e.g. 0.1
                    for j in range(4):  # ordering for vtu file version>=2.0
                        vtk_B_n_weights.append(B_n_layer1[j])
                        vtk_B_n_weights.append(B_n_layer2[j])
                        vtk_B_m_weights.append(B_m_layer1[j])
                        vtk_B_m_weights.append(B_m_layer2[j])

        else:
            raise NotImplementedError

        return vtk_B_n_weights, vtk_B_m_weights
