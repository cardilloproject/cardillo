from abc import ABC, abstractmethod
import numpy as np
from vtk import VTK_BEZIER_WEDGE, VTK_BEZIER_HEXAHEDRON, VTK_POLY_LINE

from cardillo.utility.bezier import L2_projection_Bezier_curve

from ._cross_section import (
    CrossSection,
    CircularCrossSection,
    RectangularCrossSection,
)


class RodExportBase(ABC):
    def __init__(self, cross_section: CrossSection):
        self.cross_section = cross_section

    @abstractmethod
    def r_OP(self, t, q, xi, B_r_CP): ...

    @abstractmethod
    def A_IB(self, t, q, xi): ...

    def centerline(self, q, num=100):
        q_body = q[self.qDOF]
        r = []
        for xi in np.linspace(0, 1, num):
            xi = (xi,)
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
            xi = (xi,)
            qp = q_body[self.local_qDOF_P(xi)]
            r.append(self.r_OP(1, qp, xi))

            d1i, d2i, d3i = self.A_IB(1, qp, xi).T
            d1.extend([d1i])
            d2.extend([d2i])
            d3.extend([d3i])

        return np.array(r).T, np.array(d1).T, np.array(d2).T, np.array(d3).T

    def export(
        self, sol_i, continuity="C1", circle_as_wedge=True, level="volume", **kwargs
    ):
        q = sol_i.q

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

            cells = [(VTK_POLY_LINE, list(range(num)))]

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

            assert num >= 2 * n_segments

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
                        r_OP + d2 * self.cross_section.width / 2
                        for (r_OP, d2) in zip(r_OPs.T, d2s.T)
                    ]
                )
                target_points_2 = np.array(
                    [
                        r_OP + d3 * self.cross_section.height / 2
                        for (r_OP, d3) in zip(r_OPs.T, d3s.T)
                    ]
                )
            else:
                raise NotImplementedError

            # project target points on cubic C1 Bézier curve
            _, _, points_segments_0 = L2_projection_Bezier_curve(
                target_points_0, n_segments, case=continuity
            )
            _, _, points_segments_1 = L2_projection_Bezier_curve(
                target_points_1, n_segments, case=continuity
            )
            _, _, points_segments_2 = L2_projection_Bezier_curve(
                target_points_2, n_segments, case=continuity
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
                    # for j in [0, 1, 3, 2]:  # ordering for vtu file version<2.0, e.g. 0.1
                    for j in range(4):  # ordering for vtu file version>=2.0
                        vtk_points_weights.append(points_layer1[j])
                        vtk_points_weights.append(points_layer2[j])

            p_zeta = 3
            if isinstance(self.cross_section, CircularCrossSection):
                if circle_as_wedge:
                    n_layer = 6
                    n_cell = (p_zeta + 1) * n_layer

                    higher_order_degrees = np.array([[2, 2, p_zeta]] * n_segments)

                    cells = [
                        (
                            VTK_BEZIER_WEDGE,
                            np.arange(i * n_cell, (i + 1) * n_cell),
                        )
                        for i in range(n_segments)
                    ]
                else:
                    n_layer = 9
                    n_cell = (p_zeta + 1) * n_layer

                    higher_order_degrees = np.array([[2, 2, p_zeta]] * n_segments)

                    cells = [
                        (
                            VTK_BEZIER_HEXAHEDRON,
                            np.arange(i * n_cell, (i + 1) * n_cell),
                        )
                        for i in range(n_segments)
                    ]
            if isinstance(self.cross_section, RectangularCrossSection):
                n_layer = 4
                n_cell = (p_zeta + 1) * n_layer

                higher_order_degrees = np.array([[1, 1, p_zeta]] * n_segments)

                cells = [
                    (
                        VTK_BEZIER_HEXAHEDRON,
                        np.arange(i * n_cell, (i + 1) * n_cell),
                    )
                    for i in range(n_segments)
                ]

            vtk_points_weights = np.array(vtk_points_weights)
            vtk_points = vtk_points_weights[:, :3]

            if isinstance(self.cross_section, RectangularCrossSection):
                point_data = {
                    "RationalWeights": vtk_points_weights[:, 3, None],
                }
            else:
                point_data = {
                    "RationalWeights": vtk_points_weights[:, 3, None],
                }

            cell_data = {
                "HigherOrderDegrees": higher_order_degrees,
            }

        else:
            raise NotImplementedError

        return vtk_points, cells, point_data, cell_data
