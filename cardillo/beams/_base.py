import numpy as np
from cardillo.math import pi
from abc import ABC, abstractmethod

from cardillo.discretization.bezier import L2_projection_Bezier_curve

# # Timoshenko + Petrov-Galerkin:
# # - *args for rod formulation
# def make_rod(material_model, cross_section, RodFormulation):
#     class Rod:
#         pass

class RodExportBase(ABC):
    def __init__(self):
        self.radius = 0.125
        self.a = 0.1
        self.b = 0.2

        # self.cross_section = "circle"
        # self.cross_section = "circle_wedge"
        self.cross_section = "rectangle"

        # self.num_xi = 100
        # self.num_xi = 20
        # self.num_eta = 3
        # self.num_zeta = 25

        # self.xis = np.linspace(0, 1, num=self.num_xi)
        # self.etas = np.linspace(0, 1, num=self.num_eta)
        # self.zetas = np.linspace(0, 1, num=self.num_zeta)

        # def circular_cross_section(eta, zeta):
        #     return self.radius * np.array(
        #         [
        #             0.0,
        #             eta * np.sin(2 * np.pi * zeta),
        #             eta * np.cos(2 * np.pi * zeta),
        #         ]
        #     )

        # self.cross_section = circular_cross_section

    @abstractmethod
    def r_OP(self, t, q, frame_ID, K_r_SP):
        ...

    @abstractmethod
    def A_IK(self, t, q, frame_ID):
        ...

    # TODO: Move to bezier.py
    @staticmethod
    def line2vtk(target_points, n_segments):
        (
            _,
            _,
            points_segments,
        ) = L2_projection_Bezier_curve(target_points, n_segments, case="C1")

        vtk_points = []
        for i in range(n_segments):
            # VTK ordering, see https://coreform.com/papers/implementation-of-rational-bezier-cells-into-VTK-report.pdf:
            # 1. vertices (corners)
            # 2. edges
            vtk_points.append(points_segments[i, 0])
            vtk_points.append(points_segments[i, -1])
            vtk_points.append(points_segments[i, 1])
            vtk_points.append(points_segments[i, 2])

        return vtk_points

    def export(self, sol_i, **kwargs):
        t = sol_i.t
        q = sol_i.q
        q_body = q[self.qDOF]

        level = kwargs["level"]

        n_segments = self.nelement
        if "num" in kwargs:
            num = kwargs["num"]
        else:
            num = self.nelement * 4
        num_xi = num

        r_OPs, d1s, d2s, d3s = self.frames(q, n=num_xi)

        if level in ["r", "d1", "d2", "d3"]:
            vtk_points = RodExportBase.line2vtk(r_OPs.T, n_segments)
            if level == "r":
                point_data = {}
            elif level == "d1":
                point_data = {"d1": RodExportBase.line2vtk(d1s.T, n_segments)}
            elif level == "d2":
                point_data = {"d1": RodExportBase.line2vtk(d2s.T, n_segments)}
            elif level == "d3":
                point_data = {"d1": RodExportBase.line2vtk(d3s.T, n_segments)}

            cells = [
                ("VTK_BEZIER_CURVE", np.arange(i * 4, (i + 1) * 4)[None])
                for i in range(n_segments)
            ]

            higher_order_degrees = [(np.array([3, 0, 0]),) for _ in range(n_segments)]
            cell_data = {"HigherOrderDegrees": higher_order_degrees}

        if level == "centerline + directors":
            #######################################
            # simple export of points and directors
            #######################################
            r_OPs, d1s, d2s, d3s = self.frames(q, n=num_xi)

            vtk_points = r_OPs.T

            cells = [("line", [[i, i + 1] for i in range(num_xi - 1)])]

            cell_data = {}

            point_data = {
                "d1": d1s.T,
                "d2": d2s.T,
                "d3": d3s.T,
            }

        elif level == 1:

            vtk_points = []
            d1s = []
            d2s = []
            d3s = []
            for xis in self.xis:
                el = self.element_number(xis)
                elDOF = self.elDOF[el]
                r_OP = self.r_OP(t, q_body[elDOF], frame_ID=(xis,))
                A_IK = self.A_IK(t, q_body[elDOF], frame_ID=(xis,))

                for etas in self.etas:
                    for zetas in self.zetas:
                        vtk_points.append(r_OP + A_IK @ self.cross_section(etas, zetas))

            n = len(vtk_points)
            cells = [("vertex", [[i] for i in range(n)])]
            point_data = {}
            cell_data = {}

        elif level == 2:
            ###############################
            # project on cubic Bezier curve
            ###############################
            n_segments = self.nelement

            num_xi = self.nelement * 4
            # target_points_c = self.centerline(q, n=num_xi).T
            r_OPs, d1s, d2s, d3s = self.frames(q, n=num_xi)

            vtk_points = RodExportBase.line2vtk(r_OPs.T, n_segments)
            vtk_points_d1 = RodExportBase.line2vtk(d1s.T, n_segments)
            vtk_points_d2 = RodExportBase.line2vtk(d2s.T, n_segments)
            vtk_points_d3 = RodExportBase.line2vtk(d3s.T, n_segments)

            cells = [
                ("VTK_BEZIER_CURVE", np.arange(i * 4, (i + 1) * 4)[None])
                for i in range(n_segments)
            ]

            higher_order_degrees = [(np.array([3, 0, 0]),) for _ in range(n_segments)]

            point_data = {
                "d1": vtk_points_d1,
                "d2": vtk_points_d2,
                "d3": vtk_points_d3,
            }
            cell_data = {"HigherOrderDegrees": higher_order_degrees}

        elif level == "volume":
            ################################
            # project on cubic Bezier volume
            ################################
            n_segments = self.nelement

            num_xi = self.nelement * 4
            r_OPs, d1s, d2s, d3s = self.frames(q, n=num_xi)
            target_points_centerline = r_OPs.T

            if self.cross_section == "circle":
                target_points_0 = target_points_centerline
                target_points_1 = np.array(
                    [
                        r_OP + d2 * self.radius
                        for i, (r_OP, d2) in enumerate(zip(r_OPs.T, d2s.T))
                    ]
                )
                target_points_2 = np.array(
                    [
                        r_OP + d3 * self.radius
                        for i, (r_OP, d3) in enumerate(zip(r_OPs.T, d3s.T))
                    ]
                )
            elif self.cross_section == "rectangle":
                target_points_0 = target_points_centerline
                target_points_1 = np.array(
                    [
                        r_OP + d2 * self.a
                        for i, (r_OP, d2) in enumerate(zip(r_OPs.T, d2s.T))
                    ]
                )
                target_points_2 = np.array(
                    [
                        r_OP + d3 * self.b
                        for i, (r_OP, d3) in enumerate(zip(r_OPs.T, d3s.T))
                    ]
                )
            elif self.cross_section == "circle_wedge":
                ri = self.radius
                ru = 2 * ri
                a = 2 * np.sqrt(3) * ri

                target_points_0 = np.array(
                    [r_OP - ri * d3 for i, (r_OP, d3) in enumerate(zip(r_OPs.T, d3s.T))]
                )

                target_points_1 = np.array(
                    [
                        r_OP + d2 * a / 2 - ri * d3
                        for i, (r_OP, d2, d3) in enumerate(zip(r_OPs.T, d2s.T, d3s.T))
                    ]
                )

                target_points_2 = np.array(
                    [r_OP + d3 * ru for i, (r_OP, d3) in enumerate(zip(r_OPs.T, d3s.T))]
                )
            else:
                raise NotImplementedError

            # Compute L2-optimal cubic Bezier spline from the given three curves
            # case = "C-1"
            # case = "C0"
            case = "C1"

            # project three points on the cross section
            _, _, points_segments_0 = L2_projection_Bezier_curve(
                target_points_0, n_segments, case=case
            )
            _, _, points_segments_1 = L2_projection_Bezier_curve(
                target_points_1, n_segments, case=case
            )
            _, _, points_segments_2 = L2_projection_Bezier_curve(
                target_points_2, n_segments, case=case
            )

            if self.cross_section == "circle":

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

            elif self.cross_section == "rectangle":

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

            elif self.cross_section == "circle_wedge":

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

            p_zeta = 3
            if self.cross_section == "circle":
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
            elif self.cross_section == "rectangle":
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
            elif self.cross_section == "circle_wedge":
                n_layer = 6
                n_cell = (p_zeta + 1) * n_layer

                higher_order_degrees = [
                    (np.array([2, 2, p_zeta]),) for _ in range(n_segments)
                ]

                cells = [
                    ("VTK_BEZIER_WEDGE", np.arange(i * n_cell, (i + 1) * n_cell)[None])
                    for i in range(n_segments)
                ]

            # TODO: How can we add the centerline and the directors as two
            # different sets of vtk cells?
            if False:
                # project centerline
                _, _, points_segments_centerline = L2_projection_Bezier_curve(
                    target_points_centerline, n_segments, case=case
                )

                # project directors
                _, _, points_segments_d1 = L2_projection_Bezier_curve(
                    d1s.T, n_segments, case=case
                )
                _, _, points_segments_d2 = L2_projection_Bezier_curve(
                    d2s.T, n_segments, case=case
                )
                _, _, points_segments_d3 = L2_projection_Bezier_curve(
                    d3s.T, n_segments, case=case
                )

                def curve2vtk(points):
                    n, dim = points.shape

                    # add rational weights and reshape
                    vtk_points_weights = np.ones((n, dim + 1), dtype=float)
                    vtk_points_weights[0, :3] = points[0]
                    vtk_points_weights[-1, :3] = points[-1]
                    vtk_points_weights[1:-1, :3] = points[1:-1]

                    return vtk_points_weights

                # vtk ordering
                # vtk_points_weights_r = []
                # vtk_points_weights_d1 = []
                # vtk_points_weights_d2 = []
                # vtk_points_weights_d3 = []
                # for i in range(n_segments):
                #     vtk_points_weights_r.extend(curve2vtk(points_segments_centerline[i]))
                #     vtk_points_weights_d1.extend(curve2vtk(points_segments_d1[i]))
                #     vtk_points_weights_d2.extend(curve2vtk(points_segments_d2[i]))
                #     vtk_points_weights_d3.extend(curve2vtk(points_segments_d3[i]))

                offset = cells[-1][1][0, -1]
                for i in range(n_segments):
                    vtk_points_weights.extend(curve2vtk(points_segments_centerline[i]))
                    higher_order_degrees.append((np.array([p_zeta, 0, 0]),))
                    cells.append(
                        (
                            "VTK_BEZIER_CURVE",
                            offset
                            + np.arange(i * (p_zeta + 1), (i + 1) * (p_zeta + 1))[None],
                        )
                    )
                # for i in range(n_segments):
                #     vtk_points_weights.extend(curve2vtk(points_segments_d1[i]))
                #     higher_order_degrees.append((np.array([p_zeta, 0, 0]),))
                # for i in range(n_segments):
                #     vtk_points_weights.extend(curve2vtk(points_segments_d2[i]))
                #     higher_order_degrees.append((np.array([p_zeta, 0, 0]),))
                # for i in range(n_segments):
                #     vtk_points_weights.extend(curve2vtk(points_segments_d3[i]))

            vtk_points_weights = np.array(vtk_points_weights)
            vtk_points = vtk_points_weights[:, :3]

            point_data = {
                # "d1": vtk_points_weights_d1,
                # "d2": vtk_points_weights_d2,
                # "d3": vtk_points_weights_d3,
                "RationalWeights": vtk_points_weights[:, 3],
            }

            cell_data = {
                "HigherOrderDegrees": higher_order_degrees,
            }

        # else:
        #     raise NotImplementedError

        return vtk_points, cells, point_data, cell_data
