import numpy as np
import vtk
from cardillo.rods import CircularCrossSection, RectangularCrossSection
from cardillo.utility.bezier import L2_projection_Bezier_curve


def __decorate_vtk_source(
    contr, source, A_BM=np.eye(3), B_r_CP=np.zeros(3), color=(255, 255, 255)
):
    if not hasattr(contr, "actors"):
        contr.actors = []
    if not hasattr(contr, "H_IB"):
        contr.H_IB = vtk.vtkMatrix4x4()
        contr.H_IB.Identity()
    if not hasattr(contr, "ugrid"):
        contr.ugrid = vtk.vtkUnstructuredGrid()
    if not hasattr(contr, "step_render"):

        def step_render(t, q, u):
            A_IB = contr.A_IB(t, q)
            r_OP = contr.r_OP(t, q)[:, None]
            for i in range(3):
                for j in range(3):
                    contr.H_IB.SetElement(i, j, A_IB[i, j])
                contr.H_IB.SetElement(i, 3, r_OP[i])

        contr.step_render = step_render
    H_BM = np.block(
        [
            [A_BM, B_r_CP[:, None]],
            [0, 0, 0, 1],
        ]
    )
    _H_IB = vtk.vtkMatrixToLinearTransform()
    _H_IB.SetInput(contr.H_IB)
    _H_IM = vtk.vtkTransform()
    _H_IM.PostMultiply()
    _H_IM.SetMatrix(H_BM.flatten())
    _H_IM.Concatenate(_H_IB)
    tf_filter = vtk.vtkTransformPolyDataFilter()
    tf_filter.SetInputConnection(source.GetOutputPort())
    tf_filter.SetTransform(_H_IM)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(tf_filter.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(np.array(color, float) / 255)
    # base_actor.GetProperty().SetOpacity(0.2)
    contr.actors.append(actor)


def __decorate_vtk_ugrid(
    contr, ugrid, A_BM=np.eye(3), B_r_CP=np.zeros(3), color=(255, 255, 255)
):
    if not hasattr(contr, "actors"):
        contr.actors = []
    if not hasattr(contr, "H_IB"):
        contr.H_IB = vtk.vtkMatrix4x4()
        contr.H_IB.Identity()
    if not hasattr(contr, "step_render"):

        def step_render(t, q, u):
            A_IB = contr.A_IB(t, q)
            r_OP = contr.r_OP(t, q)[:, None]
            for i in range(3):
                for j in range(3):
                    contr.H_IB.SetElement(i, j, A_IB[i, j])
                contr.H_IB.SetElement(i, 3, r_OP[i])

        contr.step_render = step_render
    H_BM = np.block(
        [
            [A_BM, B_r_CP[:, None]],
            [0, 0, 0, 1],
        ]
    )
    _H_IB = vtk.vtkMatrixToLinearTransform()
    _H_IB.SetInput(contr.H_IB)
    _H_IM = vtk.vtkTransform()
    _H_IM.PostMultiply()
    _H_IM.SetMatrix(H_BM.flatten())
    _H_IM.Concatenate(_H_IB)
    tf_filter = vtk.vtkTransformFilter()
    tf_filter.SetInputData(ugrid)
    tf_filter.SetTransform(_H_IM)

    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputConnection(tf_filter.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(np.array(color, float) / 255)
    # base_actor.GetProperty().SetOpacity(0.2)
    contr.actors.append(actor)


def decorate_box(
    contr,
    dimensions=np.ones(3),
    A_BM=np.eye(3),
    B_r_CP=np.zeros(3),
    color=(255, 255, 255),
):
    source = vtk.vtkCubeSource()
    source.SetXLength(dimensions[0])
    source.SetYLength(dimensions[1])
    source.SetZLength(dimensions[2])
    __decorate_vtk_source(contr, source, A_BM, B_r_CP, color)


def decorate_cone(
    contr,
    radius=1,
    height=2,
    resolution=30,
    A_BM=np.eye(3),
    B_r_CP=np.zeros(3),
    color=(255, 255, 255),
):
    source = vtk.vtkConeSource()
    source.SetRadius(radius)
    source.SetHeight(height)
    source.SetResolution(resolution)
    source.SetDirection(0, 0, 1)
    source.SetCenter(0, 0, height / 4)
    __decorate_vtk_source(contr, source, A_BM, B_r_CP, color)


def decorate_cylinder(
    contr,
    radius=1,
    height=2,
    resolution=30,
    A_BM=np.eye(3),
    B_r_CP=np.zeros(3),
    color=(255, 255, 255),
):
    source = vtk.vtkCylinderSource()
    source.SetRadius(radius)
    source.SetHeight(height)
    source.SetResolution(resolution)
    __decorate_vtk_source(contr, source, A_BM, B_r_CP, color)


def decorate_sphere(
    contr,
    radius=1,
    resolution=30,
    A_BM=np.eye(3),
    B_r_CP=np.zeros(3),
    color=(255, 255, 255),
):
    source = vtk.vtkSphereSource()
    source.SetRadius(radius)
    source.SetPhiResolution(int(resolution / 2 - 1))
    source.SetThetaResolution(resolution)
    __decorate_vtk_source(contr, source, A_BM, B_r_CP, color)


def decorate_capsule(
    contr,
    radius=1,
    height=2,
    resolution=30,
    A_BM=np.eye(3),
    B_r_CP=np.zeros(3),
    color=(255, 255, 255),
):
    source = vtk.vtkCylinderSource()
    source.SetRadius(radius)
    source.SetHeight(height)
    source.SetResolution(resolution)
    source.CapsuleCapOn()
    __decorate_vtk_source(contr, source, A_BM, B_r_CP, color)


def decorate_tetrahedron(
    contr, edge=1, A_BM=np.eye(3), B_r_CP=np.zeros(3), color=(255, 255, 255)
):
    # see https://de.wikipedia.org/wiki/Tetraeder
    h_D = edge * np.sqrt(3) / 2
    h_P = edge * np.sqrt(2 / 3)
    r_OM = np.array([0, h_D / 3, h_P / 4])
    p1 = np.array([-edge / 2, 0, 0]) - r_OM
    p2 = np.array([+edge / 2, 0, 0]) - r_OM
    p3 = np.array([0, h_D, 0]) - r_OM
    p4 = np.array([0, h_D / 3, h_P]) - r_OM

    points = vtk.vtkPoints()
    points.InsertNextPoint(*p1)
    points.InsertNextPoint(*p2)
    points.InsertNextPoint(*p3)
    points.InsertNextPoint(*p4)

    # The first tetrahedron
    ugrid = vtk.vtkUnstructuredGrid()
    ugrid.SetPoints(points)

    tetra = vtk.vtkTetra()

    tetra.GetPointIds().SetId(0, 0)
    tetra.GetPointIds().SetId(1, 1)
    tetra.GetPointIds().SetId(2, 2)
    tetra.GetPointIds().SetId(3, 3)

    cellArray = vtk.vtkCellArray()
    cellArray.InsertNextCell(tetra)
    ugrid.SetCells(vtk.VTK_TETRA, cellArray)

    __decorate_vtk_ugrid(contr, ugrid, A_BM, B_r_CP, color)


def decorate_axis(
    contr,
    length,
    resolution=30,
    A_BM=np.eye(3),
    B_r_CP=np.zeros(3),
    color=(255, 255, 255),
):
    source = vtk.vtkArrowSource()
    source.SetTipResolution(resolution)
    source.SetShaftResolution(resolution)
    __decorate_vtk_source(contr, source, A_BM * length, B_r_CP, color)


def decorate_coordinate_system(
    contr, length, resolution=30, A_BM=np.eye(3), B_r_CP=np.zeros(3)
):
    for i in range(3):
        if i == 0:
            color = (255, 0, 0)
        elif i == 1:
            A_BM = A_BM @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
            color = (0, 255, 0)
        elif i == 2:
            A_BM = A_BM @ np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
            color = (0, 0, 255)
        decorate_axis(contr, length, A_BM=A_BM, B_r_CP=B_r_CP, color=color)


def decorate_stl(
    contr,
    stl_file,
    scale=1e-3,
    A_BM=np.eye(3),
    B_r_CP=np.zeros(3),
    color=(255, 255, 255),
):
    source = vtk.vtkSTLReader()
    source.SetFileName(stl_file)
    source.Update()
    __decorate_vtk_source(contr, source, A_BM * scale, B_r_CP, color)


def decorate_rod(contr, nelement_visual=1, subdivision=3):
    def bezier_volume_projection(q, case="C1"):
        ################################
        # project on cubic Bezier volume
        ################################
        r = []
        d2 = []
        d3 = []

        num = nelement_visual * 4
        for xi in np.linspace(0, 1, num):
            xi = (xi,)
            qp = q[contr.local_qDOF_P(xi)]
            r.append(contr.r_OP(1, qp, xi))

            _, d2i, d3i = contr.A_IB(1, qp, xi).T
            d2.extend([d2i])
            d3.extend([d3i])

        r_OPs, d2s, d3s = np.array(r).T, np.array(d2).T, np.array(d3).T
        target_points_centerline = r_OPs.T

        # create points of the target curves (three characteristic points
        # of the cross section)
        if isinstance(contr.cross_section, CircularCrossSection):
            ri = contr.cross_section.radius
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
        elif isinstance(contr.cross_section, RectangularCrossSection):
            target_points_0 = target_points_centerline
            target_points_1 = np.array(
                [
                    r_OP + d2 * contr.cross_section.width / 2
                    for (r_OP, d2) in zip(r_OPs.T, d2s.T)
                ]
            )
            target_points_2 = np.array(
                [
                    r_OP + d3 * contr.cross_section.height / 2
                    for (r_OP, d3) in zip(r_OPs.T, d3s.T)
                ]
            )
        else:
            raise NotImplementedError

        # project target points on cubic C1 Bézier curve
        _, _, points_segments_0 = L2_projection_Bezier_curve(
            target_points_0, nelement_visual, case=case
        )
        _, _, points_segments_1 = L2_projection_Bezier_curve(
            target_points_1, nelement_visual, case=case
        )
        _, _, points_segments_2 = L2_projection_Bezier_curve(
            target_points_2, nelement_visual, case=case
        )

        if isinstance(contr.cross_section, CircularCrossSection):

            def compute_missing_points(segment, layer):
                P0 = points_segments_0[segment, layer]
                P3 = points_segments_1[segment, layer]
                P4 = points_segments_2[segment, layer]

                P5 = 2 * P0 - P3
                P1 = 0.5 * (P3 + P4)
                P0 = 0.5 * (P5 + P3)
                P2 = 0.5 * (P4 + P5)
                return np.array([P0, P1, P2, P3, P4, P5])

            # create correct VTK ordering, see
            # https://coreform.com/papers/implementation-of-rational-bezier-cells-into-VTK-report.pdf:
            vtk_points = []
            for i in range(nelement_visual):
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
                    vtk_points.append(points_layer0[j])

                # top
                for j in range(3):
                    vtk_points.append(points_layer3[j])

                ##########
                # 2. edges
                ##########

                # bottom
                for j in range(3, 6):
                    vtk_points.append(points_layer0[j])

                # top
                for j in range(3, 6):
                    vtk_points.append(points_layer3[j])

                # first and second
                for j in range(3):
                    vtk_points.append(points_layer1[j])
                    vtk_points.append(points_layer2[j])

                ##########
                # 3. faces
                ##########

                # first and second
                for j in range(3, 6):
                    vtk_points.append(points_layer1[j])
                    vtk_points.append(points_layer2[j])

        elif isinstance(contr.cross_section, RectangularCrossSection):

            def compute_missing_points(segment, layer):
                Q0 = points_segments_0[segment, layer]
                Q1 = points_segments_1[segment, layer]
                Q2 = points_segments_2[segment, layer]
                P0 = Q0 - (Q2 - Q0) - (Q1 - Q0)
                P1 = Q0 - (Q2 - Q0) + (Q1 - Q0)
                P2 = Q0 + (Q2 - Q0) + (Q1 - Q0)
                P3 = Q0 + (Q2 - Q0) - (Q1 - Q0)

                return np.array([P0, P1, P2, P3])

            vtk_points = []
            for i in range(nelement_visual):
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
                    vtk_points.append(points_layer0[j])
                # top
                for j in range(4):
                    vtk_points.append(points_layer3[j])

                ##########
                # 2. edges
                ##########
                # first and second
                # for j in [0, 1, 3, 2]:  # ordering for vtu file version<2.0, e.g. 0.1
                for j in range(4):  # ordering for vtu file version>=2.0
                    vtk_points.append(points_layer1[j])
                    vtk_points.append(points_layer2[j])

        return np.array(vtk_points)

    def init_visualization(nelement_visual, subdivision):
        contr.actors = []
        if isinstance(contr.cross_section, CircularCrossSection):
            npts = 24
            weights = [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
            ]
            degrees = [2, 2, 3]
            ctype = vtk.VTK_BEZIER_WEDGE
        elif isinstance(contr.cross_section, RectangularCrossSection):
            npts = 16
            weights = [1] * 16
            degrees = [1, 1, 3]
            ctype = vtk.VTK_BEZIER_HEXAHEDRON
        else:
            raise NotImplementedError

        ugrid = vtk.vtkUnstructuredGrid()

        # points
        contr.vtkpoints = vtk.vtkPoints()
        contr.vtkpoints.SetNumberOfPoints(npts * nelement_visual)
        ugrid.SetPoints(contr.vtkpoints)

        # cells
        ugrid.Allocate(nelement_visual)
        for i in range(nelement_visual):
            ugrid.InsertNextCell(ctype, npts, list(range(i * npts, (i + 1) * npts)))

        # point data
        pdata = ugrid.GetPointData()
        value = weights * nelement_visual
        parray = vtk.vtkDoubleArray()
        parray.SetName("RationalWeights")
        parray.SetNumberOfTuples(npts)
        parray.SetNumberOfComponents(1)
        for i, vi in enumerate(value):
            parray.InsertTuple(i, [vi])
        pdata.SetRationalWeights(parray)

        # cell data
        cdata = ugrid.GetCellData()
        carray = vtk.vtkIntArray()
        carray.SetName("HigherOrderDegrees")
        carray.SetNumberOfTuples(nelement_visual)
        carray.SetNumberOfComponents(3)
        for i in range(nelement_visual):
            carray.InsertTuple(i, degrees)
        cdata.SetHigherOrderDegrees(carray)

        filter = vtk.vtkDataSetSurfaceFilter()
        filter.SetInputData(ugrid)
        filter.SetNonlinearSubdivisionLevel(subdivision)

        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputConnection(filter.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(82 / 255, 108 / 255, 164 / 255)
        # actor.GetProperty().SetOpacity(0.2)
        contr.actors.append(actor)

    init_visualization(nelement_visual, subdivision)

    def step_render(t, q, u):
        points = bezier_volume_projection(q)
        for i, p in enumerate(points):
            contr.vtkpoints.SetPoint(i, p)
        contr.vtkpoints.Modified()

    contr.step_render = step_render
