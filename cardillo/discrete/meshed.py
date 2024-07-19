import numpy as np
import trimesh
from abc import ABC
import vtk
from vtk import VTK_TRIANGLE
from cardillo.visualization.vtk_export import make_ugrid


def _check_density_consistency(mass, B_Theta_C, kwargs):
    mass_arg = kwargs.pop("mass", None)
    B_Theta_C_arg = kwargs.pop("B_Theta_C", None)

    if (mass_arg is not None) and (not np.allclose(mass, mass_arg)):
        print("Specified mass does not correspond to mass of mesh.")
    if (B_Theta_C_arg is not None) and (not np.allclose(B_Theta_C, B_Theta_C_arg)):
        print(
            "Specified moment of inertia does not correspond to moment of inertia of mesh."
        )


def Meshed(Base):
    """Generate an object (typically with Base `Frame` or `RigidBody`)
    from a given Trimesh object.

    Parameters
    ----------
    Base :  object
        Cardillo object Frame, RigidBody or Pointmass

    Returns
    -------
    out : object
        Meshed version of Base object

    """

    class _Meshed(MeshedVisual, Base):
        def __init__(
            self,
            mesh_obj,
            density=None,
            B_r_CP=np.zeros(3),
            A_BM=np.eye(3),
            scale=1,
            color=(255, 255, 255),
            **kwargs,
        ):
            """Generate an object (typically with Base `Frame` or `RigidBody`)
            from a given Trimesh object.

            Parameters
            ----------
            mesh_obj :
                File-like object defining source of mesh or instance of trimesh
                defining the mesh
            Density : float or None
                Mass density for the computation of the inertia properties of the
                mesh. If set to None, user specified mass and B_Theta_C are used.
            B_r_CP : np.ndarray (3,)
                Offset center of mass (C) from (C)TL origin (P) in body fixed K-basis.
            A_BM: np.ndarray (3, 3)
                Tansformation from mesh-fixed basis (M) to body-fixed basis (K).
            scale: float
                Factor scaling the mesh after import.
            kwargs: dict,
                Arguments of parent class (Base) as keyword arguments
            """
            self.B_r_CP = B_r_CP
            self.A_BM = A_BM

            #############################
            # consistency checks for mesh
            #############################
            if isinstance(mesh_obj, trimesh.Trimesh):
                trimesh_obj = mesh_obj

                # primitives are converted to mesh
                if hasattr(trimesh_obj, "to_mesh"):
                    trimesh_mesh = trimesh_obj.to_mesh()
                else:
                    trimesh_mesh = trimesh_obj
                self.file_obj = None
            else:
                self.file_obj = mesh_obj
                trimesh_mesh = trimesh.load_mesh(mesh_obj)

            trimesh_mesh.apply_transform(np.diag([scale, scale, scale, 1]))

            # check if mesh represents a valid volume
            if not trimesh_mesh.is_volume:
                print(
                    "Imported mesh does not represent a volume, i.e. one of the following properties are not fulfilled: watertight, consistent winding, outward facing normals."
                )
                # try to fill the wholes
                trimesh_mesh.fill_holes()
                if not trimesh_mesh.is_volume:
                    print(
                        "Using mesh that is not a volume. Computed mass and moment of inertia might be unphyical."
                    )
                else:
                    print("Fixed mesh by filling the holes.")

            # store visual mesh in body fixed basis
            H_KM = np.eye(4)
            H_KM[:3, 3] = B_r_CP
            H_KM[:3, :3] = A_BM
            self.B_visual_mesh = trimesh_mesh.copy().apply_transform(H_KM)

            # vectors (transposed) from (C) to vertices (Qi) represented in body-fixed basis
            self.B_r_CQi_T = self.B_visual_mesh.vertices.view(np.ndarray).T

            # compute inertia quantities of body
            if density is not None:
                # set density and compute properties
                self.B_visual_mesh.density = density
                mass = self.B_visual_mesh.mass
                B_Theta_C = self.B_visual_mesh.moment_inertia
                _check_density_consistency(mass, B_Theta_C, kwargs)
                kwargs.update({"mass": mass, "B_Theta_C": B_Theta_C})

            Base.__init__(self, **kwargs)

            if self.file_obj is not None:
                source = vtk.vtkSTLReader()
                source.SetFileName(mesh_obj)
                source.Update()
                MeshedVisual.__init__(
                    self,
                    [source],
                    A_BM_list=[A_BM * scale],
                    B_r_CP_list=[B_r_CP],
                    color_list=[color],
                )

        def __getattribute__(self, name: str):
            if name == "step_render" and self.file_obj is None:
                raise AttributeError

            return super().__getattribute__(name)

        def export(self, sol_i, base_export=False, **kwargs):
            if base_export:
                return Base.export(self, sol_i, **kwargs)
            else:
                r_OC = self.r_OP(
                    sol_i.t, sol_i.q[self.qDOF]
                )  # TODO: Idea: slicing could be done on global level in Export class. Moreover, solution class should be able to return the slice, e.g., sol_i.get_q_of_body(name).
                A_IB = self.A_IB(sol_i.t, sol_i.q[self.qDOF])
                points = (r_OC[:, None] + A_IB @ self.B_r_CQi_T).T

                cells = [(VTK_TRIANGLE, face) for face in self.B_visual_mesh.faces]

            return points, cells, None, None

    return _Meshed


class MeshedVisual(ABC):
    def __init__(
        self, source_list, A_BM_list=None, B_r_CP_list=None, color_list=None
    ) -> None:
        self.actors = []
        self.H_IB = vtk.vtkMatrix4x4()
        self.H_IB.Identity()
        if A_BM_list is None:
            A_BM_list = [np.eye(3)] * len(source_list)
        if B_r_CP_list is None:
            B_r_CP_list = [np.zeros(3)] * len(source_list)
        if color_list is None:
            color_list = [(255, 255, 255)] * len(source_list)

        self.appendfilter = vtk.vtkAppendFilter()
        for source, A_BM, B_r_CP, color in zip(
            source_list, A_BM_list, B_r_CP_list, color_list
        ):
            H_BM = np.block(
                [
                    [A_BM, B_r_CP[:, None]],
                    [0, 0, 0, 1],
                ]
            )
            H_IB = vtk.vtkMatrixToLinearTransform()
            H_IB.SetInput(self.H_IB)
            H_IM = vtk.vtkTransform()
            H_IM.PostMultiply()
            H_IM.SetMatrix(H_BM.flatten())
            H_IM.Concatenate(H_IB)
            tf_filter = vtk.vtkTransformPolyDataFilter()
            tf_filter.SetInputConnection(source.GetOutputPort())
            tf_filter.SetTransform(H_IM)

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(tf_filter.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(np.array(color, float) / 255)
            # base_actor.GetProperty().SetOpacity(0.2)
            self.actors.append(actor)
            self.appendfilter.AddInputConnection(tf_filter.GetOutputPort())
        self.ugrid = vtk.vtkUnstructuredGrid()

    def step_render(self, t, q, u):
        A_IB = self.A_IB(t, q)
        r_OP = self.r_OP(t, q)[:, None]
        for i in range(3):
            for j in range(3):
                self.H_IB.SetElement(i, j, A_IB[i, j])
            self.H_IB.SetElement(i, 3, r_OP[i])
        self.appendfilter.Update()
        self.ugrid.ShallowCopy(self.appendfilter.GetOutput())

    def export(self, sol_i, base_export=False, **kwargs):
        if base_export:
            return super().export(sol_i, **kwargs)
        else:
            self.step_render(sol_i.t, sol_i.q[self.qDOF], sol_i.u[self.uDOF])
            return self.ugrid


def Box(Base):
    class _Box(MeshedVisual, Base):
        def __init__(
            self,
            dimensions=np.ones(3),
            density=None,
            color=(255, 255, 255),
            **kwargs,
        ):
            if density is not None:
                mass = density * dimensions[0] * dimensions[1] * dimensions[2]
                B_Theta_C = (
                    np.diag(
                        [
                            dimensions[1] ** 2 + dimensions[2] ** 2,
                            dimensions[0] ** 2 + dimensions[2] ** 2,
                            dimensions[0] ** 2 + dimensions[1] ** 2,
                        ]
                    )
                    * mass
                    / 12
                )
                _check_density_consistency(mass, B_Theta_C, kwargs)
                kwargs.update({"mass": mass, "B_Theta_C": B_Theta_C})
            Base.__init__(self, **kwargs)

            source = vtk.vtkCubeSource()
            source.SetXLength(dimensions[0])
            source.SetYLength(dimensions[1])
            source.SetZLength(dimensions[2])
            MeshedVisual.__init__(self, [source], color_list=[color])

    return _Box


def Cone(Base):
    class _Cone(MeshedVisual, Base):
        def __init__(
            self,
            radius=1,
            height=2,
            density=None,
            resolution=30,
            color=(255, 255, 255),
            **kwargs,
        ):
            self.radius = radius
            self.height = height
            if density is not None:
                mass = density / 3 * height * np.pi * radius**2
                B_Theta_C = (
                    np.diag(
                        [
                            0.15 * radius**2 + 0.1 * height**2,
                            0.15 * radius**2 + 0.1 * height**2,
                            0.3 * radius**2,
                        ]
                    )
                    * mass
                )
                _check_density_consistency(mass, B_Theta_C, kwargs)
                kwargs.update({"mass": mass, "B_Theta_C": B_Theta_C})
            Base.__init__(self, **kwargs)

            source = vtk.vtkConeSource()
            source.SetRadius(radius)
            source.SetHeight(height)
            source.SetResolution(resolution)
            source.SetDirection(0, 0, 1)
            source.SetCenter(0, 0, height / 4)
            MeshedVisual.__init__(self, [source], color_list=[color])

    return _Cone


def Cylinder(Base):
    class _Cylinder(MeshedVisual, Base):
        def __init__(
            self,
            radius=1,
            height=2,
            density=None,
            resolution=30,
            color=(255, 255, 255),
            **kwargs,
        ):
            self.radius = radius
            self.height = height
            if density is not None:
                mass = density * height * np.pi * radius**2
                B_Theta_C = (
                    np.diag(
                        [
                            0.25 * radius**2 + 1 / 12 * height**2,
                            0.25 * radius**2 + 1 / 12 * height**2,
                            0.5 * radius**2,
                        ]
                    )
                    * mass
                )
                _check_density_consistency(mass, B_Theta_C, kwargs)
                kwargs.update({"mass": mass, "B_Theta_C": B_Theta_C})
            Base.__init__(self, **kwargs)

            source = vtk.vtkCylinderSource()
            source.SetRadius(radius)
            source.SetHeight(height)
            source.SetResolution(resolution)
            MeshedVisual.__init__(
                self,
                [source],
                A_BM_list=[np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])],
                color_list=[color],
            )

    return _Cylinder


def Sphere(Base):
    class _Sphere(MeshedVisual, Base):
        def __init__(
            self,
            radius=1,
            density=None,
            resolution=30,
            color=(255, 255, 255),
            **kwargs,
        ):
            if density is not None:
                mass = density * 4 / 3 * np.pi * radius**3
                B_Theta_C = np.eye(3) * 2 / 5 * mass * radius**2
                _check_density_consistency(mass, B_Theta_C, kwargs)
                kwargs.update({"mass": mass, "B_Theta_C": B_Theta_C})
            Base.__init__(self, **kwargs)

            source = vtk.vtkSphereSource()
            source.SetRadius(radius)
            source.SetPhiResolution(int(resolution / 2 - 1))
            source.SetThetaResolution(resolution)
            MeshedVisual.__init__(self, [source], color_list=[color])

    return _Sphere


def Capsule(Base):

    class _Capsule(MeshedVisual, Base):
        def __init__(
            self,
            radius=1,
            height=2,
            density=None,
            resolution=30,
            color=(255, 255, 255),
            **kwargs,
        ):
            self.radius = radius
            self.height = height
            if density is not None:
                # https://www.gamedev.net/tutorials/programming/math-and-physics/capsule-inertia-tensor-r3856/
                m_cyl = density * height * np.pi * radius**2
                m_cap = density * 2 / 3 * np.pi * radius**3
                mass = m_cyl + 2 * m_cap
                B_Theta_C = (
                    np.diag(
                        [
                            0.25 * radius**2 + 1 / 12 * height**2,
                            0.25 * radius**2 + 1 / 12 * height**2,
                            0.5 * radius**2,
                        ]
                    )
                    * m_cyl
                    + np.diag(
                        [
                            0.4 * radius**2 + 0.5 * height**2 + 3 / 8 * height * radius,
                            0.4 * radius**2 + 0.5 * height**2 + 3 / 8 * height * radius,
                            0.4 * radius**2,
                        ]
                    )
                    * 2
                    * m_cap
                )
                _check_density_consistency(mass, B_Theta_C, kwargs)
                kwargs.update({"mass": mass, "B_Theta_C": B_Theta_C})
            Base.__init__(self, **kwargs)

            source = vtk.vtkCylinderSource()
            source.SetRadius(radius)
            source.SetHeight(height)
            source.SetResolution(resolution)
            source.CapsuleCapOn()
            MeshedVisual.__init__(
                self,
                [source],
                A_BM_list=[np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])],
                color_list=[color],
            )

    return _Capsule


def Tetrahedron(Base):
    MeshedBase = Meshed(Base)

    class _Tetrahedron(MeshedBase):
        def __init__(
            self,
            edge=1,
            **kwargs,
        ):
            # see https://de.wikipedia.org/wiki/Tetraeder
            h_D = edge * np.sqrt(3) / 2
            h_P = edge * np.sqrt(2 / 3)
            r_OM = np.array([0, h_D / 3, h_P / 4])
            p1 = np.array([-edge / 2, 0, 0]) - r_OM
            p2 = np.array([+edge / 2, 0, 0]) - r_OM
            p3 = np.array([0, h_D, 0]) - r_OM
            p4 = np.array([0, h_D / 3, h_P]) - r_OM
            vertices = np.vstack((p1, p2, p3, p4))

            faces = np.array([[0, 1, 3], [1, 2, 3], [2, 0, 3], [0, 2, 1]])

            trimesh_obj = trimesh.Trimesh(vertices, faces)
            super().__init__(mesh_obj=trimesh_obj, **kwargs)

    return _Tetrahedron


def Axis(Base):

    class _Axis(MeshedVisual, Base):
        def __init__(
            self,
            origin_size=0.04,
            resolution=30,
            **kwargs,
        ):
            Base.__init__(self, **kwargs)
            source = vtk.vtkArrowSource()
            source.SetTipResolution(resolution)
            source.SetShaftResolution(resolution)
            A_BM_list = []
            source_list = [source] * 3
            color_list = []
            for i in range(3):
                if i == 0:
                    A_BM = np.eye(3)
                    c = (255, 0, 0)
                elif i == 1:
                    A_BM = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
                    c = (0, 255, 0)
                elif i == 2:
                    A_BM = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
                    c = (0, 0, 255)
                A_BM_list.append(A_BM * origin_size)
                color_list.append(c)
            MeshedVisual.__init__(
                self, source_list, A_BM_list=A_BM_list, color_list=color_list
            )

    return _Axis


if __name__ == "__main__":
    pass
