import numpy as np
import trimesh
from cardillo.visualization.vtk_export import make_ugrid
from vtk import (
    VTK_TRIANGLE,
    vtkTransform,
    vtkActor,
    vtkTransformFilter,
    vtkDataSetMapper,
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

    class _Meshed(Base):
        def __init__(
            self,
            mesh_obj,
            density=None,
            B_r_CP=np.zeros(3),
            A_BM=np.eye(3),
            scale=1,
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
            else:
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

                mass_arg = kwargs.pop("mass", None)
                B_Theta_C_arg = kwargs.pop("B_Theta_C", None)

                if (mass_arg is not None) and (not np.allclose(mass, mass_arg)):
                    print("Specified mass does not correspond to mass of mesh.")
                if (B_Theta_C_arg is not None) and (
                    not np.allclose(B_Theta_C, B_Theta_C_arg)
                ):
                    print(
                        "Specified moment of inertia does not correspond to moment of inertia of mesh."
                    )

                kwargs.update({"mass": mass, "B_Theta_C": B_Theta_C})

            super().__init__(**kwargs)

        def export(self, sol_i, base_export=False, **kwargs):
            if base_export:
                return super().export(sol_i, **kwargs)
            else:
                r_OC = self.r_OP(
                    sol_i.t, sol_i.q[self.qDOF]
                )  # TODO: Idea: slicing could be done on global level in Export class. Moreover, solution class should be able to return the slice, e.g., sol_i.get_q_of_body(name).
                A_IB = self.A_IB(sol_i.t, sol_i.q[self.qDOF])
                points = (r_OC[:, None] + A_IB @ self.B_r_CQi_T).T

                cells = [(VTK_TRIANGLE, face) for face in self.B_visual_mesh.faces]

            return points, cells, None, None

        def update_vtk_tf(self, sol_i):
            A_IB = self.A_IB(sol_i.t, sol_i.q, sol_i.u)
            r_OP = self.r_OP(sol_i.t, sol_i.q, sol_i.u)[:, None]
            if not hasattr(self, "vtk_tf"):
                points, cells, point_data, cell_data = self.export(sol_i)
                ugrid = make_ugrid(points, cells, point_data, cell_data)
                self.vtk_tf = vtkTransform()
                tf0 = vtkTransform()
                tf0.SetMatrix(
                    np.block([[A_IB.T, -A_IB.T @ r_OP], [0, 0, 0, 1]]).flatten()
                )
                tf0.PostMultiply()
                tf0.Concatenate(self.vtk_tf)
                tf_filter = vtkTransformFilter()
                tf_filter.SetTransform(tf0)
                tf_filter.SetInputData(ugrid)
                map = vtkDataSetMapper()
                map.SetInputConnection(tf_filter.GetOutputPort())
                actor = vtkActor()
                actor.SetMapper(map)
                self.actor = actor
            H_IB = np.block(
                [
                    [A_IB, r_OP],
                    [0, 0, 0, 1],
                ]
            )
            self.vtk_tf.SetMatrix(H_IB.flatten())
            return self.actor

    return _Meshed


def Box(Base):
    MeshedBase = Meshed(Base)

    class _Box(MeshedBase):
        def __init__(
            self,
            dimensions=np.ones(3),
            **kwargs,
        ):
            self.dimensions = dimensions
            trimesh_obj = trimesh.creation.box(extents=dimensions)
            super().__init__(mesh_obj=trimesh_obj, **kwargs)

    return _Box


def Cone(Base):
    MeshedBase = Meshed(Base)

    class _Cone(MeshedBase):
        def __init__(
            self,
            radius=1,
            height=2,
            **kwargs,
        ):
            self.radius = radius
            self.height = height
            trimesh_obj = trimesh.creation.cone(radius, height)
            super().__init__(mesh_obj=trimesh_obj, **kwargs)

    return _Cone


def Cylinder(Base):
    MeshedBase = Meshed(Base)

    class _Cylinder(MeshedBase):
        def __init__(
            self,
            radius=1,
            height=2,
            **kwargs,
        ):
            self.radius = radius
            self.height = height
            trimesh_obj = trimesh.creation.cylinder(radius, height=height)
            super().__init__(mesh_obj=trimesh_obj, **kwargs)

    return _Cylinder


def Sphere(Base):
    MeshedBase = Meshed(Base)

    class _Sphere(MeshedBase):
        def __init__(
            self,
            radius=1,
            subdivisions=2,
            **kwargs,
        ):
            self.radius = radius
            self.subdivisions = subdivisions
            trimesh_obj = trimesh.creation.icosphere(
                radius=radius, subdivisions=subdivisions
            )
            super().__init__(mesh_obj=trimesh_obj, **kwargs)

    return _Sphere


def Capsule(Base):
    MeshedBase = Meshed(Base)

    class _Capsule(MeshedBase):
        def __init__(
            self,
            radius=1,
            height=2,
            **kwargs,
        ):
            self.radius = radius
            self.height = height
            trimesh_obj = trimesh.creation.capsule(radius=radius, height=height)
            super().__init__(mesh_obj=trimesh_obj, **kwargs)

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
    MeshedBase = Meshed(Base)

    class _Axis(MeshedBase):
        def __init__(
            self,
            origin_size=0.04,
            **kwargs,
        ):
            trimesh_obj = trimesh.creation.axis(origin_size=origin_size)
            super().__init__(mesh_obj=trimesh_obj, **kwargs)

    return _Axis
