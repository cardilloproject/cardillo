import numpy as np
import trimesh
import warnings


def Meshed(Base):
    class _Meshed(Base):
        def __init__(
            self,
            trimesh_obj,
            density=None,
            K_r_SP=np.zeros(3),
            A_KM=np.eye(3),
            **kwargs
        ):
            """Generate an object (typically with Base `Frame` or `RigidBody`) from a given Trimesh object
            Args:
                trimesh_obj: instance of trimesh defining the mesh
                density: mass density for the computation of the inertia properties of the mesh. If set to None, user specified mass and K_Theta_S are used.
                K_r_SP (np.ndarray): offset center of mass (S) from STL origin (P) in body fixed K-frame
                A_KM (np.ndarray): tansformation from mesh-fixed frame (M) to body-fixed frame (K)
            """
            self.K_r_SP = K_r_SP
            self.A_KM = A_KM

            #############################
            # consistency checks for mesh
            #############################
            assert isinstance(trimesh_obj, trimesh.Trimesh)

            # primitives are converted to mesh
            if hasattr(trimesh_obj, "to_mesh"):
                trimesh_obj = trimesh_obj.to_mesh()

            # check if mesh represents a valid volume
            if not trimesh_obj.is_volume:
                print(
                    "Imported mesh does not represent a volume, i.e. one of the following properties are not fulfilled: watertight, consistent winding, outward facing normals."
                )
                # try to fill the wholes
                trimesh_obj.fill_holes()
                if not trimesh_obj.is_volume:
                    print(
                        "Using mesh that is not a volume. Computed mass and moment of inertia might be unphyical."
                    )
                else:
                    print("Fixed mesh by filling the holes.")

            # store visual mesh in body fixed frame
            H_KM = np.eye(4)
            H_KM[:3, 3] = K_r_SP
            H_KM[:3, :3] = A_KM
            self.K_visual_mesh = trimesh_obj.copy().apply_transform(H_KM)

            # vectors (transposed) from S to vertices represented in body-fixed frame
            self.K_r_SQi_T = self.K_visual_mesh.vertices.view(np.ndarray).T

            # compute inertia quantities of body
            if density is not None:
                # set density and compute properties
                self.K_visual_mesh.density = density
                mass = self.K_visual_mesh.mass
                K_Theta_S = self.K_visual_mesh.moment_inertia

                mass_arg = kwargs.pop("mass", None)
                K_Theta_S_arg = kwargs.pop("K_Theta_S", None)

                if (mass_arg is not None) and (not np.allclose(mass, mass_arg)):
                    warnings.warn("Specified mass does not correspond to mass of mesh.")
                if (K_Theta_S_arg is not None) and (
                    not np.allclose(K_Theta_S, K_Theta_S_arg)
                ):
                    warnings.warn(
                        "Specified moment of inertia does not correspond to moment of inertia of mesh."
                    )

                kwargs.update({"mass": mass, "K_Theta_S": K_Theta_S})

            super().__init__(**kwargs)

        # def get_visual_mesh_wrt_I(self, t, q):
        #     H_IK = np.eye(4)
        #     H_IK[:3, 3] = self.r_OP(t, q)
        #     H_IK[:3, :3] = self.A_IK(t, q)
        #     return self.K_visual_mesh.copy().apply_transform(H_IK)

        def export(self, sol_i, base_export=False, **kwargs):
            if base_export:
                return super().export(sol_i, **kwargs)
            else:
                r_OS = self.r_OP(
                    sol_i.t, sol_i.q[self.qDOF]
                )  # TODO: slicing could be done on global level in Export class. Moreover, solution class should be able to return the slice, e.g., sol_i.get_q_of_body(name).
                A_IK = self.A_IK(sol_i.t, sol_i.q[self.qDOF])
                points = (r_OS[:, None] + A_IK @ self.K_r_SQi_T).T

                cells = [
                    ("triangle", self.K_visual_mesh.faces),
                ]

            return points, cells, None, None

    return _Meshed

