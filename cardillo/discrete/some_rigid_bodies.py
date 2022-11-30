import numpy as np
from meshzoo import uv_sphere
from cardillo.discrete import ConvexRigidBody


def Ball(RigidBodyParametrization):
    class _Ball(RigidBodyParametrization):
        def __init__(self, m, r, q0, u0=None):
            K_theta_S = 2 / 5 * m * r**2 * np.eye(3)
            self.r = r
            super().__init__(m, K_theta_S, q0, u0)

        def export(self, sol_i, resolution=20, base_export=False, **kwargs):
            if base_export:
                points, cells, point_data, cell_data = super().export(sol_i)
            else:
                points_sphere, cells_sphere = uv_sphere(
                    num_points_per_circle=resolution,
                    num_circles=resolution,
                    radius=self.r,
                )
                points, vel, acc = [], [], []
                for point in points_sphere:
                    points.append(self.r_OP(sol_i.t, sol_i.q[self.qDOF], K_r_SP=point))
                    vel.append(
                        self.v_P(
                            sol_i.t,
                            sol_i.q[self.qDOF],
                            sol_i.u[self.uDOF],
                            K_r_SP=point,
                        )
                    )
                    if sol_i.u_dot is not None:
                        acc.append(
                            self.a_P(
                                sol_i.t,
                                sol_i.q[self.qDOF],
                                sol_i.u[self.uDOF],
                                sol_i.u_dot[self.uDOF],
                                K_r_SP=point,
                            )
                        )
                    cells = [("polyhedron", np.array([cells_sphere]))]
                    if sol_i.u_dot is not None:
                        point_data = dict(v=vel, a=acc)
                    else:
                        point_data = dict(v=vel)
            return points, cells, point_data, None

    return _Ball


def Box(RigidBodyParametrization):
    class _Box(ConvexRigidBody(RigidBodyParametrization)):
        def __init__(self, length, width, height, q0, u0, rho=None, mass=None):
            points = np.array(
                [
                    [0, 0, 0],
                    [length, 0, 0],
                    [length, width, 0],
                    [0, width, 0],
                    [0, 0, height],
                    [length, 0, height],
                    [length, width, height],
                    [0, width, height],
                ]
            )
            super().__init__(points, rho, mass, q0, u0)

    return _Box
