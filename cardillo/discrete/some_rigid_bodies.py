import numpy as np
from meshzoo import uv_sphere
from cardillo.discrete import new_convex_rigid_body


def new_ball(base_class, m, r, q0, u0=None):
    class Ball(base_class):
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

    return Ball(m, r, q0, u0)


def new_box(base_class, length, width, height, rho=None, mass=None, q0=None, u0=None):
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
    return new_convex_rigid_body(base_class, points, rho, mass, q0, u0)
