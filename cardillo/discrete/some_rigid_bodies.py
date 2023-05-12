import numpy as np
from cardillo.visualization import vtk_sphere


def Ball(RigidBodyParametrization):
    class _Ball(RigidBodyParametrization):
        def __init__(self, mass, radius, q0=None, u0=None):
            K_theta_S = 2 / 5 * mass * radius**2 * np.eye(3)
            self.radius = radius
            super().__init__(mass, K_theta_S, q0, u0)

        def export(self, sol_i, resolution=20, base_export=False, **kwargs):
            if base_export:
                points, cells, point_data, cell_data = super().export(sol_i)
            else:
                points_sphere, cell, point_data = vtk_sphere(self.radius)
                cells = [cell]
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
                point_data.update({"v": vel})
                if sol_i.u_dot is not None:
                    point_data.update({"a": acc})

            return points, cells, point_data, None

    return _Ball


def Box(RigidBodyParametrization):
    class _Box(RigidBodyParametrization):
        def __init__(
            self, dimensions, mass=None, density=None, K_theta_S=None, q0=None, u0=None
        ):
            assert len(dimensions) == 3, "3 dimensions are needed to generate a box"
            self.a, self.b, self.c = dimensions

            p1 = np.array([0.5 * self.a, -0.5 * self.b, 0.5 * self.c])
            p2 = np.array([0.5 * self.a, 0.5 * self.b, 0.5 * self.c])
            p3 = np.array([0.5 * self.a, 0.5 * self.b, -0.5 * self.c])
            p4 = np.array([0.5 * self.a, -0.5 * self.b, -0.5 * self.c])
            p5 = np.array([-0.5 * self.a, -0.5 * self.b, 0.5 * self.c])
            p6 = np.array([-0.5 * self.a, 0.5 * self.b, 0.5 * self.c])
            p7 = np.array([-0.5 * self.a, 0.5 * self.b, -0.5 * self.c])
            p8 = np.array([-0.5 * self.a, -0.5 * self.b, -0.5 * self.c])
            self.points = np.vstack((p1, p2, p3, p4, p5, p6, p7, p8))

            if density is not None:
                mass = density * self.a * self.b * self.c
            elif mass is not None:
                mass = mass
            else:
                raise TypeError("mass and density cannot both have type None")

            K_theta_S = (
                (mass / 12)
                * np.array(
                    [
                        [self.b**2 + self.c**2, 0, 0],
                        [0, self.a**2 + self.c**2, 0],
                        [0, 0, self.a**2 + self.b**2],
                    ]
                )
                if K_theta_S is None
                else K_theta_S
            )

            super().__init__(mass, K_theta_S, q0, u0)

        def export(self, sol_i, base_export=False, **kwargs):
            if base_export:
                points, cells, point_data, cell_data = super().export(sol_i)
                return points, cells, point_data, cell_data
            else:
                points, vel, acc = [], [], []
                for point in self.points:
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
                cells = [("hexahedron", [[0, 1, 2, 3, 4, 5, 6, 7]])]

                if sol_i.u_dot is not None:
                    point_data = dict(v=vel, a=acc)
                else:
                    point_data = dict(v=vel)
            return points, cells, point_data, None

    return _Box


if __name__ == "__main__":
    from cardillo.discrete import RigidBodyQuaternion

    dimensions = np.array([3, 1, 2])
    box = Box(RigidBodyQuaternion)(dimensions=dimensions, density=1)
    q0 = np.array([*(0.5 * dimensions), 1, 0, 0, 0], dtype=float)
    ball = Ball(RigidBodyQuaternion)(mass=1, radius=0.2, q0=q0)

    from cardillo import System

    system = System()
    system.add(ball, box)
    system.assemble()

    from cardillo.solver import Solution

    t = [0]
    q = [system.q0]
    u = [system.u0]
    sol = Solution(t=t, q=q, u=u)

    from pathlib import Path
    from cardillo.visualization import Export

    path = Path(__file__)
    e = Export(path.parent, path.stem, True, 30, sol)
    e.export_contr(ball)
    e.export_contr(box)
