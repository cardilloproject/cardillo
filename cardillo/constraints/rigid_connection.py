import numpy as np
from cardillo.constraints._base import (
    PositionOrientationBase,
    ProjectedPositionOrientationBase,
)


class RigidConnection(PositionOrientationBase):
    def __init__(
        self,
        subsystem1,
        subsystem2,
        r_OJ0=None,
        A_IJ0=None,
        xi1=None,
        xi2=None,
        name="rigid_connection",
    ):
        self.name = name
        super().__init__(
            subsystem1,
            subsystem2,
            r_OJ0=r_OJ0,
            A_IJ0=A_IJ0,
            projection_pairs_rotation=[(1, 2), (2, 0), (0, 1)],
            xi1=xi1,
            xi2=xi2,
        )


# TODO: Measure performance of this implementation!
# class RigidConnection(ProjectedPositionOrientationBase):
#     def __init__(
#         self,
#         subsystem1,
#         subsystem2,
#         xi1=None,
#         xi2=None,
#     ):
#         super().__init__(
#             subsystem1,
#             subsystem2,
#             constrained_axes_displacement=(0, 1, 2),
#             projection_pairs_rotation=[(0, 1), (1, 2), (2, 0)],
#             xi1=xi1,
#             xi2=xi2,
#         )
