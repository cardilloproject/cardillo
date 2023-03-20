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
        frame_ID1=np.zeros(3),
        frame_ID2=np.zeros(3),
    ):
        super().__init__(
            subsystem1,
            subsystem2,
            r_OB0=None,
            A_IB0=None,
            projection_pairs_rotation=[(0, 1), (1, 2), (2, 0)],
            frame_ID1=frame_ID1,
            frame_ID2=frame_ID2,
        )


# TODO: Measure performance of this implementation!
# class RigidConnection(ProjectedPositionOrientationBase):
#     def __init__(
#         self,
#         subsystem1,
#         subsystem2,
#         frame_ID1=np.zeros(3),
#         frame_ID2=np.zeros(3),
#     ):
#         super().__init__(
#             subsystem1,
#             subsystem2,
#             constrained_axes_displacement=(0, 1, 2),
#             projection_pairs_rotation=[(0, 1), (1, 2), (2, 0)],
#             frame_ID1=frame_ID1,
#             frame_ID2=frame_ID2,
#         )
