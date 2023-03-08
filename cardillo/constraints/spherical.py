import numpy as np
from cardillo.constraints._base import PositionOrientationBase


class Spherical(PositionOrientationBase):
    def __init__(
        self,
        subsystem1,
        subsystem2,
        r_OB0,
        frame_ID1=np.zeros(3),
        frame_ID2=np.zeros(3),
    ):
        super().__init__(
            subsystem1,
            subsystem2,
            r_OB0=r_OB0,
            A_IB0=None,
            projection_pairs=[],
            frame_ID1=frame_ID1,
            frame_ID2=frame_ID2,
        )
