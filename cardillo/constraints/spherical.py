import numpy as np
from cardillo.constraints._base import PositionOrientationBase


class Spherical(PositionOrientationBase):
    def __init__(
        self,
        subsystem1,
        subsystem2,
        r_OJ0,
        frame_ID1=None,
        frame_ID2=None,
    ):
        super().__init__(
            subsystem1,
            subsystem2,
            r_OJ0=r_OJ0,
            A_IJ0=None,
            projection_pairs_rotation=[],
            frame_ID1=frame_ID1,
            frame_ID2=frame_ID2,
        )
