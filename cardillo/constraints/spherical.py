import numpy as np
from cardillo.constraints._base import PositionOrientationBase


class Spherical(PositionOrientationBase):
    def __init__(
        self,
        subsystem1,
        subsystem2,
        r_OJ0,
        xi1=None,
        xi2=None,
        **kwargs,
    ):
        super().__init__(
            subsystem1,
            subsystem2,
            r_OJ0=r_OJ0,
            A_IJ0=None,
            projection_pairs_rotation=[],
            xi1=xi1,
            xi2=xi2,
            **kwargs,
        )
