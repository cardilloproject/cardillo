import numpy as np
from cardillo.constraints._base import ProjectedPositionOrientationBase


class Prismatic(ProjectedPositionOrientationBase):
    def __init__(
        self,
        subsystem1,
        subsystem2,
        axis,
        r_OJ0=None,
        A_IJ0=None,
        xi1=None,
        xi2=None,
    ):
        assert axis in (0, 1, 2)

        # remove free axis
        constrained_axes_displacement = np.delete((0, 1, 2), axis)

        # all orientations are constrained
        projection_pairs_rotation = [(0, 1), (1, 2), (2, 0)]

        super().__init__(
            subsystem1,
            subsystem2,
            r_OJ0=r_OJ0,
            A_IJ0=A_IJ0,
            constrained_axes_translation=constrained_axes_displacement,
            projection_pairs_rotation=projection_pairs_rotation,
            xi1=xi1,
            xi2=xi2,
        )
