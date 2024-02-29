import numpy as np
from cardillo.constraints._base import ProjectedPositionOrientationBase


class Planarizer(ProjectedPositionOrientationBase):
    def __init__(
        self,
        subsystem1,
        subsystem2,
        axis,
        r_OJ0=None,
        A_IJ0=None,
        frame_ID1=None,
        frame_ID2=None,
    ):
        assert axis in (0, 1, 2)

        # plane axes
        plane_axes = np.delete((0, 1, 2), axis)

        # project plane normal on both plane axes
        projection_pairs_rotation = [
            (axis, plane_axes[0]),
            (axis, plane_axes[1]),
        ]

        super().__init__(
            subsystem1,
            subsystem2,
            r_OJ0=r_OJ0,
            A_IJ0=A_IJ0,
            constrained_axes_translation=[axis],
            projection_pairs_rotation=projection_pairs_rotation,
            frame_ID1=frame_ID1,
            frame_ID2=frame_ID2,
        )
