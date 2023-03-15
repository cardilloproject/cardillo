import numpy as np
from cardillo.constraints._base import ProjectedPositionOrientationBase


class Slider1D(ProjectedPositionOrientationBase):
    def __init__(
        self,
        subsystem1,
        subsystem2,
        axis,
        r_OB0=None,
        A_IB0=None,
        frame_ID1=np.zeros(3),
        frame_ID2=np.zeros(3),
    ):
        raise RuntimeError("Slider1D is not tested!")

        assert axis in (0, 1, 2)

        # remove free axis
        constrained_axes_displacement = np.delete((0, 1, 2), axis)

        super().__init__(
            subsystem1,
            subsystem2,
            r_OB0=r_OB0,
            A_IB0=A_IB0,
            constrained_axes_translation=constrained_axes_displacement,
            projection_pairs_rotation=[],
            frame_ID1=frame_ID1,
            frame_ID2=frame_ID2,
        )


class Slider2D(ProjectedPositionOrientationBase):
    def __init__(
        self,
        subsystem1,
        subsystem2,
        axes,
        r_OB0=None,
        A_IB0=None,
        frame_ID1=np.zeros(3),
        frame_ID2=np.zeros(3),
    ):
        raise RuntimeError("Slider2D is not tested!")

        assert len(axes) == 2
        assert axes[0] != axes[1]
        for axis in axes:
            assert axis in (0, 1, 2)

        # remove free axis
        constrained_axes_displacement = np.delete((0, 1, 2), axes)

        super().__init__(
            subsystem1,
            subsystem2,
            r_OB0=r_OB0,
            A_IB0=A_IB0,
            constrained_axes_translation=constrained_axes_displacement,
            projection_pairs_rotation=[],
            frame_ID1=frame_ID1,
            frame_ID2=frame_ID2,
        )
