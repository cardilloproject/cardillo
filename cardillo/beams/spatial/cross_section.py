import numpy as np
from cardillo.math import pi


class CrossSection:
    # TODO: Is this a property of the cross section? Or should we move it to
    # the beam class?
    @property
    def line_density(self):
        """Line density of the cross section."""
        return self._line_density

    @property
    def area(self):
        """Area of the cross section."""
        return self._area

    @property
    def first_moment(self):
        """Vector containing the first moment of areas."""
        return self._first_moment

    @property
    def second_moment(self):
        """Matrix containing the second moment of areas."""
        return self._second_moment


class UserDefinedCrossSection(CrossSection):
    def __init__(self, line_density, area, first_moment, second_moment):
        self._line_density = line_density
        self._area = area
        self._first_moment = first_moment
        self._second_moment = second_moment


class CircularCrossSection(CrossSection):
    def __init__(self, line_density, radius):
        self._line_density = line_density
        self._radius = radius
        self._area = pi * radius**2
        # see https://de.wikipedia.org/wiki/Fl%C3%A4chenmoment
        self._first_moment = np.zeros(3)
        # https://en.wikipedia.org/wiki/List_of_second_moments_of_area
        self._second_moment = np.diag([2, 1, 1]) / 4 * pi * radius**4


class RectangularCrossSection(CrossSection):
    """Rectangular cross section.

    Note:
    -----
    * width denotes the dimension in d2-direction
    * height defindenoteses the dimension in d3-direction
    """

    def __init__(self, line_density, width, height):
        self._line_density = line_density
        self._width = width
        self._height = height
        self._area = width * height
        # see https://de.wikipedia.org/wiki/Fl%C3%A4chenmoment
        self._first_moment = np.zeros(3)
        # https://en.wikipedia.org/wiki/List_of_second_moments_of_area
        self._second_moment = (
            np.diag(
                [
                    width * height**3 + width**3 * height,
                    width * height**3,
                    width**3 * height,
                ]
            )
            / 12
        )


class QuadraticCrossSection(RectangularCrossSection):
    def __init__(self, line_density, width):
        super().__init__(line_density, width, width)
