import numpy as np


class CrossSection:
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
    def __init__(self, area, first_moment, second_moment):
        self._area = area
        self._first_moment = first_moment
        self._second_moment = second_moment


class CircularCrossSection(CrossSection):
    def __init__(self, radius):
        self._radius = radius
        self._area = np.pi * radius**2
        # see https://de.wikipedia.org/wiki/Fl%C3%A4chenmoment
        self._first_moment = np.zeros(3)
        # https://en.wikipedia.org/wiki/List_of_second_moments_of_area
        self._second_moment = np.diag([2, 1, 1]) / 4 * np.pi * radius**4

    @property
    def radius(self):
        return self._radius


class RectangularCrossSection(CrossSection):
    """Rectangular cross section.

    Note:
    -----
    * width denotes the dimension in d2-direction
    * height denotes the dimension in d3-direction
    """

    def __init__(self, width, height):
        self._width = width
        self._height = height
        self._area = width * height
        # see https://de.wikipedia.org/wiki/Fl%C3%A4chenmoment
        self._first_moment = np.zeros(3, dtype=float)
        # https://en.wikipedia.org/wiki/List_of_second_moments_of_area
        self._second_moment = (
            np.diag(
                [
                    width * height**3 + width**3 * height,
                    width * height**3,
                    width**3 * height,
                ]
            )
            / 12.0
        )

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height


class CrossSectionInertias:
    def __init__(
        self, line_density=None, cross_section=None, A_rho0=1.0, K_I_rho0=np.eye(3)
    ):
        if line_density is None or cross_section is None:
            self.A_rho0 = A_rho0
            self.K_I_rho0 = K_I_rho0
        else:
            self.A_rho0 = line_density * cross_section.area
            self.K_I_rho0 = line_density * cross_section.second_moment
