from abc import ABC, abstractmethod
import numpy as np


class CrossSection(ABC):
    """Abstract class definition for rod cross-sections."""

    @property
    @abstractmethod
    def area(self):
        """Area of the cross-section."""
        ...

    @property
    @abstractmethod
    def first_moment(self):
        """Vector containing the first moments of area."""
        ...

    @property
    @abstractmethod
    def second_moment(self):
        """Matrix containing the second moments of area."""
        ...


class UserDefinedCrossSection(CrossSection):
    def __init__(self, area, first_moment, second_moment):
        """User defined cross-section.

        Parameters
        ----------
        area: float
            Area of the cross-section.
        first_moment: np.ndarray (3,)
            Vector containing the first moments of area.
        second_moment: np.ndarray (3, 3)
            Matrix containing the second moments of area.
        """
        self._area = area
        self._first_moment = first_moment
        self._second_moment = second_moment

    @property
    def area(self):
        return self._area

    @property
    def first_moment(self):
        return self._first_moment

    @property
    def second_moment(self):
        return self._second_moment


class CircularCrossSection(CrossSection):
    def __init__(self, radius):
        """Circular cross-section.

        Parameters
        ----------
        radius: float
            Radius of the cross-section
        """
        self._radius = radius
        self._area = np.pi * radius**2
        # see https://en.wikipedia.org/wiki/First_moment_of_area
        self._first_moment = np.zeros(3)
        # https://en.wikipedia.org/wiki/List_of_second_moments_of_area
        self._second_moment = np.diag([2, 1, 1]) / 4 * np.pi * radius**4

    @property
    def area(self):
        return self._area

    @property
    def first_moment(self):
        return self._first_moment

    @property
    def second_moment(self):
        return self._second_moment

    @property
    def radius(self):
        return self._radius


class RectangularCrossSection(CrossSection):
    def __init__(self, width, height):
        """Rectangular cross-section.

        Parameters:
        -----
        width: float
            Cross-section dimension in in e_y^K-direction.
        height: float
            Cross-section dimension in in e_z^K-direction.
        """
        self._width = width
        self._height = height
        self._area = width * height
        # see https://en.wikipedia.org/wiki/First_moment_of_area
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
    def area(self):
        return self._area

    @property
    def first_moment(self):
        return self._first_moment

    @property
    def second_moment(self):
        return self._second_moment

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height


class CrossSectionInertias:
    def __init__(
        self, density=None, cross_section=None, A_rho0=1.0, K_I_rho0=np.eye(3)
    ):
        """Inertial properties of cross-sections. Centerline must coincide with line of centroids.

        Parameters:
        -----
        density: float
            Mass per unit reference volume of the rod.
        cross_section: CrossSection
            Cross-section object, which provides cross-section area and second moment of area.
        A_rho0: float
            Cross-section mass density, i.e., mass per unit reference length of rod.
        K_I_rho0: np.array(3, 3)
            Cross-section inertia tensor represented in the cross-section-fixed K-Basis.

        """
        if density is None or cross_section is None:
            self.A_rho0 = A_rho0
            self.K_I_rho0 = K_I_rho0
        else:
            self.A_rho0 = density * cross_section.area
            self.K_I_rho0 = density * cross_section.second_moment
