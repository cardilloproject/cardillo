import numpy as np
from cardillo.math import pi
from abc import ABCMeta, abstractmethod


class AbstractCrossSection(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    # TODO: What are the abstract quantities that are required?
    @property
    @abstractmethod
    def line_density(self):
        """Line density of the cross section."""
        pass

    @property
    @abstractmethod
    def area(self):
        """Area of the cross section."""
        pass

    @property
    @abstractmethod
    def first_moment(self):
        """Vector containing the first moment of areas."""
        pass

    @property
    @abstractmethod
    def second_moment(self):
        """Matrix containing the second moment of areas."""
        pass


class CircularCrossSection(AbstractCrossSection):
    def __init__(self, line_density, radius):
        self.line_density = line_density
        self.radius = radius
        self.area = pi * radius**2
        # TODO:
        self.first_moment = np.array([])
        # https://en.wikipedia.org/wiki/List_of_second_moments_of_area
        self.first_moment = np.diag([2, 1, 1]) / 4 * pi * radius**4


# class CircularBeamCrossSection(BeamCrossSection):
#     def __init__(self, radius, line_density):
#         self.radius = radius

#         # density per line element
#         self.line_density = line_density  # TODO: Move to base class

#         # area of the cross section
#         self.area = pi * radius**2

#         # second moments of area, see
#         # https://en.wikipedia.org/wiki/List_of_second_moments_of_area
#         self.I2 = self.I3 = 0.25 * pi * radius**4
#         self.I1 = self.I2 + self.I3

#         # inertia properties used in director beam formulations
#         self.A_rho0 = self.line_density * self.area
#         self.B_rho0 = np.zeros(3)  # symmetric cross section
#         self.C_rho0 = self.line_density * np.diag([0, self.I3, self.I2])

#         # Inertia properties for "angular velocity + virtual rotation" beam 
#         # formulations.
#         # Note: These should coincide with the definition used for rigid bodies
#         self.I_rho0 = self.line_density * np.diag([self.I1, self.I2, self.I3])

#         # TODO:
#         super().__init__()


class RectangularCrossSection(AbstractCrossSection):
    """Rectangular cross section.
    
    Note:
    -----
    * width denotes the dimension in d2-direction
    * height defindenoteses the dimension in d3-direction
    """
    def __init__(self, line_density, width, height):
        self.line_density = line_density
        self.width = width
        self.height = height
        self.area = width * height
        # TODO:
        self.first_moment = np.array([])
        # https://en.wikipedia.org/wiki/List_of_second_moments_of_area
        self.first_moment = np.diag([
            width * height**3 + width**3 * height,
            width * height**3,
            width**3 * height
        ]) / 12

class QuadraticCrossSection(RectangularCrossSection):
    def __init__(self, line_density, width):
        super().__init__(line_density, width, width)
