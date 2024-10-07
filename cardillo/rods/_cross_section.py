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


class ExportableCrossSection(CrossSection):
    @property
    @abstractmethod
    def vtk_degree(self): ...

    @property
    @abstractmethod
    def vtk_points_per_layer(self): ...

    @abstractmethod
    def vtk_compute_points(self, r_OP_segments, d2_segments, d3_segments): ...


class UserDefinedCrossSection(CrossSection):
    def __init__(self, area, first_moment, second_moment):
        """User defined cross-section.

        Parameters
        ----------
        area : float
            Area of the cross-section.
        first_moment : np.ndarray (3,)
            Vector containing the first moments of area.
        second_moment : np.ndarray (3, 3)
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


circle_as_wedge = True
# circle_as_wedge = False


class CircularCrossSection(ExportableCrossSection):
    def __init__(self, radius):
        """Circular cross-section.

        Parameters
        ----------
        radius : float
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

    @property
    def vtk_degree(self):
        return 2

    @property
    def vtk_points_per_layer(self):
        if circle_as_wedge:
            return 6
        else:
            return 9

    def vtk_compute_points(self, r_OP_segments, d2_segments, d3_segments):
        if circle_as_wedge:

            # points:
            #         ^ d3
            #         |
            #         x  P4
            #       / | \
            #      /-----\
            # P2  x   |   x  P1
            #    /|---o---|\--> d2
            #   / \   |   / \
            #  x---\--x--/---x
            #  P5     P0     P3
            #
            # P0-P1-P2 form the inner equilateral triangle. Later refered to as points for layer 0 and 3 or edges for layer 1 and 2. They are weighted with 1
            # P3-P4-P5 from the outer equilateral traingle. Later refered to as edges for layer 0 and 3 or faces for layer 1 and 2. They are weighted with 1/2

            # radii
            ri = self.radius
            ru = 2 * ri
            a = 2 * np.sqrt(3) * ri

            def compute_points(segment, layer):
                r_OP = r_OP_segments[segment, layer]
                d2 = d2_segments[segment, layer]
                d3 = d3_segments[segment, layer]

                P0 = r_OP - ri * d3
                P3 = r_OP + a / 2 * d2 - ri * d3
                P4 = r_OP + d3 * ru

                P5 = 2 * P0 - P3
                P1 = 0.5 * (P3 + P4)
                P0 = 0.5 * (P5 + P3)
                P2 = 0.5 * (P4 + P5)

                dim = len(P0)
                points_weights = np.zeros((6, dim + 1))
                points_weights[0] = np.array([*P0, 1])
                points_weights[1] = np.array([*P1, 1])
                points_weights[2] = np.array([*P2, 1])
                points_weights[3] = np.array([*P3, 1 / 2])
                points_weights[4] = np.array([*P4, 1 / 2])
                points_weights[5] = np.array([*P5, 1 / 2])

                return points_weights

            return compute_points

        else:

            def compute_points(segment, layer):
                r_OP = r_OP_segments[segment, layer]
                d2 = d2_segments[segment, layer]
                d3 = d3_segments[segment, layer]

                P2 = r_OP + d3 * self.radius
                P1 = r_OP + d2 * self.radius
                P8 = r_OP

                P0 = 2 * P8 - P2
                P3 = 2 * P8 - P1
                P4 = (P0 + P1) - P8
                P5 = (P1 + P2) - P8
                P6 = (P2 + P3) - P8
                P7 = (P3 + P0) - P8

                dim = len(P0)
                s22 = np.sqrt(2) / 2
                points_weights = np.zeros((9, dim + 1))
                points_weights[0] = np.array([*P0, 1])
                points_weights[1] = np.array([*P1, 1])
                points_weights[2] = np.array([*P2, 1])
                points_weights[3] = np.array([*P3, 1])
                points_weights[4] = np.array([*P4, s22])
                points_weights[5] = np.array([*P5, s22])
                points_weights[6] = np.array([*P6, s22])
                points_weights[7] = np.array([*P7, s22])
                points_weights[8] = np.array([*P8, 1])

                return points_weights

            return compute_points

    def vtk_compute_surface_normals(self, r_OP_segments, d2_segments, d3_segments):
        if circle_as_wedge:
            # TODO: think about how to proceed with the normals

            # TODO: preprocess these

            # alpha measures the angle to d2
            alpha0 = np.pi * 3 / 2
            alpha1 = np.pi / 6
            alpha2 = np.pi * 5 / 6
            alpha3 = np.pi * 11 / 6
            alpha4 = np.pi / 2
            alpha5 = np.pi * 7 / 6

            alphas = [alpha0, alpha1, alpha2, alpha3, alpha4, alpha5]
            trig_alphas = [(np.sin(alpha), np.cos(alpha)) for alpha in alphas]

            def compute_normals(segment, layer):
                # TODO: these are not the normals if there is shear!
                d2ii = d2_segments[segment, layer]
                d3ii = d3_segments[segment, layer]

                normals = np.zeros([6, 3])
                for i in range(6):
                    salpha, calpha = trig_alphas[i]
                    normals[i, :] = calpha * d2ii + salpha * d3ii

                return normals

            return compute_normals


class RectangularCrossSection(ExportableCrossSection):
    def __init__(self, width, height):
        """Rectangular cross-section.

        Parameters:
        -----
        width : float
            Cross-section dimension in in e_y^K-direction.
        height : float
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

    @property
    def vtk_degree(self):
        return 1

    @property
    def vtk_points_per_layer(self):
        return 4

    def vtk_compute_points(self, r_OP_segments, d2_segments, d3_segments):
        def compute_points(segment, layer):
            r_OP = r_OP_segments[segment, layer]
            d2 = d2_segments[segment, layer]
            d3 = d3_segments[segment, layer]

            #       ^ d3
            # P3    |     P2
            # x-----+-----x
            # |     |     |
            # |-----o-----Q1--> d2
            # |           |
            # x-----------x
            # P0          P1

            r_PP2 = d2 * self.width / 2 + d3 * self.height / 2
            r_PP1 = d2 * self.width / 2 - d3 * self.height / 2

            P0 = r_OP - r_PP2
            P1 = r_OP + r_PP1
            P2 = r_OP + r_PP2
            P3 = r_OP - r_PP1

            dim = len(P0)
            points_weights = np.zeros((4, dim + 1))
            points_weights[0] = np.array([*P0, 1])
            points_weights[1] = np.array([*P1, 1])
            points_weights[2] = np.array([*P2, 1])
            points_weights[3] = np.array([*P3, 1])

            return points_weights

        return compute_points


class CrossSectionInertias:
    def __init__(
        self, density=None, cross_section=None, A_rho0=1.0, B_I_rho0=np.eye(3)
    ):
        """Inertial properties of cross-sections. Centerline must coincide with line of centroids.

        Parameters:
        -----
        density : float
            Mass per unit reference volume of the rod.
        cross_section : CrossSection
            Cross-section object, which provides cross-section area and second moment of area.
        A_rho0 : float
            Cross-section mass density, i.e., mass per unit reference length of rod.
        B_I_rho0 : np.array(3, 3)
            Cross-section inertia tensor represented in the cross-section-fixed K-Basis.

        """
        if density is None or cross_section is None:
            self.A_rho0 = A_rho0
            self.B_I_rho0 = B_I_rho0
        else:
            self.A_rho0 = density * cross_section.area
            self.B_I_rho0 = density * cross_section.second_moment
