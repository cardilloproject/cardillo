from abc import ABC, abstractmethod
from collections import namedtuple
from enum import IntEnum
import numpy as np
from vtk import (
    VTK_BEZIER_HEXAHEDRON,
    VTK_BEZIER_WEDGE,
    VTK_LAGRANGE_HEXAHEDRON,
    VTK_LAGRANGE_WEDGE,
)


vtk_types = IntEnum(
    "VTK_TYPES",
    [("Hexahedron", 0), ("Wedge", 1)],
)
vtk_interpolation = namedtuple("vtk_interpolation", ["Hexahedron", "Wedge"])
vtk_bezier = vtk_interpolation(VTK_BEZIER_HEXAHEDRON, VTK_BEZIER_WEDGE)
vtk_lagrange = vtk_interpolation(VTK_LAGRANGE_HEXAHEDRON, VTK_LAGRANGE_WEDGE)


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

    @property
    @abstractmethod
    def vtk_rational_weights(self): ...

    @abstractmethod
    def vtk_compute_points(
        self, r_OP_segments, d2_segments, d3_segments, interpolation=vtk_bezier
    ): ...

    @abstractmethod
    def vtk_connectivity(self, p_zeta): ...


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


class CircularCrossSection(ExportableCrossSection):
    def __init__(self, radius, *, export_as_wedge=True):
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

        self.circle_as_wedge = export_as_wedge
        self.compute_my_alphas()

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

    def compute_my_alphas(self):
        # alpha measures the angle to d2
        if self.circle_as_wedge:
            self.alphas = [
                np.pi * 3 / 2,
                np.pi / 6,
                np.pi * 5 / 6,
                np.pi * 11 / 6,
                np.pi / 2,
                np.pi * 7 / 6,
            ]

        else:
            self.alphas = [
                np.pi * 3 / 2,
                0,
                np.pi / 2,
                np.pi,
                np.pi * 7 / 4,
                np.pi / 4,
                np.pi * 3 / 4,
                np.pi * 5 / 4,
                0,
            ]

    @property
    def point_etas(self):
        return self.alphas

    def B_r_PQ(self, eta):
        return self.radius * np.array([0, np.cos(eta), np.sin(eta)])

    def B_r_PQ_eta(self, eta):
        return self.radius * np.array([0, -np.sin(eta), np.cos(eta)])

    @property
    def vtk_degree(self):
        return 2

    @property
    def vtk_points_per_layer(self):
        if self.circle_as_wedge:
            return 6
        else:
            return 9

    def vtk_connectivity(self, p_zeta):
        nlayers = p_zeta + 1
        ppl = self.vtk_points_per_layer
        npoints = nlayers * ppl

        numbered_layers = np.array(
            [np.arange(ppl * layer, ppl * (layer + 1)) for layer in range(p_zeta + 1)]
        )

        if self.circle_as_wedge:
            vtk_cell_type = vtk_types.Wedge
            # fmt: off
            connectivity_main = np.concatenate(
                [
                    numbered_layers[0, :3],     # vertices bottom
                    numbered_layers[-1, :3],    # vertices top
                    numbered_layers[0, 3:],     # edges bottom
                    numbered_layers[-1, 3:],    # edges top
                    numbered_layers[1:-1, 0],   # edge1 middle
                    numbered_layers[1:-1, 1],   # edge2 middle
                    numbered_layers[1:-1, 2],   # edge3 middle
                    numbered_layers[1:-1, 3],   # faces1 middle
                    numbered_layers[1:-1, 4],   # faces2 middle
                    numbered_layers[1:-1, 5],   # faces3 middle
                ]
            )
            connectivity_flat = np.concatenate(
                [
                    numbered_layers[-1, :3],    # vertices top (last)
                    np.arange(0, 3) + npoints,  # vertices bottom (next)
                    numbered_layers[-1, 3:],    # edges top (last)
                    np.arange(3, 6) + npoints,  # edges bottom (next)  
                ]
            )
            # fmt: on
        else:
            vtk_cell_type = vtk_types.Hexahedron

            # fmt: off
            connectivity_main = np.concatenate(
                [
                    numbered_layers[0, :4],     # vertices bottom
                    numbered_layers[-1, :4],    # vertices top
                    numbered_layers[0, 4:-1],   # edges bottom
                    numbered_layers[-1, 4:-1],  # edges top
                    numbered_layers[1:-1,0],    # edges middle 0
                    numbered_layers[1:-1,1],    # edges middle 1
                    numbered_layers[1:-1,2],    # edges middle 2
                    numbered_layers[1:-1,3],    # edges middle 3
                    numbered_layers[1:-1,7],    # faces middle 7
                    numbered_layers[1:-1,5],    # faces middle 5
                    numbered_layers[1:-1,4],    # faces middle 4
                    numbered_layers[1:-1,6],    # faces middle 6
                    [numbered_layers[0,-1]],    # face bottom
                    [numbered_layers[-1,-1]],   # face top
                    numbered_layers[1:-1,-1],   # volume middle 8
                ]
            )
            connectivity_flat = np.concatenate(
                [
                    numbered_layers[-1, :4],    # vertices top (last)
                    np.arange(0, 4) + npoints,  # vertices bottom (next)
                    numbered_layers[-1, 4:-1],  # edges top (last)
                    np.arange(4, 8) + npoints,  # edges bottom (next)
                    [npoints],                  # face top (last)
                    [npoints + ppl - 1],        # face bottom (next)   
                ]
            )
            # fmt: on

        return vtk_cell_type, connectivity_main, connectivity_flat

    @property
    def vtk_rational_weights(self):
        if self.circle_as_wedge:
            return np.array([1, 1, 1, 1 / 2, 1 / 2, 1 / 2])
        else:
            s22 = np.sqrt(2) / 2
            return np.array([1, 1, 1, 1, s22, s22, s22, s22, 1])

    def vtk_compute_points(
        self, r_OP_segments, d2_segments, d3_segments, interpolation=vtk_bezier
    ):
        if self.circle_as_wedge:

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
            sa = ri / 2
            ca = ri * np.sqrt(3) / 2

            def compute_points(segment, layer):
                r_OP = r_OP_segments[segment, layer]
                d2 = d2_segments[segment, layer]
                d3 = d3_segments[segment, layer]

                P0 = r_OP - ri * d3
                P1 = r_OP + ca * d2 + sa * d3
                P2 = r_OP - ca * d2 + sa * d3

                if interpolation == vtk_bezier:
                    P3 = r_OP + 2 * ca * d2 - ri * d3
                    P4 = r_OP + 2 * ri * d3
                    P5 = r_OP - 2 * ca * d2 - ri * d3

                elif interpolation == vtk_lagrange:
                    # set P3, P4, P5 on the circle
                    P3 = r_OP + ca * d2 - sa * d3
                    P4 = r_OP + ri * d3
                    P5 = r_OP - ca * d2 - sa * d3

                return np.vstack([P0, P1, P2, P3, P4, P5])

            return compute_points

        else:

            sqrt2_inv = 1 / np.sqrt(2)

            def compute_points(segment, layer):
                r_OP = r_OP_segments[segment, layer]
                d2 = d2_segments[segment, layer]
                d3 = d3_segments[segment, layer]

                rd2 = d2 * self.radius
                rd3 = d3 * self.radius

                # points on the circle
                P0 = r_OP - rd3
                P1 = r_OP + rd2
                P2 = r_OP + rd3
                P3 = r_OP - rd2

                if interpolation == vtk_bezier:
                    # points on the outer square
                    P4 = r_OP - rd3 + rd2
                    P5 = r_OP + rd2 + rd3
                    P6 = r_OP + rd3 - rd2
                    P7 = r_OP - rd2 - rd3
                elif interpolation == vtk_lagrange:
                    # points on the circle
                    P4 = r_OP + (rd2 - rd3) * sqrt2_inv
                    P5 = r_OP + (rd2 + rd3) * sqrt2_inv
                    P6 = r_OP + (rd3 - rd2) * sqrt2_inv
                    P7 = r_OP - (rd2 + rd3) * sqrt2_inv

                # center point
                P8 = r_OP

                return np.vstack([P0, P1, P2, P3, P4, P5, P6, P7, P8])

            return compute_points


class RectangularCrossSection(ExportableCrossSection):
    def __init__(self, width, height):
        """Rectangular cross-section.

        Parameters:
        -----
        width : float
            Cross-section dimension in in e_y^B-direction.
        height : float
            Cross-section dimension in in e_z^B-direction.
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

    def vtk_connectivity(self, p_zeta):
        nlayers = p_zeta + 1
        ppl = self.vtk_points_per_layer
        npoints = nlayers * ppl

        numbered_layers = np.array(
            [np.arange(ppl * layer, ppl * (layer + 1)) for layer in range(p_zeta + 1)]
        )

        vtk_cell_type = vtk_types.Hexahedron
        # fmt: off
        connectivity_main = np.concatenate(
            [
                numbered_layers[0, :4],     # vertices bottom
                numbered_layers[-1, :4],    # vertices top
                numbered_layers[1:-1,0],    # edges middle 0
                numbered_layers[1:-1,1],    # edges middle 1
                numbered_layers[1:-1,2],    # edges middle 2
                numbered_layers[1:-1,3],    # edges middle 3
            ]
        )
        connectivity_flat = np.concatenate(
            [
                numbered_layers[-1, :4],    # vertices top (last)
                np.arange(0, 4) + npoints,  # vertices bottom (next)
            ]
        )
        # fmt: on

        return vtk_cell_type, connectivity_main, connectivity_flat

    @property
    def vtk_rational_weights(self):
        return np.repeat(1, 4)

    def vtk_compute_points(
        self, r_OP_segments, d2_segments, d3_segments, interpolation=None
    ):
        def compute_points(segment, layer):
            r_OP = r_OP_segments[segment, layer]
            d2 = d2_segments[segment, layer]
            d3 = d3_segments[segment, layer]

            #       ^ d3
            # P3    |     P2
            # x-----+-----x
            # |     |     |
            # |-----o-----|--> d2
            # |           |
            # x-----------x
            # P0          P1

            r_PP2 = d2 * self.width / 2 + d3 * self.height / 2
            r_PP1 = d2 * self.width / 2 - d3 * self.height / 2

            P0 = r_OP - r_PP2
            P1 = r_OP + r_PP1
            P2 = r_OP + r_PP2
            P3 = r_OP - r_PP1

            return np.vstack([P0, P1, P2, P3])

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
            Cross-section inertia tensor represented in the cross-section-fixed B-Basis.

        """
        if density is None or cross_section is None:
            self.A_rho0 = A_rho0
            self.B_I_rho0 = B_I_rho0
        else:
            self.A_rho0 = density * cross_section.area
            self.B_I_rho0 = density * cross_section.second_moment
