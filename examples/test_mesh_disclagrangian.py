from cardillo.discretization.mesh1D import Mesh1D

import numpy as np

from cardillo.discretization.lagrange import LagrangeKnotVector

polynomial_degree_n = 2
nelement = 2
knot_vector_n = LagrangeKnotVector(polynomial_degree_n, nelement)

# mesh_n = Mesh1D(knot_vector_n, nquadrature=1, dim_q = 3, derivative_order=0, basis="Lagrange_Disc")
mesh_n = Mesh1D(
    knot_vector_n, nquadrature=1, dim_q=3, derivative_order=0, basis="Lagrange"
)

exit()
