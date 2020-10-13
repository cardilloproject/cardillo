from .mesh1D import Mesh1D
from .mesh2D import Mesh2D
from .mesh3D import Mesh3D
from .mesh1D_lagrange import Mesh1D_lagrange
from .mesh2D_lagrange import Mesh2D_lagrange
from .mesh3D_lagrange import Mesh3D_lagrange

def Mesh(dimension, knot_vector_objs, nqp_per_dim, derivative_order, basis, nq_n):
    if basis == 'B-spline':
        if dimension == 1:
            return Mesh1D(knot_vector_objs, nqp_per_dim, derivative_order=derivative_order, nq_n=nq_n)

        if dimension == 2:
            return Mesh2D(knot_vector_objs, nqp_per_dim, derivative_order=derivative_order, nq_n=nq_n)

        if dimension == 3:
            return Mesh3D(knot_vector_objs, nqp_per_dim, derivative_order=derivative_order, nq_n=nq_n)

    if basis == 'lagrange':
        if dimension == 1:
            return Mesh1D_lagrange(knot_vector_objs, nqp_per_dim, derivative_order=derivative_order, nq_n=nq_n)

        if dimension == 2:
            return Mesh2D_lagrange(knot_vector_objs, nqp_per_dim, derivative_order=derivative_order, nq_n=nq_n)

        if dimension == 3:
            return Mesh3D_lagrange(knot_vector_objs, nqp_per_dim, derivative_order=derivative_order, nq_n=nq_n)