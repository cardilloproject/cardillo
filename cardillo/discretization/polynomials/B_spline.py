from geomdl.knotvector import generate
from geomdl.helpers import find_span_linear, find_span_binsearch, find_spans
from geomdl.helpers import basis_function_ders, basis_functions_ders
from geomdl import BSpline
from geomdl import fitting

from geomdl.visualization import VisMPL

import numpy as np
import matplotlib.pyplot as plt

def uniform_knot_vector(degree, nEl):
    return generate(degree, nEl + degree)

def find_knotspan(degree, knot_vector, knots):
    m = len(knot_vector)

    if not hasattr(knots, '__len__'):
        return find_span_linear(degree, knot_vector, m, knots)
    else:
        return find_spans(degree, knot_vector, m, knots)

def B_spline_basis(degree, order, knot_vector, knots):
    spans = find_knotspan(degree, knot_vector, knots)
    if not hasattr(spans, '__len__'):
        ders = basis_function_ders(degree, knot_vector, spans, knots, order)
    else:
        ders = basis_functions_ders(degree, knot_vector, spans, knots, order)
    # TODO: do we want another ordering here?
    return np.array(ders)

# TODO: wrap fit_B_spline for 3D vectors

def test_B_splines():
    ###########################################################################
    # generate uniform open knot vector
    ###########################################################################
    # degree_max = 4
    # degree = int(np.ceil(np.random.rand(1) * degree_max))
    # nEl_max = 5
    # nEl = int(np.ceil(np.random.rand(1) * nEl_max))
    degree = 2
    nEl = 10
    m = nEl + degree
    knot_vector_geomdl = generate(degree, m)
    knot_vector = uniform_knot_vector(degree, nEl)
    assert knot_vector_geomdl == knot_vector
    print(f'{"-" * 80}')
    print(f'degree: {degree}; nEl: {nEl};\nknot_vector = {knot_vector}')
    print(f'{"-" * 80}\n')

    ###########################################################################
    # find knot span
    ###########################################################################
    xi = np.random.rand(1)[0]
    span_linear = find_span_linear(degree, knot_vector, m, xi)
    span_binsearch = find_span_binsearch(degree, knot_vector, m, xi)
    span = find_knotspan(degree, knot_vector, xi)
    assert span_linear == span_binsearch and span == span_linear

    print(f'{"-" * 80}')
    print(f'xi: {xi}; span: {span_linear}')

    knots = np.random.rand(3)
    spans_geomdl = find_spans(degree, knot_vector, m, knots)
    spans = find_spans(degree, knot_vector, m, knots)
    assert spans_geomdl == spans
    print(f'knots: {knots}; spans: {spans}')
    print(f'{"-" * 80}\n')

    ###########################################################################
    # basis functions and derivatives
    ###########################################################################
    # # xi = 0
    # # xi = 0.5
    # xi = 1
    # span = find_span_linear(degree, knot_vector, m, xi)
    # order = 2
    # ders = basis_function_ders(degree, knot_vector, span, xi, order)
    # print(f'{"-" * 80}')
    # print(f'order: {order}; xi: {xi}; span: {span}; \nbasis_function_ders:\n{np.array(ders)}')

    order = 1
    # knots = [0.01, 0.5, 0.99]
    knots = np.random.rand(4)
    spans = find_spans(degree, knot_vector, m, knots)
    ders_geomdl = basis_functions_ders(degree, knot_vector, spans, knots, order)
    ders = B_spline_basis(degree, order, knot_vector, knots)
    assert np.allclose(np.array(ders_geomdl), ders)
    print(f'ders.shape: {ders.shape}')
    print(f'{"-" * 80}')
    print(f'order: {order}; knots: {knots}; spans: {spans}; \nbasis_functions_ders:\n{ders}')
    print(f'{"-" * 80}\n')

def visualize_B_splines():
    # ###########################################################################
    # # B-spline curves
    # ###########################################################################

    # Create the curve instance
    crv = BSpline.Curve()

    # Set degree
    crv.degree = 2

    # Set control points
    crv.ctrlpts = [[1, 0, 0.1], [1.2, 1, -0.75], [1.5, 1, 0.9], [2, 1, 0]]

    # Set knot vector
    crv.knotvector = [0, 0, 0, 0.5, 1, 1, 1]

    # Import Matplotlib visualization module
    from geomdl.visualization import VisMPL

    # Set the visualization component of the curve
    crv.vis = VisMPL.VisCurve3D()

    # Plot the curve
    crv.render()

    ###########################################################################
    # curve interpolation
    ###########################################################################

    # The NURBS Book Ex9.1
    points = ((0, 0), (3, 4), (-1, 4), (-4, 0), (-4, -3))
    degree = 3  # cubic curve

    # Do global curve interpolation
    curve = fitting.interpolate_curve(points, degree)

    # Plot the interpolated curve
    curve.delta = 0.01
    curve.vis = VisMPL.VisCurve2D()
    curve.render()

    # Visualize data and evaluated points together
    import numpy as np
    import matplotlib.pyplot as plt
    evalpts = np.array(curve.evalpts)
    datapts = np.array(points)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot(evalpts[:, 0], evalpts[:, 1])
    ax.scatter(datapts[:, 0], datapts[:, 1], color="red")
    plt.show()

    ###########################################################################
    # curve fitting
    ###########################################################################
    from geomdl.visualization import VisMPL as vis

    # The NURBS Book Ex9.1
    points = ((0, 0), (3, 4), (-1, 4), (-4, 0), (-4, -3))
    degree = 3  # cubic curve

    # Do global curve approximation
    curve = fitting.approximate_curve(points, degree)

    # Plot the interpolated curve
    curve.delta = 0.01
    curve.vis = VisMPL.VisCurve3D()
    curve.render()

    # Visualize data and evaluated points together
    evalpts = np.array(curve.evalpts)
    datapts = np.array(points)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot(evalpts[:, 0], evalpts[:, 1])
    ax.scatter(datapts[:, 0], datapts[:, 1], color="red")
    plt.show()

if __name__ == "__main__":
    test_B_splines()
    # visualize_B_splines()