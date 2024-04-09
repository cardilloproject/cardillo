import numpy as np
from cardillo.rods.discretization import gauss, lobatto
import pytest


def test_gauss():
    points, weigths = gauss(1)
    assert np.allclose(points, np.array([0], dtype=float))
    assert np.allclose(weigths, np.array([2], dtype=float))

    points, weigths = gauss(2)
    assert np.allclose(points, np.array([-1, 1], dtype=float) / np.sqrt(3))
    assert np.allclose(weigths, np.array([1, 1], dtype=float))

    points, weigths = gauss(3)
    assert np.allclose(points, np.array([-1, 0, 1], dtype=float) * np.sqrt(3 / 5))
    assert np.allclose(weigths, np.array([5, 8, 5], dtype=float) / 9)


def test_lobatto1():
    with pytest.raises(AssertionError):
        lobatto(-1)
    with pytest.raises(AssertionError):
        lobatto(0)
    with pytest.raises(AssertionError):
        lobatto(1)


def test_lobatto2():
    points, weigths = lobatto(2)
    assert np.allclose(points, np.array([-1, 1], dtype=float))
    assert np.allclose(weigths, np.array([1, 1], dtype=float))

    points, weigths = lobatto(3)
    assert np.allclose(points, np.array([-1, 0, 1], dtype=float))
    assert np.allclose(weigths, np.array([1, 4, 1], dtype=float) / 3)

    points, weigths = lobatto(4)
    assert np.allclose(
        points, np.array([-1, -1 / np.sqrt(5), 1 / np.sqrt(5), 1], dtype=float)
    )
    assert np.allclose(weigths, np.array([1, 5, 5, 1], dtype=float) / 6)

    points, weigths = lobatto(5)
    assert np.allclose(
        points, np.array([-1, -np.sqrt(3 / 7), 0, np.sqrt(3 / 7), 1], dtype=float)
    )
    assert np.allclose(
        weigths, np.array([1 / 10, 49 / 90, 32 / 45, 49 / 90, 1 / 10], dtype=float)
    )


if __name__ == "__main__":
    test_gauss()
    test_lobatto1()
    test_lobatto2()
