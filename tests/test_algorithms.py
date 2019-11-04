import numpy as np
import unittest
import itertools
from math import sin, cos
from fpmlib.projections import Ball, HalfSpace
from fpmlib.nonexpansive import Intersection
from fpmlib.typing import NonexpansiveMap
from fpmlib.algorithms import *


class _Rotation(NonexpansiveMap):
    @property
    def ndim(self):
        return 2

    def __init__(self, sol: np.ndarray, alpha: float = 0.1):
        self._sol = sol.copy()
        self._M = np.array([[cos(alpha), -sin(alpha)], [sin(alpha), cos(alpha)]])

    def __call__(self, x):
        out = x - self._sol
        out = self._M.dot(out)
        out += self._sol
        return out
    
    def __contains__(self, x):
        return x == self._sol


class TestFindKrasnoselskiiMann(unittest.TestCase):
    def test_rotation(self):
        T = _Rotation(np.array([1, -2]))
        x0 = np.ones(2)
        x = find(T, x0, method='Krasnoselskii-Mann', tol=1e-8)
        self.assertIsNot(x, x0)
        np.testing.assert_equal(x0, np.ones(2))
        np.testing.assert_almost_equal(x, np.array([1, -2]), decimal=7)

    def test_customized_alpha(self):
        T = _Rotation(np.array([1, -2]))
        x0 = np.ones(2)
        x = find(T, np.ones(2), method='Krasnoselskii-Mann', tol=1e-8, options={'steps': itertools.repeat(0.1)})
        self.assertIsNot(x, x0)
        np.testing.assert_equal(x0, np.ones(2))
        np.testing.assert_almost_equal(x, np.array([1, -2]), decimal=7)

    def test_invalid_alpha(self):
        T = _Rotation(np.array([1, -2]))
        x0 = np.ones(2)
        x = find(T, np.ones(2), method='Krasnoselskii-Mann', options={'steps': itertools.repeat(0), 'maxiter': 1000})
        self.assertIsNot(x, x0)
        np.testing.assert_equal(x0, np.ones(2))
        np.testing.assert_almost_equal(x, np.ones(2))


class TestFindHishinuma2015(unittest.TestCase):
    def test_rotation(self):
        T = _Rotation(np.array([1, -2]))
        x0 = np.ones(2)
        x = find(T, x0, method='Hishinuma2015', tol=1e-8)
        self.assertIsNot(x, x0)
        np.testing.assert_equal(x0, np.ones(2))
        np.testing.assert_almost_equal(x, np.array([1, -2]), decimal=7)


class TestFindHalpern(unittest.TestCase):
    def test_rotation(self):
        T = _Rotation(np.array([1, -2]))
        x0 = np.ones(2)
        x = find(T, x0, method='Halpern', tol=1e-8)
        self.assertIsNot(x, x0)
        np.testing.assert_equal(x0, np.ones(2))
        np.testing.assert_almost_equal(x, np.array([1, -2]), decimal=7)

    def test_nearest(self):
        T = Intersection([
            HalfSpace(np.array([-1, 1]), 0),
            Ball(np.zeros(2), 1)
        ])
        x0 = np.array([5, 10])
        x = find(T, x0, method='Halpern', tol=1e-3)
        self.assertIsNot(x, x0)
        np.testing.assert_equal(x0, np.array([5, 10]))
        np.testing.assert_almost_equal(x, np.array([2 ** -0.5, 2 ** -0.5]), decimal=2)
