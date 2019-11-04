import numpy as np
import unittest
from fpmlib.projections import HalfSpace, Box, Ball
from fpmlib.nonexpansive import *


class TestIntersection(unittest.TestCase):
    def test_1d(self):
        # p := [-1, 1]
        p = Intersection([HalfSpace(np.array([1]), 1), HalfSpace(np.array([-1]), 1)])
        self.assertFalse(np.array([-2]) in p)
        np.testing.assert_almost_equal(p(np.array([-2])), np.array([-1.5]))
        self.assertTrue(np.array([-1]) in p)
        np.testing.assert_equal(p(np.array([-1])), np.array([-1]))
        self.assertTrue(np.array([0]) in p)
        np.testing.assert_equal(p(np.array([0])), np.array([0]))
        self.assertTrue(np.array([1]) in p)
        np.testing.assert_equal(p(np.array([1])), np.array([1]))
        self.assertFalse(np.array([2]) in p)
        np.testing.assert_almost_equal(p(np.array([2])), np.array([1.5]))

    def test_ndim(self):
        p = Intersection([HalfSpace(np.array([1, 0]), 1), HalfSpace(np.array([0, 1]), 1)])
        self.assertTrue(np.array([0, 0]) in p)
        np.testing.assert_equal(p(np.array([0, 0])), np.array([0, 0]))
        self.assertTrue(np.array([1, 0]) in p)
        np.testing.assert_equal(p(np.array([1, 0])), np.array([1, 0]))
        self.assertTrue(np.array([0, 1]) in p)
        np.testing.assert_equal(p(np.array([0, 1])), np.array([0, 1]))
        self.assertTrue(np.array([1, 1]) in p)
        np.testing.assert_equal(p(np.array([1, 1])), np.array([1, 1]))
        self.assertFalse(np.array([2, 2]) in p)
        np.testing.assert_almost_equal(p(np.array([2, 2])), np.array([1.5, 1.5]))

    def test_ndim_none(self):
        p = Intersection([Box(0), HalfSpace(np.array([1, 1]), 1)])
        self.assertEqual(p.ndim, 2)
        self.assertFalse(np.array([-2, -2]) in p)
        self.assertTrue(np.array([0, 0]) in p)
        self.assertFalse(np.array([1, 1]) in p)

    def test_ndim_invalid(self):
        with self.assertRaises(ValueError):
            Intersection([
                Box(np.array([1, 2])),
                Box(np.array([1, 2, 3])),
            ])

class TestComposition(unittest.TestCase):
    def test_1d(self):
        # p := [-1, 1]
        p = Composition([HalfSpace(np.array([1]), 1), HalfSpace(np.array([-1]), 1)])
        self.assertFalse(np.array([-2]) in p)
        np.testing.assert_almost_equal(p(np.array([-2])), np.array([-1]))
        self.assertTrue(np.array([-1]) in p)
        np.testing.assert_equal(p(np.array([-1])), np.array([-1]))
        self.assertTrue(np.array([0]) in p)
        np.testing.assert_equal(p(np.array([0])), np.array([0]))
        self.assertTrue(np.array([1]) in p)
        np.testing.assert_equal(p(np.array([1])), np.array([1]))
        self.assertFalse(np.array([2]) in p)
        np.testing.assert_almost_equal(p(np.array([2])), np.array([1]))

    def test_nd(self):
        p = Composition([Ball(np.array([-1, 0, 0, 0, 0]), 1), Ball(np.array([1, 0, 0, 0, 0]), 1)])
        self.assertTrue(np.zeros(5) in p)
        self.assertFalse(np.array([-1., 0., 0., 0., 0.]) in p)
        np.testing.assert_almost_equal(p(np.array([-1., 0., 0., 0., 0.])), np.zeros(5))
        self.assertFalse(np.array([1., 0., 0., 0., 0.]) in p)
        np.testing.assert_almost_equal(p(np.array([1., 0., 0., 0., 0.])), np.zeros(5))

    def test_ndim_none(self):
        p = Composition([Box(0), HalfSpace(np.array([1, 1]), 1)])
        self.assertEqual(p.ndim, 2)
        self.assertFalse(np.array([-2, -2]) in p)
        self.assertTrue(np.array([0, 0]) in p)
        self.assertFalse(np.array([1, 1]) in p)

    def test_ndim_invalid(self):
        with self.assertRaises(ValueError):
            Composition([
                Box(np.array([1, 2])),
                Box(np.array([1, 2, 3])),
            ])

    def test_nonexpansive_error(self):
        nonexp1 = Intersection([Box(0)])
        nonexp2 = Box(0)
        with self.assertRaises(ValueError):
            Composition([nonexp1, nonexp2])
