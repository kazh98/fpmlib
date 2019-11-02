import numpy as np
import unittest
from fpmlib.projections import HalfSpace, Box
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
