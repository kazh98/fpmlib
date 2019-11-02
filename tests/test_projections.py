import numpy as np
import unittest
from fpmlib.projections import *


class TestBox(unittest.TestCase):
    def test_behavior(self):
        p = Box(np.array([-1., -2.]), np.array([3., 4.]))
        self.assertTrue(np.array([0., 0.]) in p)
        np.testing.assert_equal(
            p(np.array([0., 0.])), np.array([0., 0.]))
        self.assertTrue(np.array([-1., -2.]) in p)
        np.testing.assert_equal(
            p(np.array([-1., -2.])), np.array([-1., -2.]))
        self.assertTrue(np.array([3., 4.]) in p)
        np.testing.assert_equal(
            p(np.array([3., 4.])), np.array([3., 4.]))
        self.assertFalse(np.array([-3., -4.]) in p)
        np.testing.assert_equal(
            p(np.array([-3., -4.])), np.array([-1., -2.]))
        self.assertFalse(np.array([5., 6.]) in p)
        np.testing.assert_equal(
            p(np.array([5., 6.])), np.array([3., 4.]))

    def test_lb_only(self):
        p = Box(lb=np.ones(2))
        self.assertFalse(np.zeros(2) in p)
        np.testing.assert_equal(
            p(np.zeros(2)), np.ones(2))
        self.assertTrue(np.ones(2) in p)
        np.testing.assert_equal(
            p(np.ones(2)), np.ones(2))
        self.assertTrue(np.full(2, 2) in p)
        np.testing.assert_equal(
            p(np.full(2, 2)), np.full(2, 2))

    def test_ub_only(self):
        p = Box(ub=np.ones(2))
        self.assertTrue(np.zeros(2) in p)
        np.testing.assert_equal(
            p(np.zeros(2)), np.zeros(2))
        self.assertTrue(np.ones(2) in p)
        np.testing.assert_equal(
            p(np.ones(2)), np.ones(2))
        self.assertFalse(np.full(2, 2) in p)
        np.testing.assert_equal(
            p(np.full(2, 2)), np.ones(2))

    def test_scalar(self):
        p = Box(0, 1)
        self.assertFalse(np.full(10, -1) in p)
        np.testing.assert_equal(
            p(np.full(10, -1)), np.full(10, 0))
        self.assertTrue(np.full(10, 0) in p)
        np.testing.assert_equal(
            p(np.full(10, 0)), np.full(10, 0))
        self.assertTrue(np.full(10, 1) in p)
        np.testing.assert_equal(
            p(np.full(10, 1)), np.full(10, 1))
        self.assertFalse(np.full(10, 2) in p)
        np.testing.assert_equal(
            p(np.full(10, 2)), np.full(10, 1))

    def test_reallocation(self):
        p = Box(np.zeros(2), np.ones(2))
        x = np.array([0.5, 0.5])
        self.assertIsNot(p(x), x)

    def test_lb_dim(self):
        p = Box(np.zeros(5))
        self.assertEqual(p.ndim, 5)

    def test_ub_dim(self):
        p = Box(-12, np.zeros(5))
        self.assertEqual(p.ndim, 5)
    
    def test_none_dim(self):
        p = Box(3, 5)
        self.assertIsNone(p.ndim)


class TestHalfSpace(unittest.TestCase):
    def test_behavior(self):
        p = HalfSpace(np.array([1., 2.]), 3.)
        np.testing.assert_array_equal(
            p(np.array([0., 0.])), np.array([0., 0.]))
        np.testing.assert_array_equal(
            p(np.array([0., 1.5])), np.array([0., 1.5]))
        np.testing.assert_array_equal(
            p(np.array([3., 0.])), np.array([3., 0.]))
        np.testing.assert_array_equal(
            p(np.array([3., 0.])), np.array([3., 0.]))
        np.testing.assert_array_almost_equal(
            p(np.array([4., 2.])), np.array([3., 0.]))

    def test_behavior_negative(self):
        p = HalfSpace(np.array([-1., -1.]), -1.)
        np.testing.assert_array_almost_equal(
            p(np.array([0., 0.])), np.array([0.5, 0.5]))
        np.testing.assert_array_equal(
            p(np.array([0.5, 0.5])), np.array([0.5, 0.5]))
        np.testing.assert_array_equal(
            p(np.array([1., 1.])), np.array([1., 1.]))
        np.testing.assert_array_equal(
            p(np.array([0., 1.])), np.array([0., 1.]))

    def test_w_changed(self):
        w = np.array([1., 1.])
        p = HalfSpace(w, 3.)
        w[0] = 3.
        np.testing.assert_equal(
            p(np.array([3., 0.])), np.array([3., 0.]))

    def test_reallocation(self):
        p = HalfSpace(np.array([1., 1.]), 3.)
        x = np.array([1., 1.])
        self.assertIsNot(p(x), x)
        x = np.array([3., 3.])
        self.assertIsNot(p(x), x)

    def test_nonzero(self):
        with self.assertRaisesRegex(ValueError, 'must be a nonzero vector'):
            HalfSpace(np.zeros(100), 1)

    def test_matrix(self):
        with self.assertRaisesRegex(ValueError, 'must be a vector'):
            HalfSpace(np.ones([5, 5]), 1)

    def test_ndim(self):
        p = HalfSpace(np.ones(5), 1)
        self.assertEqual(p.ndim, 5)
