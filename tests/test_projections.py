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

    def test_copying_parameters(self):
        lb = np.array([1, 2, 3, 4, 5])
        ub = np.array([6, 7, 8, 9, 10])
        p = Box(lb, ub)
        self.assertIsNot(lb, p._lb)
        self.assertIsNot(ub, p._ub)

    def test_contains1(self):
        p = Box(3, 5)
        self.assertFalse(np.array([2]) in p)
        self.assertTrue(np.array([3]) in p)
        self.assertTrue(np.array([4]) in p)
        self.assertTrue(np.array([5]) in p)
        self.assertFalse(np.array([6]) in p)

    def test_contains2(self):
        p = Box(np.array([1, 2]), np.array([3, 4]))
        self.assertFalse(np.array([0, 1]) in p)
        self.assertFalse(np.array([0, 2]) in p)
        self.assertFalse(np.array([0, 3]) in p)
        self.assertFalse(np.array([0, 4]) in p)
        self.assertFalse(np.array([0, 5]) in p)
        self.assertFalse(np.array([1, 1]) in p)
        self.assertTrue(np.array([1, 2]) in p)
        self.assertTrue(np.array([1, 3]) in p)
        self.assertTrue(np.array([1, 4]) in p)
        self.assertFalse(np.array([1, 5]) in p)
        self.assertFalse(np.array([2, 1]) in p)
        self.assertTrue(np.array([2, 2]) in p)
        self.assertTrue(np.array([2, 3]) in p)
        self.assertTrue(np.array([2, 4]) in p)
        self.assertFalse(np.array([2, 5]) in p)
        self.assertFalse(np.array([3, 1]) in p)
        self.assertTrue(np.array([3, 2]) in p)
        self.assertTrue(np.array([3, 3]) in p)
        self.assertTrue(np.array([3, 4]) in p)
        self.assertFalse(np.array([3, 5]) in p)
        self.assertFalse(np.array([4, 1]) in p)
        self.assertFalse(np.array([4, 2]) in p)
        self.assertFalse(np.array([4, 3]) in p)
        self.assertFalse(np.array([4, 4]) in p)
        self.assertFalse(np.array([4, 5]) in p)


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

    def test_copying_parameters(self):
        w = np.ones(10)
        p = HalfSpace(w, 1)
        self.assertIsNot(w, p._w)

    def test_contains1(self):
        p = HalfSpace(np.array([1]), 1)
        self.assertTrue(np.array([0]) in p)
        self.assertTrue(np.array([1]) in p)
        self.assertFalse(np.array([2]) in p)

    def test_contains2(self):
        p = HalfSpace(np.array([1, -1]), 1)
        self.assertTrue(np.array([0, 0]) in p)
        self.assertTrue(np.array([0.5, -0.5]) in p)
        self.assertFalse(np.array([1, -1]) in p)


class TestBall(unittest.TestCase):
    def test_behavior(self):
        p = Ball(np.array([2., -3.]), 1.)
        np.testing.assert_equal(
            p(np.array([2., -3.])), np.array([2., -3.]))
        np.testing.assert_equal(
            p(np.array([1., -3.])), np.array([1., -3.]))
        np.testing.assert_equal(
            p(np.array([3., -3.])), np.array([3., -3.]))
        np.testing.assert_equal(
            p(np.array([2., -4.])), np.array([2., -4.]))
        np.testing.assert_equal(
            p(np.array([2., -2.])), np.array([2., -2.]))
        np.testing.assert_almost_equal(
            p(np.array([2., -1.])), np.array([2., -2.]))
        np.testing.assert_almost_equal(
            p(np.array([5., -3.])), np.array([3., -3.]))
        np.testing.assert_almost_equal(
            p(np.array([3., -2.])), np.array([2., -3.]) + 2 ** -0.5)

    def test_c_changed(self):
        c = np.array([1., 1.])
        p = Ball(c, 1.)
        c[0] = 3.
        np.testing.assert_equal(
            p(np.array([1., 1.])), np.array([1., 1.]))

    def test_reallocation(self):
        p = Ball(np.array([2., -3.]), 1.)
        x = np.array([2., -3.])
        self.assertIsNot(p(x), x)
        x = np.array([3., -2.])
        self.assertIsNot(p(x), x)

    def test_negative_radius(self):
        with self.assertRaisesRegex(ValueError, 'must be a positive real'):
            Ball(np.zeros(10), -1)

    def test_matrix(self):
        with self.assertRaisesRegex(ValueError, 'must be a vector'):
            Ball(np.ones([5, 5]), 1)

    def test_ndim(self):
        p = Ball(np.ones(5), 1)
        self.assertEqual(p.ndim, 5)

    def test_copying_parameters(self):
        c = np.ones(10)
        p = Ball(c, 1)
        self.assertIsNot(c, p._c)

    def test_contains(self):
        p = Ball(np.array([1, -2]), 1)
        self.assertTrue(np.array([1, -2]) in p)
        self.assertTrue(np.array([1, -1]) in p)
        self.assertFalse(np.array([1, -0.9]) in p)
