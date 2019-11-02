#!/usr/bin/env python3
import numpy as np
import unittest
from fpmlib.typing import *
from fpmlib.contracts import *


class IdentFixedPointMap(FixedPointMap):
    @property
    def ndim(self):
        return self._ndim

    def __init__(self, ndim=None):
        self._ndim = ndim

    def __call__(self, x):
        return x.copy()

    def __contains__(self, x):
        return True


class IdentMetricProjection(MetricProjection):
    @property
    def ndim(self):
        return self._ndim

    def __init__(self, ndim=None):
        self._ndim = ndim

    def __call__(self, x):
        return x.copy()

    def __contains__(self, x):
        return True


class TestCheckFixedPointMap(unittest.TestCase):
    def setUp(self):
        self.T = IdentFixedPointMap()
        self.P = IdentMetricProjection()
        self.N = IdentMetricProjection(10)

    def test_positive1(self):
        check_fixed_point_map(self.T)

    def test_positive2(self):
        check_fixed_point_map(self.P)

    def test_positive3(self):
        check_metric_projection(self.P)

    def test_positive4(self):
        check_nonexpansive_map(self.P)

    def test_positive5(self):
        check_firmly_nonexpansive_map(self.P)

    def test_negative1(self):
        with self.assertRaisesRegex(ValueError, r"^Expected MetricProjection,"):
            check_metric_projection(self.T)

    def test_negative2(self):
        with self.assertRaisesRegex(ValueError, r"^Expected FixedPointMap,"):
            check_fixed_point_map(None)

    def test_negative3(self):
        with self.assertRaisesRegex(ValueError, r"^Expected NonexpansiveMap,"):
            check_nonexpansive_map(self.T)

    def test_negative4(self):
        with self.assertRaisesRegex(ValueError, r"^Expected FirmlyNonexpansiveMap,"):
            check_firmly_nonexpansive_map(self.T)

    def test_dim1(self):
        check_metric_projection(self.N)

    def test_dim2(self):
        check_metric_projection(self.N, 10)

    def test_dim3(self):
        with self.assertRaisesRegex(ValueError, 'Expected 10-dimensional vector, but got 15-dimensional one'):
            check_metric_projection(self.N, 15)

    def test_dim4(self):
        check_metric_projection(self.P, 15)
