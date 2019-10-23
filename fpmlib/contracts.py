#!/usr/bin/env python3
"""
``fpmlib.contracts`` module
---------------------------

This module provides functions for the validation of the use of ``fpmlib`` package.
"""

from typing import Any, Callable
from .typing import *
__all__ = [
    'check_fixed_point_map', 'check_nonexpansive_map', 'check_firmly_nonexpansive_map',
    'check_metric_projection',
]


def __check_instance(t: type) -> Callable[[Any], None]:
    def T(o: Any) -> None:
        if not isinstance(o, t):
            raise ValueError('Expected %s, but got %s' % (t.__name__, type(o).__name__))
    T.__doc__ = 'Check if the given argument is an instance of %s class.' % t.__name__
    return T


check_fixed_point_map = __check_instance(FixedPointMap)
check_nonexpansive_map = __check_instance(NonexpansiveMap)
check_firmly_nonexpansive_map = __check_instance(FirmlyNonexpansiveMap)
check_metric_projection = __check_instance(MetricProjection)
