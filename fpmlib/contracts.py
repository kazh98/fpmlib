#!/usr/bin/env python3
"""
``fpmlib.contracts`` module
---------------------------

This module provides functions for the validation of the use of ``fpmlib`` package.
"""

from typing import Any, Optional, Callable, cast
from .typing import *
__all__ = [
    'check_fixed_point_map', 'check_nonexpansive_map', 'check_firmly_nonexpansive_map',
    'check_metric_projection',
]


def __check_instance(t: FixedPointMap) -> Callable[[Any], None]:
    def T(o: Any, ndim: Optional[int]=None) -> None:
        if not isinstance(o, t):
            raise ValueError('Expected %s, but got %s' % (t.__name__, type(o).__name__))
        o = cast(FixedPointMap, o)
        if ndim is not None and o.ndim is not None and o.ndim != ndim:
            raise ValueError('Expected %s on %d-dimensional Euclidean space, but got one on %d-dimensional space' % (t.__name__, ndim, o.ndim))
    T.__doc__ = 'Check if the given argument is an instance of ``%s`` class.' % t.__name__ + r"""

    :param o: An object to be validated.
    :param ndim: A number of vector dimensions which is expected as acceptable one to the given map.
        If ``None`` is specified, this function will not validate the acceptable number of dimension to the given map.
    """
    return T


check_fixed_point_map = __check_instance(FixedPointMap)
check_nonexpansive_map = __check_instance(NonexpansiveMap)
check_firmly_nonexpansive_map = __check_instance(FirmlyNonexpansiveMap)
check_metric_projection = __check_instance(MetricProjection)
