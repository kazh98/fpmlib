#!/usr/bin/env python3
"""
Nonexpansive mappings
---------------------

The following items are automatically loaded when ``fpmlib`` package is imported.
"""

import numpy as np
from typing import Iterable
from .typing import FixedPointMap, FirmlyNonexpansiveMap, NonexpansiveMap
from .contracts import check_firmly_nonexpansive_map, check_nonexpansive_map
__all__ = ['Intersection', 'Composition']


class Intersection(NonexpansiveMap):
    r"""
    A nonexpansive mapping whose fixed point set coincides with the intersection of the fixed point sets of given nonexpansive mappings.
    The generated mapping computes the barycenter of each point transformed by given mappings, i.e., for given :math:`T_i\ (i=1,2,\ldots,K)` and for any :math:`x\in H`, it computes

    .. math::
        T(x):=\frac{1}{K}\sum_{i=1}^K T_i(x).

    This construction method is based on Propositions 4.9 and 4.47 in [Bauschke2017]_.

    :param maps: A list of nonexpansive mappings.
    """

    @property
    def ndim(self):
        return self._ndim

    def __init__(self, maps: Iterable[NonexpansiveMap]):
        maps = tuple(maps)
        if len(maps) < 1:
            raise ValueError('At least one mapping must be given.')
        ndim = ([m.ndim for m in maps if isinstance(m, FixedPointMap) and m.ndim] + [None])[0]
        for m in maps:
            check_nonexpansive_map(m, ndim)
        
        self._maps = maps
        self._ndim = ndim

    def __call__(self, x):
        return np.average([m(x) for m in self._maps])

    def __contains__(self, x):
        return all(x in m for m in self._maps)


class Composition(NonexpansiveMap):
    r"""
    A nonexpansive mapping whose fixed point set coincides with the intersection of the fixed point sets of given nonexpansive mappings.
    The generated mapping is the composition of given mappings, i.e., for given :math:`T_i\ (i=1,2,\ldots,K)`, it is defined by

    .. math::
        T:=T_1\circ T_2\circ \ldots \circ T_K.

    Except for the last element, each element in parameter ``maps`` must be ``FirmlyNonexpansiveMap``.
    If the intersection of the fixed point sets of given nonexpansive mappings is empty, ``__contains__`` operator may not return a correct value.
    This construction method is based on Propositions 4.9 and 4.49 in [Bauschke2017]_.

    :param maps: A list of nonexpansive mappings.
    """

    @property
    def ndim(self):
        return self._ndim
    
    def __init__(self, maps: Iterable[NonexpansiveMap]):
        maps = tuple(maps)
        if len(maps) < 1:
            raise ValueError('At least one mapping must be given.')
        ndim = ([m.ndim for m in maps if isinstance(m, FixedPointMap) and m.ndim] + [None])[0]
        for m in maps[:-1]:
            check_firmly_nonexpansive_map(m, ndim)
        check_nonexpansive_map(maps[-1], ndim)

        self._maps = maps
        self._ndim = ndim
    
    def __call__(self, x):
        out = x
        for m in reversed(self._maps):
            out = m(out)
        return out
    
    def __contains__(self, x):
        return all(x in m for m in self._maps)
