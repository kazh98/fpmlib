#!/usr/bin/env python3
"""
Types appearing in ``fpmlib`` package
-------------------------------------

``fpmlib`` module is provided with object-oriented design using the abstract base classes infrastructure (``abc`` package).
``fpmlib.typing`` module provides the abstract base classes used to express each class provided in ``fpmlib`` package.
The following items are automatically loaded when ``fpmlib`` package is imported.
"""

from __future__ import annotations
import numpy as np
from abc import abstractmethod
from collections.abc import Callable, Container
from typing import Optional
__all__ = ['FixedPointMap', 'NonexpansiveMap', 'FirmlyNonexpansiveMap', 'MetricProjection']


class FixedPointMap(Callable, Container):
    """
    An abstract base class that expresses a mapping :math:`T` from a Hilbert space :math:`H` onto itself.
    This is a superclass of all mappings provided from the `fpmlib` package.
    """

    @property
    @abstractmethod
    def ndim(self) -> Optional[int]:
        r"""
        Number of vector dimensions which this mapping deal with.
        If the value is ``None``, this mapping accepts any vector in arbitrary dimension.
        """

        raise NotImplementedError()

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        r"""
        Map the given point :math:`x\in H` to :math:`T(x)`.
        """

        raise NotImplementedError()

    @abstractmethod
    def __contains__(self, x: np.ndarray) -> bool:
        r"""
        Return ``True`` if the given point :math:`x` is a fixed point of this mapping :math:`T`, that is, :math:`x\in\mathrm{Fix}(T):=\{u\in H:T(u)=u\}`.
        Otherwise, return ``False``.
        """

        raise NotImplementedError()


class NonexpansiveMap(FixedPointMap):
    r"""
    An abstract base class that expresses a nonexpansive mapping, that is, a mapping :math:`T:H\to H` which satisfies :math:`\|T(x)-T(y)\|\le\|x-y\|` for any :math:`x, y\in H`.
    """


class FirmlyNonexpansiveMap(NonexpansiveMap):
    r"""
    An abstract base class that expresses a firmly nonexpansive mapping, that is, a mapping :math:`T:H\to H` which satisfies :math:`\|T(x)-T(y)\|^2+\|(\mathrm{Id}-T)(x)-(\mathrm{Id}-T)(y)\|^2\le\|x-y\|^2` for any :math:`x, y\in H`.
    """

    @staticmethod
    def from_nonexpansive(T: NonexpansiveMap, alpha: float=0.5) -> FirmlyNonexpansiveMap:
        # TODO: Test this method
        # return __FirmlyNonexpansiveMap(T, alpha)
        return None


class __FirmlyNonexpansiveMap(FirmlyNonexpansiveMap):
    def __init__(self, T: NonexpansiveMap, alpha: float=0.5):
        assert 0 < alpha <= 0.5, 'Argument alpha must be between 0 and 0.5.'

        self.__T = T
        self.__alpha = alpha

    def __call__(self, x: np.ndarray) -> np.ndarray:
        u = self.__T(x)
        if u is x:
            u = x.copy()
        u *= self.__alpha
        u += (1 - self.__alpha) * x
        return u

    def __contains__(self, x: np.ndarray) -> bool:
        return x in self.__T


class MetricProjection(FirmlyNonexpansiveMap):
    r"""
    An abstract base class that expresses a metric projection onto some nonempty, closed, convex subset of :math:`H`, that is, a mapping :math:`T:H\to H` which satisfies :math:`T(x)\in\mathrm{Fix}(T)` and :math:`\|T(x)-x\|=\inf_{y\in\mathrm{Fix}(T)}\|x-y\|` for any :math:`x\in H`.
    """
