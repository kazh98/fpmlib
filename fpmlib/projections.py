#!/usr/bin/env python3
"""
Metric projections
------------------

There are some convex sets onto which the metric projections can be computed explicitly.
``fpmlib.projections`` module provides them.
The following items are automatically loaded when ``fpmlib`` package is imported.
"""

import numpy as np
from typing import Optional, Union
from .typing import MetricProjection
__all__ = ['Box', 'HalfSpace', 'Ball']


class Box(MetricProjection):
    r"""
    The metric projection onto the orthotope defined with its lower and upper bound of each dimension.
    For given :math:`(\mathbf{lb}_i)_{i=1}^N\in\mathbb{R}^N` and :math:`(\mathbf{ub}_i)_{i=1}^N\in\mathbb{R}^N``, the fixed point set of the created mapping :math:`T` is

    .. math::
        \mathrm{Fix}(T)=\{(x_i)_{i=1}^N\in\mathbb{R}^N:\mathbf{lb}_i\le x_i\le\mathbf{ub}_i\ (i=1,2,\ldots,N)\}.

    For any ``ndarray`` vector :math:`x`, :math:`T(x)` is equivalent to :math:`\mathtt{np.clip}(x, \mathbf{lb}, \mathbf{ub})`.

    :param lb:
        The ``ndarray`` vector whose element expresses the lower bound corresponding to each dimension.
        If a ``float`` value is specified, it is dealt with as the vector whose all elements are set as the given value.
        If ``None`` is specified, the fixed point set of the created mapping is unbounded below.
    :param ub:
        The ``ndarray`` vector whose element expresses the upper bound corresponding to each dimension.
        If a ``float`` value is specified, it is dealt with as the vector whose all elements are set as the given value.
        If ``None`` is specified, the fixed point set of the created mapping is unbounded above.
    """

    def __init__(self, lb: Optional[Union[np.ndarray, float]] = None, ub: Optional[Union[np.ndarray, float]] = None):
        self._lb = lb
        self._ub = ub

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self._lb, self._ub)

    def __contains__(self, x: np.ndarray) -> bool:
        if self._lb is not None and not (self._lb <= x).all():
            return False
        if self._ub is not None and not (x <= self._ub).all():
            return False
        return True


class HalfSpace(MetricProjection):
    r"""
    The metric projection :math:`P_H` onto the closed half-space
    
    .. math::
        H:=\{x\in\mathbb{R}^N:\langle w, x\rangle\le d\},

    where :math:`w\in\mathbb{R}^N\setminus\{0\}` and :math:`d\in\mathbb{R}`.

    :param w:
        The ``ndarray`` vector which defines the half-space as its parameter :math:`w`.
    :param d:
        The ``float`` value which defines the half-space as its parameter :math:`d`.
    """

    def __init__(self, w: np.ndarray, d: float):
        raise NotImplementedError()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def __contains__(self, x: np.ndarray) -> bool:
        raise NotImplementedError()


class Ball(MetricProjection):
    r"""
    The metric projection :math:`P_B` onto the closed ball with center :math:`c\in\mathbb{R}^N` and radius :math:`r\in\mathbb{R}`, that is

    .. math::
        B:=\{x\in\mathbb{R}^N:\|x-c\|\le r\}.

    :param c:
        The ``ndarray`` vector which expresses the center of the closed ball.
    :param r:
        The ``float`` value which expresses the radius of the closed ball.
    """

    def __init__(self, w: np.ndarray, d: float):
        raise NotImplementedError()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def __contains__(self, x: np.ndarray) -> bool:
        raise NotImplementedError()
