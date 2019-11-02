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
        An ``ndarray`` vector whose element expresses the lower bound corresponding to each dimension.
        If a ``float`` value is specified, it is dealt with as the vector whose all elements are set as the given value.
        If ``None`` is specified, the fixed point set of the created mapping is unbounded below.
    :param ub:
        An ``ndarray`` vector whose element expresses the upper bound corresponding to each dimension.
        If a ``float`` value is specified, it is dealt with as the vector whose all elements are set as the given value.
        If ``None`` is specified, the fixed point set of the created mapping is unbounded above.
    """

    @property
    def ndim(self):
        if isinstance(self._lb, np.ndarray):
            return self._lb.shape[0]
        if isinstance(self._ub, np.ndarray):
            return self._ub.shape[0]
        return None

    def __init__(self, lb: Optional[Union[np.ndarray, float]] = None, ub: Optional[Union[np.ndarray, float]] = None):
        if isinstance(lb, np.ndarray) and isinstance(ub, np.ndarray) and lb.shape != ub.shape:
            raise ValueError('Vectors lb and ub must have the same number of dimensions')
        if isinstance(lb, np.ndarray):
            lb = lb.copy()
        if isinstance(ub, np.ndarray):
            ub = ub.copy()
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
        An ``ndarray`` vector which defines the half-space as its parameter :math:`w`.
    :param d:
        A ``float`` value which defines the half-space as its parameter :math:`d`.
    """

    @property
    def ndim(self):
        return self._w.shape[0]

    def __init__(self, w: np.ndarray, d: float):
        if not isinstance(w, np.ndarray) or len(w.shape) != 1:
            raise ValueError('Parameter w must be a vector.')
        l = np.linalg.norm(w)
        if l == 0:
            raise ValueError('Parameter w must be a nonzero vector.')
        self._w = w / l
        self._d = d / l

    def __call__(self, x: np.ndarray) -> np.ndarray:
        det = self._d - np.inner(self._w, x)
        if det >= 0:
            y = x.copy()
        else:
            y = det * self._w
            y += x
        return y

    def __contains__(self, x: np.ndarray) -> bool:
        return (self._d - np.inner(self._w, x)) >= 0


class Ball(MetricProjection):
    r"""
    The metric projection :math:`P_B` onto the closed ball with center :math:`c\in\mathbb{R}^N` and radius :math:`r\in\mathbb{R}`, that is

    .. math::
        B:=\{x\in\mathbb{R}^N:\|x-c\|\le r\}.

    :param c:
        An ``ndarray`` vector which expresses the center of the closed ball.
    :param r:
        A ``float`` value which expresses the radius of the closed ball.
    """

    @property
    def ndim(self) -> int:
        return self._c.shape[0]

    def __init__(self, c: np.ndarray, r: float):
        if r < 0:
            raise ValueError('Parameter `r` must be a positive real.')
        if not isinstance(c, np.ndarray) or len(c.shape) != 1:
            raise ValueError('Parameter `c` must be a vector.')
        self._c = c.copy()
        self._r = r

    def __call__(self, x: np.ndarray) -> np.ndarray:
        v = x - self._c
        d = np.linalg.norm(v)
        if d <= self._r:
            return x.copy()
        else:
            v *= self._r / d
            v += self._c
            return v

    def __contains__(self, x: np.ndarray) -> bool:
        raise np.linalg.norm(x - self._c) <= self._r
