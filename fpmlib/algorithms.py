#!/usr/bin/env python3
"""
Algorithms
----------

The following items are automatically loaded when ``fpmlib`` package is imported.
"""

import numpy as np
import itertools
from typing import Any, Optional, Iterable, Dict
from .typing import NonexpansiveMap
from .contracts import check_nonexpansive_map
__all__ = ['find']


def _find_krasnoselskii_mann(
    T: NonexpansiveMap,
    x0: np.ndarray,
    tol: float,
    maxiter: Optional[int] = None,
    steps: Optional[Iterable[float]] = None
) -> np.ndarray:
    if steps is None:
        steps = itertools.repeat(0.5)
    if maxiter is not None:
        steps = itertools.islice(steps, maxiter)

    x = x0.copy()
    for step in steps:
        Tx = T(x)
        if np.linalg.norm(Tx - x) < tol:
            break
        Tx *= step
        x *= 1. - step
        x += Tx

    return x


def _find_hishinuma2015(
    T: NonexpansiveMap,
    x0: np.ndarray,
    tol: float,
    maxiter: Optional[int] = None,
    steps: Optional[Iterable[float]] = None,
    beta: Optional[Iterable[float]] = None
) -> np.ndarray:
    if steps is None:
        steps = itertools.repeat(0.5)
    if maxiter is not None:
        steps = itertools.islice(steps, maxiter)
    if beta is None:
        beta = map(lambda n: n ** -1.001, itertools.count(1))

    Tx = T(x0)
    x, d = x0.copy(), Tx - x0
    for step, b in zip(steps, beta):
        if np.linalg.norm(Tx - x) < tol:
            break
        # d = (Tx - x) + b * d
        d *= b
        Tx -= x
        d += Tx
        # y = x + d
        # x = x + step * (y - x)
        x += step * d
        #
        Tx = T(x)

    return x


def _find_halpern(
    T: NonexpansiveMap,
    x0: np.ndarray,
    tol: float,
    maxiter: Optional[int] = None,
    steps: Optional[Iterable[float]] = None
):
    if steps is None:
        steps = map(lambda n: 1 / n, itertools.count(1))
    if maxiter is not None:
        steps = itertools.islice(steps, maxiter)

    x = x0.copy()
    for step in steps:
        Tx = T(x)
        if np.linalg.norm(Tx - x) < tol:
            break
        # x = step * x0 + (1 - step) * Tx
        x = step * x0
        Tx *= 1 - step
        x += Tx

    return x


def find(
    T: NonexpansiveMap,
    x0: np.ndarray,
    method: str = 'Krasnoselskii-Mann',
    tol: float = 1e-7,
    options: Dict[str, Any] = {}
) -> np.ndarray:
    r"""
    Find a fixed point of given nonexpansive mapping.

    :param T: A nonexpansive mapping whose fixed point is desired to be found.
    :param x0: An initial point.
    :param method: Name of method to be used. We can use one of the following:

        ``Halpern``
            Halpern's algorithm ([Halpern1967]_).
            This method finds the nearest fixed point to the initial point, i.e., :math:`x^\star\in\mathrm{Fix}(T)` such that :math:`\|x^\star-x_0\|=\inf_{x\in\mathrm{Fix}(T)}\|x-x_0\|`.
        ``Hishinuma2015``
            Accelerated Krasnosel'skii-Mann algorithm based on conjugate gradient method ([Hishinuma2015]_).
        ``Krasnoselskii-Mann`` (default)
            Krasnosel'skii-Mann algorithm ([Krasnoselskii1955]_, [Mann1953]_).

    :param tol: Error tolerance, i.e., for the obtained solution :math:`x^\star`, :math:`\|x^\star-T(x^\star)\|<\mathtt{tol}` will be guaranteed.
    :param options: A dictionary passed to the solver. We can give the following parameters:
        
        maxiter: int
            Maximum number of iterations.
        steps: Iterable[float]
            A step size sequence.
            When ``method = 'Krasnoselskii-Mann'`` or its variant ``method = 'Hishinuma2015'``, it is used as the sequence :math:`\{\alpha_k\}\subset(0, 1)` for the Krasnosel'skii-Mann iteration :math:`x_{k+1}:=x_k+\alpha_k(T(x_k)-x_k)\ (k\in\mathbb{N})`.
            When ``method = 'Halpern'``, it is used as the sequence :math:`\{\lambda_k\subset(0, 1)\}` for the Halpern's iteration :math:`x_{k+1}:=\lambda_k x_0+(1-\lambda_k)T(x_k)`.
        beta: Iterable[float]
            A step size sequence to be used as an acceleration parameter.
            When ``method = 'Hishinuma2015'``, it is passed to Algorithm 3.1 in [Hishinuma2015]_ as the parameter :math:`\{\beta_n\}`.
        
    :return: the obtained solution.
    """

    if len(x0.shape) != 1:
        raise ValueError('x0 must be a vector.')
    check_nonexpansive_map(T, x0.shape[0])

    if method == 'Halpern':
        return _find_halpern(T, x0, tol, **options)
    if method == 'Hishinuma2015':
        return _find_hishinuma2015(T, x0, tol, **options)
    if method == 'Krasnoselskii-Mann':
        return _find_krasnoselskii_mann(T, x0, tol, **options)
    raise ValueError('Unknown algorithm %s is specified.' % method)
