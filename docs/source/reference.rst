API Reference
=============
In this page, we describes each API provided from the ``fpmlib`` package.
Here, :math:`\mathbb{R}` denotes the set of all ``float`` values and :math:`\mathbb{R}^N` denotes the set of all ``ndarray`` instances whose shape is ``[N]``.
The standard inner product :math:`\langle\cdot,\cdot\rangle:\mathbb{R}^N\times\mathbb{R}^N\to\mathbb{R}` corresponds to :math:`\mathtt{numpy.inner}(\cdot,\cdot)` in implementation, and its induced norm :math:`\|x\|:=\langle x,x\rangle^\frac{1}{2}` for given ``ndarray`` vector :math:`x` corresponds to :math:`\mathtt{numpy.linalg.norm}(x)`.

.. automodule:: fpmlib.typing
    :special-members:
.. automodule:: fpmlib.projections
.. automodule:: fpmlib.nonexpansive
.. automodule:: fpmlib.algorithms
.. automodule:: fpmlib.contracts
