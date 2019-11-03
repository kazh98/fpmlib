import unittest
from timeit import timeit


class TestNumPyComputation(unittest.TestCase):
    def test_preallocation(self):
        worse = timeit("c[:] = a; c += b", "import numpy as np; a, b, c = np.random.rand(100), np.random.rand(100), np.empty(100)")
        better = timeit("c = a + b", "import numpy as np; a, b = np.random.rand(100), np.random.rand(100)")
        self.assertLess(better, worse)

    def test_inplace(self):
        worse = timeit("a += b", "import numpy as np; a, b = np.random.rand(100), np.random.rand(100)")
        better = timeit("a[:] = a + b", "import numpy as np; a, b = np.random.rand(100), np.random.rand(100)")
        self.assertLess(better, worse)


class TestFindFunction(unittest.TestCase):
    def test_mann_vs_hishinuma(self):
        import numpy as np
        from math import sin, cos
        from fpmlib.typing import NonexpansiveMap

        class _Rotation(NonexpansiveMap):
            @property
            def ndim(self):
                return 2

            def __init__(self, sol: np.ndarray, alpha: float = 0.1):
                self._sol = sol.copy()
                self._M = np.array([[cos(alpha), -sin(alpha)], [sin(alpha), cos(alpha)]])

            def __call__(self, x):
                out = x - self._sol
                out = self._M.dot(out)
                out += self._sol
                return out
            
            def __contains__(self, x):
                return x == self._sol

        ns = {
            "T": _Rotation([1, -2]),
            "x0": np.ones(2),
        }
        worse = timeit("find(T, x0, method='Hishinuma2015', tol=1e-3)", "from fpmlib.algorithms import find", globals=ns, number=10)
        better = timeit("find(T, x0, method='Krasnoselskii-Mann', tol=1e-3)", "from fpmlib.algorithms import find", globals=ns, number=10)
        self.assertLess(better, worse)
