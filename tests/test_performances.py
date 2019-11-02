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
