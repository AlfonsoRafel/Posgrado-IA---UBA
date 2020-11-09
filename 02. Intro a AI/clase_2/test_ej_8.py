from unittest import TestCase
import math

import numpy as np

from ejercicio_8 import pca_scikit, pca


class PCATestCase(TestCase):

    def test_pca(self):
        x = np.array([[0.4, 4800, 5.5], [0.7, 12104, 5.2], [1, 12500, 5.5], [1.5, 7002, 4.0]])
        components = 3
        actual = pca(x, components)
        desired = pca_scikit(x, components)

        np.testing.assert_allclose(actual, desired)