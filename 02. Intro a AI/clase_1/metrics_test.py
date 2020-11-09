from unittest import TestCase
import math

import numpy as np

from ejercicio_7 import BaseMetric, Precision, Recall, Accuracy, QueryMeanPrecision, QueryMeanPrecisionAtK


class NormTestCase(TestCase):

    def test_BaseMetric(self):
        truth = np.array([1, 1, 0, 1, 1, 1, 0, 0, 0, 1])
        prediction = np.array([1, 1, 1, 1, 0, 0, 1, 1, 0, 0])
        query_ids = np.array([1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4])
        predicted_rank = np.array([0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4, 0, 1, 2, 3])
        truth_relevance = np.array([1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1])
        k=3

        expected = np.array([0.5, 0.5, 0.4, 0.5, 0.5])
        # Loop over list.
        i = 0
        result = np.zeros(5)
        for Metric in BaseMetric.__subclasses__():
            print('Metric')
            p = Metric(truth=truth, prediction=prediction, query_ids=query_ids, predicted_rank=predicted_rank, truth_relevance=truth_relevance,k=3 )
            result[i] = p()
            i = i+1
        np.testing.assert_equal(expected, result)

