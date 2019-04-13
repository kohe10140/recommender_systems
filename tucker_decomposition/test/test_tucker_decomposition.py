import unittest

import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal
from tensorly.decomposition import tucker

import tensor_decomposition


class TestTuckerDecompostion(unittest.TestCase):

    def test_init(self):
        list_shape = [[10, 20, 30]]
        list_rank = [[2, 5, 4]]
        list_expected = [([10, 20, 30], [2, 5, 4])]
        for shape, rank, expected in zip(list_shape, list_rank, list_expected):
            tr = tucker_decomposition.TuckerRecommendation(shape, rank)
            actual = (tr.shape, tr.rank)
            self.assertEqual(actual, expected)


    def test_fit_core_factors(self):
        np.random.seed(0)
        list_rank = [[2, 5, 4]]
        list_X_train = [np.random.random((10, 10, 10))]
        for rank, X_train in zip(list_rank, list_X_train):
            tr = tucker_decomposition.TuckerRecommendation(rank=rank)
            tr.fit(X_train)
            actual_core = tr.core
            actual_factors = tr.factors
            expect_core, expected_factors = tucker(X_train, rank=rank)

            self.assertEqual(actual_core.all(), expect_core.all())
            for i in range(len(rank)):
                self.assertEqual(actual_factors[i].all(), expected_factors[i].all())


if __name__ == '__main__':
    unittest.main()
