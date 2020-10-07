import unittest

import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal
from tensorly.decomposition import tucker

from tensor_decomposition import tensor_decomposition 


class TestTuckerDecompostion(unittest.TestCase):

    def test_init(self):
        list_shape = [[10, 20, 30]]
        list_rank = [[2, 5, 4]]
        list_expected = [([10, 20, 30], [2, 5, 4])]
        for shape, rank, expected in zip(list_shape, list_rank, list_expected):
            tr = tensor_decomposition.TuckerRecommendation(shape, rank)
            actual = (tr.shape, tr.rank)
            self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
