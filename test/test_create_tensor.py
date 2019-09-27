import unittest

import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal
from tensorly.decomposition import tucker

from recommender_systems.tensor_decomposition import create_tensor

class TestCreateTensor(unittest.TestCase):

    def test_allocate_id(self):
        df = pd.DataFrame([['hoge_a', 'foo_a', 'bar_a'],
                           ['hoge_b', 'foo_b', 'bar_a'],
                           ['hoge_b', 'foo_c', 'bar_b']
                          ]
                         )
        df.columns = ['hoge', 'foo', 'bar']

        actual_df = df.copy()
        for col_name in ['hoge', 'foo', 'bar']:
            actual_label, actual_unique = create_tensor.allocate_id(actual_df, col_name)

        df['hoge_id'] = [0, 1, 1]
        df['foo_id'] = [0, 1, 2]
        df['bar_id'] = [0, 0, 1]
                                                                                   
        assert_frame_equal(actual_df, df)


    def test_melting_matrix(self):
        list_x = [np.array([[5. ,  4.],
                            [-2.5, 0.]])
                 ]
        list_expected = [np.array([[0, 0, 5.],
                                   [0, 1, 4.],
                                   [1, 0, -2.5],
                                   [1, 1, 0.]])
                        ]
        for x, expected in zip(list_x, list_expected):
            actual = create_tensor.melting_matrix(x)
            self.assertEqual(actual.all(), expected.all())


    def test_create_tensor_from_df(self):

        df = pd.DataFrame([['hoge_a', 'foo_a',  0],
                           ['hoge_b', 'foo_b',  1],
                           ['hoge_b', 'foo_c',  2]
                          ]
                         )
        df.columns = ['hoge', 'foo', 'val']

        list_df = [df]
        list_mode = [['hoge', 'foo',]]
        list_shape = [(2, 3)]
        list_value_name = ['val']
              
        for df, mode, shape, value_name in zip(list_df, list_mode, list_shape, list_value_name):
            ## Make an actual tensor ############################
            ct = create_tensor.CreateTensor(df, value_name, mode)
            actual = ct.create_tensor_from_df(missing_val=0)
            #####################################################

            ## Make an expected tansor ##########################
            for col_name in mode:
                actual_label, actual_unique = create_tensor.allocate_id(df, col_name)
            expected = np.array([[0, 0, 0],
                                 [0, 1, 2]
                                ]
                               )

            #####################################################
            self.assertEqual(actual.all(), expected.all())


if __name__== '__main__':
    unittest.main()
