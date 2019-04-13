import unittest

import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal
from tensorly.decomposition import tucker

import recommender_systems.create_tensor


class TestCreateTensor(unittest.TestCase):

    def test_allocate_id(self):
        df = pd.read_csv('/home/nishi/b4/data/working/test_df.csv', index_col=0)
        actual_df = df.copy()
        for col_name in ['atom_A', 'atom_B', 'prototype']:
            actual_label, actual_unique = create_tensor.allocate_id(actual_df, col_name)

        df['atom_A_id'] = [0, 1, 2, 1, 3]
        df['atom_B_id'] = [0, 1, 2, 3, 4]
        df['prototype_id'] = [0, 1, 2, 3, 4]
        
        assert_frame_equal(actual_df, df)

        
#    def test_create_tensor_from_df(self):
#        df = pd.read_csv('/home/nishi/b4/data/working/test_df.csv', index_col=0)
#        list_df = [df]
#        list_mode = [['atom_A', 'atom_B', 'prototype']]
#        list_shape = [(4, 5, 5)]
#        list_value_name = ['form_ene_per_atom']
#        for df, mode, shape, value_name in zip(list_df, list_mode, list_shape, list_value_name):
#
#            for col_name in mode:
#                actual_label, actual_unique = create_tensor.allocate_id(df, col_name)
#
#            actual = create_tensor.transform_df_to_tensor(df, mode, shape, value_name)
#            expected = np.zeros(shape)
#            mode_ids = [col_name + '_id' for col_name in mode]
#            indexes = df[mode_ids]
#            values = df[value_name]
#            expected[list(indexes.values.T)] = values.values
#            self.assertEqual(actual.all(), expected.all())


    def test_create_tensor_from_df(self):
        df = pd.read_csv('/home/nishi/b4/data/working/test_df.csv', index_col=0)
        list_df = [df]
        list_mode = [['atom_A', 'atom_B', 'prototype']]
        list_shape = [(4, 5, 5)]
        list_value_name = ['form_ene_per_atom']

        for df, mode, shape, value_name in zip(list_df, list_mode, list_shape, list_value_name):

            ## Make an actual tensor ############################
            ct = create_tensor.CreateTensor(df, value_name, mode)
            actual = ct.create_tensor_from_df(df)
            #####################################################
  
            ## Make an expected tansor ##########################
            for col_name in mode:
                actual_label, actual_unique = create_tensor.allocate_id(df, col_name)
            mode_ids = [col_name + '_id' for col_name in mode]
            indexes = df[mode_ids]
            values = df[value_name]
            expected = df[mode_ids + [value_name]]
            #####################################################

            assert_frame_equal(actual, expected)


    def test_create_tensor_from_df_for_trainset(self):
        df = pd.read_csv('/home/nishi/b4/data/working/test_df.csv', index_col=0)
        list_df = [df]
        list_mode = [['atom_A', 'atom_B', 'prototype']]
        list_shape = [(4, 5, 5)]
        list_value_name = ['form_ene_per_atom']

        for df, mode, shape, value_name in zip(list_df, list_mode, list_shape, list_value_name):

            ## Make an actual tensor ############################
            ct = create_tensor.CreateTensor(df, value_name, mode)
            actual = ct.create_tensor_from_df(df[df['atom_A'] == 'W'])
            #####################################################
  
            ## Make an expected tansor ##########################
            for col_name in mode:
                actual_label, actual_unique = create_tensor.allocate_id(df, col_name)
            expected = np.zeros(shape)
            mode_ids = [col_name + '_id' for col_name in mode]
            df_trainset = df[df['atom_A'] == 'W']
            indexes = df_trainset[mode_ids]
            values = df_trainset[value_name]
            expected = df_trainset[mode_ids + [value_name]]
            #####################################################

            assert_frame_equal(actual, expected)


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


if __name__ == '__main__':
    unittest.main()
