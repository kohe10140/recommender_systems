import numpy as np
import pandas as pd
import tensorly as tl


def allocate_id(df, col_name):
    label, unique = pd.factorize(df[col_name])
    col_name_id = col_name + '_id'
    df[col_name_id] = label
    return label, unique


def melting_matrix(x):
    melted = np.array([[int(i), int(j), x[i][j]] for i in range(x.shape[0]) for j in range(x.shape[1])])
    return melted


class CreateTensor:
    
    def __init__(self, df, value_name, mode=[None, None, None]):
        self.df = df
        self.value_name = value_name
        self.mode = mode
        ## Allocate id to each mode ######################
        self.labels = [allocate_id(self.df, col_name)[0] for col_name in mode]
        self.uniques = [allocate_id(self.df, col_name)[1] for col_name in mode]
        ##################################################
        self.shape = tuple([len(unique) for unique in self.uniques])
        self.tensor = np.full(self.shape, np.nan)


    def create_tensor_from_df(self, missing_val='mean'):
        """
        attributes
            self.tensor : numpy array
                tensor to be decomposed 

            self.dataset : pandas dataframe
                trimed dataset that has columns of mode and entries of the tensor

            self.indexes : pandas dataframe
                labels of the tensor's mode

            self.values : pandas series
                entries of the tensor
        """
        self.missing_val = missing_val
        self.mode_ids = [col_name + '_id' for col_name in self.mode]

        self.dataset = self.df[self.mode_ids + [self.value_name]]
        self.indexes = self.df[self.mode_ids]
        self.values = self.df[self.value_name]
        
        self.tensor[list(self.indexes.values.T)] = self.values.values

        if self.missing_val == 'mean':
            self.tensor[np.isnan(self.tensor)] = np.nanmean(self.tensor)
        else:
            self.tensor[np.isnan(self.tensor)] = self.missing_val

        return self.tensor
            
