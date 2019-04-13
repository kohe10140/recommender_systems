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
        self.labels = []
        self.uniques = []
        for col_name in mode:
            label, unique = allocate_id(df, col_name)
            self.labels.append(label)
            self.uniques.append(unique)
        ##################################################
        self.shape = tuple([len(unique) for unique in self.uniques])
        self.tensor = np.full(self.shape, np.nan)


    def create_tensor_from_df(self, df, how_fillna='column_mean'):
        self.mode_ids = [col_name + '_id' for col_name in self.mode]

        self.dataset = df[self.mode_ids + [self.value_name]]
        self.indexes = df[self.mode_ids]
        self.values = df[self.value_name]
        
        self.tensor[list(self.indexes.values.T)] = self.values.values

        if how_fillna == 'column_mean':
            dataframe = pd.DataFrame(self.tensor)
            dataframe = dataframe.fillna(dataframe.mean())
            self.tensor = dataframe.values            

        elif how_fillna == 'row_mean':
            dataframe = pd.DataFrame(self.tensor.T)
            dataframe = dataframe.fillna(dataframe.mean())
            self.tensor = dataframe.values.T            

        elif how_fillna == 'global_mean':
            self.tensor[np.isnan(self.tensor)] = np.nanmean(self.tensor)

        else:
            raise Exception('invalid method')
            # ToDo : preprocessing default values for tensor

        self.dataset = melting_matrix(self.tensor)

        return self.dataset
            
