import numpy as np
import pandas as pd
import tensorly as tl


def allocate_id(df, col_name, mode_member=None):
    col_name_id = col_name + '_id'
    if mode_member is not None:
        mode_member = mode_member.astype(object)
        unique = mode_member
        label = np.array([np.where(data==mode_member)[0][0] for data in df[col_name]]) 
        df[col_name_id] = list(label)
    else:
        label, unique = pd.factorize(df[col_name])
        df[col_name_id] = list(label)
    df[col_name_id] = df[col_name_id].astype(np.int64)
    return label, unique


def melting_matrix(x):
    melted = np.array([[int(i), int(j), x[i][j]] for i in range(x.shape[0]) for j in range(x.shape[1])])
    return melted


class CreateTensor:
    """
    To create the tensor to decomposite

    Attributes
    ----------
        value_name : str
            the name of data to be entries of the tensor

        mode : list 
            A column name of the dataframe

        mode_members : list of numpy.array
            The entry's label of each mode
        
        tensor : numpy.array
            tensor to be decomposed 

        dataset : pandas.dataframe
            trimed dataset that has columns of mode and entries of the tensor

        indexes : pandas.dataframe
            labels of the tensor's mode

        values : pandas.series
            entries of the tensor

    Examples
    --------
    >>> mode = ['dog', 'cat', 'monkey'] 
    >>> dog_breeds = np.array(['shiba', 'husky', 'retriever'])
    ...
    >>> mode_members = [dog_breeds, cat_breeds, monkey_breeds] 
    >>> ct = CreateTensor(df, 'average_weight', mode, mode_member)
    """

    def __init__(self, df, value_name, mode=[None], mode_members=None):
        self.df = df
        self.value_name = value_name
        self.mode = mode
        self.mode_members = mode_members 
        ## Allocate id to each mode ######################
        if mode_members != None:
            self.labels = [allocate_id(self.df, col_name, mode_member)[0] for col_name, mode_member in zip(mode, mode_members)]
            self.uniques = [allocate_id(self.df, col_name, mode_member)[1] for col_name, mode_member in zip(mode, mode_members)]
        else:
            self.labels = [allocate_id(self.df, col_name)[0] for col_name in mode]
            self.uniques = [allocate_id(self.df, col_name)[1] for col_name in mode]
        ##################################################
        self.shape = tuple([len(unique) for unique in self.uniques])
        self.tensor = np.full(self.shape, np.nan)


    def create_tensor_from_df(self, missing_val='mean'):
        self.missing_val = missing_val
        self.mode_ids = [col_name + '_id' for col_name in self.mode]

        self.dataset = self.df[self.mode_ids + [self.value_name]]
        self.indexes = self.df[self.mode_ids]
        self.values = self.df[self.value_name]
        
        self.tensor[tuple(self.indexes.values.T)] = self.values.values

        if self.missing_val == 'mean':
            self.tensor[np.isnan(self.tensor)] = np.nanmean(self.tensor)
        else:
            self.tensor[np.isnan(self.tensor)] = self.missing_val

        return self.tensor
            
