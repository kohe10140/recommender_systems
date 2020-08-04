import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD
import tensorly as tl
from tensorly.decomposition import tucker

#from recommender_systems.create_tensor import CreateTensor


class TuckerRecommendation(BaseEstimator, RegressorMixin):
    
    def __init__(self, shape=[None, None, None], rank=[None, None, None], missing_val='mean',
                 n_iter_max=100, tol=0.0001, random_state=0, verbose=False):
        self.shape = shape
        self.rank = rank
        self.missing_val = missing_val   
        self.n_iter_max = n_iter_max
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose


    def fit(self, X, y):
        X = X.astype(int)
        self.tensor = np.full(self.shape, np.nan)
        self.tensor[tuple(X.T)] = y

        if self.missing_val == 'mean':
            self.tensor[np.isnan(self.tensor)] = np.nanmean(self.tensor)
        else:
            self.tensor[np.isnan(self.tensor)] = self.missing_val

        self.core, self.factors = tucker(self.tensor, rank=self.rank, n_iter_max=self.n_iter_max,
                                         tol=self.tol, random_state=self.random_state,
                                         verbose=self.verbose)

        # TODO: Confirm tensorly version '0.4.5'
        self.tucker_tensor = tl.tucker_to_tensor(self.core, self.factors)
        return self


    def predict(self, test_X):
        tucker_tensor = self.tucker_tensor
        return tucker_tensor[tuple(test_X.T)]
    
            
    def score(self, test_X, test_y):
        test_X = test_X.astype(int)
        prediction = self.predict(test_X)
        mse = mean_squared_error(test_y, prediction)
        return -np.sqrt(mse)


class SymTucker(TuckerRecommendation):

    def __init__(self, shape=[None, None, None], rank=[None, None, None], missing_val='mean',
                 sym_df=None, n_iter_max=100, tol=0.0001, random_state=0, verbose=False):
        """
        sym_df : pandas.DataFrame
            A symmetoric dataframe of the exploration space.
            ex.) obj_A is symmetric with obj_B
             idx|obj_A|obj_B|obj_C|
            ----|-----|-----|-----|
               0|    0|    1|    1| ---
               1|    1|    2|    1|    |- P
               2|    2|    3|    0| ---   
               0|    1|    0|    1| ---   
               1|    2|    1|    1|    |- Q
               2|    3|    2|    0| ---

            -> P is symmetric with Q in the upper table.
            index must be indicate the same data.
            The size of sym_df must be equal to the size of X.
        """
        super().__init__(shape=shape, rank=rank, missing_val=missing_val,
                         n_iter_max=n_iter_max, tol=tol,  random_state=random_state, verbose=verbose)
        self.sym_df = sym_df


    def fit(self, X, y):
        """
        X : ndarray of shape(n_samples)
            The index(es) of training X.
        y : ndarray of shape(n_samples)
            The array of training y(objective variables).
        """
        X = self.sym_df.loc[X].values
        y = np.hstack([y, y])
        super().fit(X, y)


    def predict(self, test_X):
        """
        X : ndarray of shape(n_samples)
            The index(es) of training X.
        """
        df_test_X = self.sym_df.loc[test_X]
        tucker_tensor = self.tucker_tensor
        pred_y = tucker_tensor[tuple(df_test_X.values.T)]
        pred_y = pd.DataFrame(pred_y, index=df_test_X.index)
        # Aggregate symmetory objects 
        pred_y = pred_y.groupby(pred_y.index).mean().values.ravel()
        return pred_y


    def score(self, test_X, test_y):
        test_X = test_X.astype(int)
        prediction = self.predict(test_X)
        mse = mean_squared_error(test_y, prediction)
        return -np.sqrt(mse)


class TuckerClassifier(BaseEstimator, ClassifierMixin):
    
    def __init__(self, shape=[None, None, None], rank=[None, None, None], missing_val='mean',
                 n_iter_max=100, tol=0.0001, random_state=0, verbose=False):
        self.shape = shape
        self.rank = rank
        self.missing_val = missing_val   
        self.n_iter_max = n_iter_max
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose


    def fit(self, X, y):
        X = X.astype(int)
        tensor = np.full(self.shape, np.nan)
        tensor[tuple(X.T)] = y

        if self.missing_val == 'mean':
            tensor[np.isnan(tensor)] = np.nanmean(tensor)
        else:
            tensor[np.isnan(tensor)] = self.missing_val

        self.core, self.factors = tucker(tensor, rank=self.rank, n_iter_max=self.n_iter_max,
                                         tol=self.tol, random_state=self.random_state,
                                         verbose=self.verbose)

        self.tucker_tensor = tl.tucker_to_tensor(self.core, self.factors)
        return self


    def predict(self, test_X):
        tucker_tensor = self.tucker_tensor
        return tucker_tensor[tuple(test_X.T)]
    
            
    def score(self, test_X, test_y):
        test_X = test_X.astype(int)
        prediction = self.predict(test_X)
        mse = mean_squared_error(test_y, prediction)
        return -np.sqrt(mse)


class SVD(BaseEstimator, RegressorMixin):

    def __init__(self, shape, rank=None, missing_val='mean'):
        self.shape = shape
        self.rank = rank
        self.missing_val = missing_val


    def fit(self, X, y):
        self.R = np.full(self.shape, np.nan)
        X = X.astype(int)
        self.R[list(X.T)] = y

        if self.missing_val == 'mean':
            self.R[np.isnan(self.R)] = np.nanmean(self.R)
        else:
            self.R[np.isnan(self.R)] = self.missing_val

        t_svd = TruncatedSVD(n_components=self.rank, random_state=0)
        self.U = t_svd.fit_transform(self.R)
        self.Vh = t_svd.components_
        self.singular_values_ = t_svd.singular_values_

        self.R_tilda = self.U @ self.Vh
        return self


    def predict(self, test_X):
        R_tilda = self.R_tilda
        test_X = test_X.astype(int)
        return R_tilda[list(test_X.T)]


    def score(self, test_X, test_y):
        prediction = self.predict(test_X)
        mse = mean_squared_error(test_y, prediction)
        return -np.sqrt(mse)