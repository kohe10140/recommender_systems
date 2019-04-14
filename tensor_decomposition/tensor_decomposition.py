import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
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
        tensor = np.full(self.shape, np.nan)
        tensor[list(X.T)] = y

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
        return tucker_tensor[list(test_X.T)]
    
            
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

        if missing_val == 'mean':
            self.R[np.isnan(self.R)] = np.nanmean(self.R)
        else:
            self.R[np.isnan(self.R)] = missing_val

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



