import numpy as np
import pandas as pd
import copy
from numba import guvectorize, float64, int64, jit, njit


def norm(df):
    # data = (df.values - df.mean(1).values.reshape(-1, 1)) / df.std(1).replace(0, 1).values.reshape(-1, 1)
    data = df.subtract(df.mean(1), axis=0).divide(df.std(1), axis=0)
    return data

@jit(nopython=True)
def industryNeutral(factor, df_industry):
    adj_factors = np.full_like(factor, np.nan)
    for i in range(factor.shape[0]):
        y = factor[i]
        x = df_industry[i]
        no_nan = ~np.isnan(y)
        beta = y[no_nan].dot(x[no_nan]) / x[no_nan].sum(0)
        beta[np.isnan(beta)] = 0
        adj_factors[i][no_nan] = y[no_nan] - (x[no_nan].dot(beta))
    return adj_factors

def mv_Neutral(factor, mv0):
    mv = copy.deepcopy(mv0)
    # mv[np.isnan(factor) | np.isinf(factor + mv)] = np.nan
    value_thres = 1e12
    # new addition by junting chen
    mv = mv.reindex_like(factor)
    # new addition by junting chen
    mv[np.isnan(factor) | (np.abs(factor + mv)>value_thres)] = np.nan
    factor[np.isnan(mv)] = np.nan
    factor = norm(factor)
    mv = norm(mv)
    temp_mv = mv.fillna(0).values.reshape(list(mv.shape) + [1])
    # 1、保持date不变。在symbol上做点乘 X.T * y
    # 2、保持date不变，计算mv的 X.T*X
    beta = (np.einsum("ik, ikn->in", factor.fillna(0), temp_mv) / np.einsum("ijk, ikn->ijn",
                                                                            temp_mv.transpose(0, 2, 1),
                                                                            temp_mv).reshape(-1, 1))
    return (factor - mv.values * beta).values