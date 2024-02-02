"""
This code attempts to reproduce:
Adaptive Normalization: A novel data normalization approach for non-stationary time series

Conference: International Joint Conference on Neural Networks, IJCNN 2010, Barcelona, Spain, 18-23 July, 2010
Eduardo Ogasawara et Al.

"""

import numpy as np
import matplotlib.pyplot as plt

def EMA(sequence, ma_win):
    """
    Exponential moving average of time series

    Args:
        sequence (np.ndarray): array of dimensions (batch, features, lenght)
    """
    EMA_vals = np.empty((sequence.shape[0], sequence.shape[1], sequence.shape[2] - ma_win + 1))
    alpha = 2/(ma_win+1)

    for i in range(0, sequence.shape[2] - ma_win + 1):
        if i == 0:
            EMA_vals[:,:,0] = (1/ma_win)*np.sum(sequence[:, :, i : i+ma_win], axis = 2)

        else:
            EMA_vals [:,:,i] = (1-alpha)*EMA_vals[:,:,i-1] + alpha*sequence[:, :, i+ma_win-1]

    return EMA_vals

def SMA(sequence, ma_win):
    """
    Simple moving average of time series
    """
    SMA_vals = np.empty((sequence.shape[0], sequence.shape[1], sequence.shape[2] - ma_win + 1))

    for i in range(0, sequence.shape[2] - ma_win + 1):

        SMA_vals[:,:,i] = (1/ma_win)*np.sum(sequence[:, :, i : i+ma_win], axis = 2)

    return SMA_vals

def get_R(sequence, MA, sl_win):
    
    R = np.zeros((sequence.shape[0], sequence.shape[1], sequence.shape[2] - sl_win + 1, sl_win))

    for i in range(0, sequence.shape[2] - sl_win+1):

        S = sequence[:,:,i:i+sl_win]
        Si = MA[:,:,i].reshape((sequence.shape[0], sequence.shape[1], -1))

        R_new = S/Si
        R[:,:,i] = R_new

    _rmv_outliers(R, sequence)

    return R

def lvl_adj_R(MA, R):
    adj = 0
    return adj

def levl_adj_DSW(MA, DSW):
    adj_DSW = 0
    return adj_DSW

def _rmv_outliers (R, sequence):
    
    outliers = np.quantile(R.reshape(sequence.shape[0], sequence.shape[1], -1), [0.25,0.75], axis = 2).T

    IQR = (outliers[:, :, 1] - outliers[:, :, 0]).reshape((outliers.shape[0], outliers.shape[1], -1))

    low_lim = outliers[:,:,0].reshape((sequence.shape[0], sequence.shape[1], -1)) - 3*IQR
    high_lim = outliers[:,:,1].reshape((sequence.shape[0], sequence.shape[1], -1)) + 3*IQR

    lims = np.concatenate([low_lim, high_lim], axis = 2)

    # set rows/Disjoint Sliding Window to NONE if any value in row outside of quartiles
    #for i,batch in enumerate(R):
    #    for j,feature in enumerate(batch):
    #        for m, DSW in enumerate(feature):
#
    #            cond_list = [True for val in DSW if val<lims[i][j][0] or val>lims[i][j][1]]
#
    #            if True in cond_list:
    #                R[i,j,m].fill(None)

    _minmax_norm(R, high_lim, low_lim)

def _minmax_norm(R, high_lim, low_lim):

    min_R = np.min(R.reshape((R.shape[0], R.shape[1], -1)), axis =2).reshape((R.shape[0], R.shape[1], -1))
    max_R = np.max(R.reshape((R.shape[0], R.shape[1], -1)), axis =2).reshape((R.shape[0], R.shape[1], -1))

    min_a = np.max(np.concatenate((min_R, low_lim), axis = 2), axis = 2).reshape(R.shape[0], R.shape[1], -1)
    max_a = np.min(np.concatenate((max_R, high_lim), axis = 2), axis = 2).reshape(R.shape[0], R.shape[1], -1)

    for i in range(0,R.shape[-2]):
        for j in range(0, R.shape[-1]):
            pldr = (high_lim - low_lim)*(R[:, :, i,j].reshape((R.shape[0], R.shape[1], -1))-min_a)/(max_a - min_a) + low_lim
            R[:,:, i, j] = pldr.reshape(R[:,:,i,j].shape)

    


def main():
    pass

if __name__ == '__main__':

    pass
