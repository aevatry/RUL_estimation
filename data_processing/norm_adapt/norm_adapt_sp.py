"""
This code attempts to reproduce:
Adaptive Normalization: A novel data normalization approach for non-stationary time series

Conference: International Joint Conference on Neural Networks, IJCNN 2010, Barcelona, Spain, 18-23 July, 2010
Eduardo Ogasawara et Al.

"""

import numpy as np

def EMA(sequence: np.ndarray, ma_win: int):
    """
    Exponential moving average of time series

    Args:
        sequence (np.ndarray): array of dimensions (features, lenght)
        ma_win (int): number of points taken in the moving average window

    Returns:
        Sk (np.ndarray): array of the corresponding moving average values
    """
    EMA_vals = np.empty((sequence.shape[0], sequence.shape[1] - ma_win + 1))
    alpha = 2/(ma_win+1)

    for i in range(0, sequence.shape[1] - ma_win + 1):
        if i == 0:
            EMA_vals[:,0] = (1/ma_win)*np.sum(sequence[:, 0 : 0+ma_win], axis = 1)

        else:
            EMA_vals [:,i] = (1-alpha)*EMA_vals[:,i-1] + alpha*sequence[:, i+ma_win-1]

    return EMA_vals

def SMA(sequence: np.ndarray, ma_win: int):
    """
    Simple moving average of time series

    Args:
        sequence (np.ndarray): array of dimensions (features, lenght)
        ma_win (int): number of points taken in the moving average window

    Returns:
        Sk (np.ndarray): array of the corresponding moving average values
    """
    SMA_vals = np.empty((sequence.shape[0], sequence.shape[1] - ma_win + 1))

    for i in range(0, sequence.shape[1] - ma_win + 1):

        SMA_vals[:,i] = (1/ma_win)*np.sum(sequence[:, i : i+ma_win], axis = 1)

    return SMA_vals


def get_R(sequence: np.ndarray, ma_win:int, high_norma:int = 1, low_norma:int = 0, MA_mode:str = 'EMA', sl_win:int = None):

    """
    Get the Adaptive Normalization of the input sequence 

    Args:
        sequence (np.ndarray): array of dimensions (features, lenght). Lenght need to be the same for all features
        ma_win (int): number of points taken in the moving average window
        high_norma (int, default = 1): higher value for normalisation
        low_norma (int, default = 0): lower value for normalisation
        MA_mode (str, default = 'EMA): Mode of the moving average series. Two implemented: 'EMA' (Exponential moving average) and 'SMA' (Simple Moving Average)
        sl_win (int, optional): number fo points in each Disjoint Sliding Window. Default is put to lenght of the sequence

    Returns:
        R (np.ndarray): Normalized sequence with dimensions (batch, features, lenght) if sl_win = lenght
                        IF sl_win != lenght, we have R (batch, feature, lenght - sl_win +1, sl_win)
    """
    
    if MA_mode == 'SMA':
        MA = SMA(sequence, ma_win)
    elif MA_mode == 'EMA':
        MA = EMA(sequence, ma_win)
    else: 
        raise NotImplementedError(f"The moving average mode needs to be 'EMA' or 'SMA' but got {MA_mode} instead")

    if sl_win == None:
        sl_win = sequence.shape[-1]

    print(sl_win)
    
    R = np.zeros((sequence.shape[0], sequence.shape[1] - sl_win + 1, sl_win))

    for i in range(0, R.shape[1]):
        S = sequence[:,i:i+sl_win]
        Sk = MA[:,i].reshape((R.shape[0],-1))
        R_new = S/Sk
        R[:,i] = R_new

    _outliers_norm(R, high_norma, low_norma)
    return R



def _outliers_norm (R, high_norma, low_norma):
    
    quantiles = np.array([np.quantile(arr, [0.25,0.75], axis = 0) for arr in R.reshape((R.shape[0], -1))])

    IQR = (quantiles[:, 1] - quantiles[:, 0]).reshape((R.shape[0], -1))

    low_lim = quantiles[:,0].reshape((R.shape[0], -1)) - 1.5*IQR
    high_lim = quantiles[:,1].reshape((R.shape[0], -1)) + 1.5*IQR

    # set rows/Disjoint Sliding Window to NONE if any value in row outside of quartiles
    # block below is NOT to be used: because we want really long w sections in the R series, really high probability that one of the vals will be off: voids to whole line. And we still need corrects low_lim and high_lim for minmax normalization
    '''
    for i,batch in enumerate(R):
        for j,feature in enumerate(batch):
            for m, DSW in enumerate(feature):

                cond_list = [True for val in DSW if val<lims[i][j][0] or val>lims[i][j][1]]

                if True in cond_list:
                    R[i,j,m].fill(None)
    '''

    _minmax_norm(R, high_lim, low_lim, high_norma, low_norma)

def _minmax_norm(R, high_lim, low_lim, high_norma, low_norma):

    min_R = np.min( R.reshape((R.shape[0], -1)), axis=1 ).reshape((R.shape[0], -1))
    max_R = np.max( R.reshape((R.shape[0], -1)), axis=1 ).reshape((R.shape[0], -1))

    min_a = np.max(np.concatenate((min_R, low_lim), axis = 1), axis = 1).reshape(R.shape[0], -1)
    max_a = np.min(np.concatenate((max_R, high_lim), axis = 1), axis = 1).reshape(R.shape[0], -1)

    for i in range(0,R.shape[-2]):
        for j in range(0, R.shape[-1]):

            holder = (high_norma - low_norma)* (R[:, i,j].reshape((R.shape[0], -1)) - min_a)/(max_a - min_a) + low_norma
            R[:, i, j] = holder.reshape((R.shape[0]))

    

def lvl_adj_R(MA, R):
    adj = 0

    for DSW in R[:,:,]:
        pass
    return adj

def levl_adj_DSW(MA, DSW):
    adj_DSW = 0
    return adj_DSW