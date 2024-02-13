"""
This code attempts to reproduce:
Adaptive Normalization: A novel data normalization approach for non-stationary time series

Conference: International Joint Conference on Neural Networks, IJCNN 2010, Barcelona, Spain, 18-23 July, 2010
Eduardo Ogasawara et Al.

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import os


class NORM_ADAPT(Dataset):
    """Face Landmarks dataset. Example from PyTorch: see https://pytorch.org/tutorials/beginner/data_loading_tutorial.html"""

    def __init__(self, train_dir:str, **kwargs)-> torch.Tensor: 

        """
        Arguments:
            train_dir (string): path to a directory with all csv files with n time series of features with labels (labels are csv last series)
            kwargs (unpacked dict, optional): dict to set the options, where the keys and explanation of options are:
                permutations : int (default = 200) -> number of truncated time series we want to extract for 1 epoch
                min_lenght : int (default = 5000) -> The minimum lenght for all series
                label_start : float (default = 0.0) -> From 0 to 1, the proportion of the series that has a label of full RUL
        """
        super().__init__()

        try:
            self.permutations = kwargs['permutations']
            print(f"permutations set to custom value: {self.permutations}")
        except:
            self.permutations = 200
            print(f"permutations set to default: {self.permutations}")

        try:
            self.min_lenght = kwargs['min_len']
            print(f"min lenght set to custom value: {self.min_lenght}")
        except:
            self.min_lenght = 5000
            print(f"min lenght set to default: {self.min_lenght}")
        
        try:
            label_start = kwargs['label_start']
            print(f"label_start set to custom value: {label_start}")
            if label_start > 1 or label_start<0:
                label_start=0
                print(f"Label starting point was out of range: {label_start}, so it's been set at 0")
        except:
            label_start = 0
            print(f"label_start set to default: {label_start}")



        # Checking for different types of text based files
        signs = [' ', ',', ';','    ']
        series_t = []
        labels = []
        for file_path in os.listdir(train_dir):

            full_path = '/'.join([train_dir, file_path])

            for sign in signs:
                try:
                    series_t += [np.loadtxt(full_path, delimiter=sign)]
                    labels += [rul_labels(series_t[-1].shape[1], label_start)]
                    print(f"Sign : '{sign}' works")
                except : 
                    print(f"Sign : '{sign}' does not work")

        self.series_t = series_t
        self.labels = labels

        all_series_len = [series_t[i].shape[1] for i in range(len(series_t))]
        self.all_series_len = all_series_len

    def __len__(self):
        return self.permutations

    def __getitem__(self, idx):
        
        # Draw the random sequence lenght
        a = 0.5
        z = np.random.uniform(size = 1)
        z = a*z[0]**3 - a*1.5*z[0]**2 + (1+0.5*a)*z[0]


        sequence_lenght = int(self.min_lenght + z*(max(self.all_series_len) - self.min_lenght))

        existence = self._selfbatch(self.all_series_len, sequence_lenght)
        

        label = torch.as_tensor(np.array([[self.labels[i][sequence_lenght]] for i in existence]), dtype=torch.float32)
        # List of all series features

        features = np.array([get_R(self.series_t[i][:, 0 : sequence_lenght], ma_win=300)  for i in existence])

        features = torch.as_tensor( features.squeeze(2) , dtype= torch.float32)

        return features, label
    
    
    def _selfbatch(self, all_series_len, seq_len):

        # decides which series are included in thos batch
        existence = [i for i in range(len(all_series_len)) if seq_len<all_series_len[i]]

        return existence
    



class NORM_ADAPT_EVAL(Dataset):
    
    """
    Arguments:
        train_dir (string): path to a directory with all csv files with n time series of features with labels (labels are csv last series)
        step (int): evaluate all indexes that are a multiple of steps
        file_num (int, default = 1): useful if there are multiple files in the eval directory
        kwargs (unpacked dict, optional): dict to set the options, where the keys and explanation of options are:
            permutations : int (default = 200) -> number of truncated time series we want to extract for 1 epoch
            min_lenght : int (default = 5000) -> The minimum lenght for all series
            label_start : float (default = 0.0) -> From 0 to 1, the proportion of the series that has a label of full RUL
    """

    def __init__(self, eval_dir:str, step:int = 10, file_num:int =1, **kwargs)-> torch.Tensor: 
        super().__init__()

        try:
            self.min_lenght = kwargs['min_len']
            print(f"min lenght set to custom value: {self.min_lenght}")
        except:
            self.min_lenght = 5000
            print(f"min lenght set to default: {self.min_lenght}")
        
        try:
            label_start = kwargs['label_start']
            print(f"label_start set to custom value: {label_start}")
            if label_start > 1 or label_start<0:
                label_start=0
                print(f"Label starting point was out of range: {label_start}, so it's been set at 0")
        except:
            label_start = 0
            print(f"label_start set to default: {label_start}")


        self.step = step

        # Checking for different types of text based files
        signs = [' ', ',', ';','    ']
        series_t = []
        labels = []
        for file_path in os.listdir(eval_dir):

            full_path = '/'.join([eval_dir, file_path])

            for sign in signs:
                try:
                    series_t += [np.loadtxt(full_path, delimiter=sign)]
                    labels += [rul_labels(series_t[-1].shape[1], label_start)]
                    print(f"Sign : '{sign}' works")
                except : 
                    print(f"Sign : '{sign}' does not work")

        self.series_t = series_t[file_num-1]
        self.label = labels[file_num-1]

    def __len__(self):
        return int( (self.series_t.shape[1] - self.min_lenght) / self.step)

    def __getitem__(self, idx):
        

        sequence_lenght = int(self.min_lenght + self.step*idx)
        
        label = torch.as_tensor(self.label[sequence_lenght], dtype=torch.float32)
        # List of all series features

        features = np.array([get_R(self.series_t[:, 0 : sequence_lenght], ma_win=300)])
        features = torch.as_tensor( features.squeeze(2) , dtype= torch.float32)

        return features, label
    


# Helper functions


def rul_labels(len_series, ignr_prctg):

    a = -1 /(len_series*(1-ignr_prctg))
    b = 1/(1-ignr_prctg)

    labels = a * np.arange(start=0, stop=len_series, step=1) + b

    labels =np.minimum(labels, 1)

    return labels


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