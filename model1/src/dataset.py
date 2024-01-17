from torch.utils.data import Dataset
import torch
import numpy as np
import os

class RUL_Dataset(Dataset):
    """Face Landmarks dataset. Example from PyTorch: see https://pytorch.org/tutorials/beginner/data_loading_tutorial.html"""

    def __init__(self, train_dir, permutations, max_starting = 1e4, min_lenght = 3600, transform=None):
        """
        Arguments:
            train_dir (string): path to a directory with all csv files with n time series of features with labels (labels are csv last series) .
            permutations (integer): number of truncated time series we want to extract for 1 epoch
            max_starting (int): The maximum value for the starting point of the series
            min_lenght (int): The minimum lenght for all series
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform
        self.permutations = permutations
        self.min_lenght = min_lenght
        self.max_starting = max_starting

        # Checking for different types 
        signs = [' ', ',', ';','    ']
        series_t = []
        for file_path in os.listdir(train_dir):

            full_path = '/'.join([train_dir, file_path])

            for sign in signs:
                try:
                    series_t += [torch.Tensor(np.loadtxt(full_path, delimiter=sign))]
                    print(f"Sign : '{sign}' works")
                except : 
                    print(f"Sign : '{sign}' does not work")

        self.series_t = series_t

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

        existence, start_points = self._selfbatch(self.all_series_len, sequence_lenght)
        

        label = torch.as_tensor(np.array([[self.series_t[i][-1, start_points[i]+sequence_lenght ]] for i in existence]))
        # List of all series features
        features = torch.as_tensor(np.array([self.series_t[i][:-1, start_points[i] : start_points[i]+sequence_lenght]  for i in existence]))

        return features, label
    
    
    def _selfbatch(self, all_series_len, seq_len):

        # decides which series are included in thos batch
        existence = [i for i in range(len(all_series_len)) if seq_len<all_series_len[i]]

        # randomly select start_points that guarantee existence
        start_points = np.random.randint(low=0, high=[max(all_series_len[i] - seq_len, 1) for i in range(len(all_series_len))])

        return existence, start_points