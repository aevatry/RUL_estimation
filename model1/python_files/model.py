from torch import nn
import numpy as np

from spatial_pyramidal_pooling import spatial_pyramid_pool
from variable_maxpool_layer import var_maxpool

# We want to use the GPU if it is available
#device = (
#    "cuda"
#    if torch.cuda.is_available()
#    else "cpu"
#)

class CNN1D_RUL(nn.Module):

    def __innit__(self, pyramid_pool_bins:tuple, device):
        super().__innit__()

        self.pyramid_pool_bins = pyramid_pool_bins
        self.device = device
        # Now, need to add layers, see : https://pytorch.org/docs/stable/nn.html
        self.conv1_1D_depthwise = nn.Conv1d(in_channels= 4, out_channels= 4*16, kernel_size= 30, stride= 1, groups= 4) #depthwise
        self.conv1_1D_pairwise = nn.Conv1d(in_channels= 4*16, out_channels= 32, kernel_size= 1, stride= 1) #pairwise
        #Pooling layers are inside of the variable maxpool functions
        self.conv2_1D = nn.Conv1d(in_channels = 32, out_channels= 64, kernel_size= 10, stride = 1)


        # Number of input features: number of channels outputs of previous layer * sum of all bin sizes
        self.Linear1 = nn.Linear(in_features=64*self.get_total_bins(), out_features=250, device=device)
        self.Linear2 = nn.Linear(in_features=250, out_features=1, device=device)

        self.LeakyReLu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh() 

    def forward(self, X): #This function is the one called to do a forward pass
        
        # First Separable 1D Convolution
        X = self.conv1_1D_depthwise(X)
        X = self.LeakyReLu(X)
        X = self.conv1_1D_pairwise(X)
        X = self.LeakyReLu(X)

        # Variational Pooling
        X = var_maxpool(previous_conv=X, kernel_size=2) 

        # Second 1D Convolution
        X = self.conv2_1D(X)
        X = self.LeakyReLu(X)

        # Spatial Pyramidal Pooling
        X = spatial_pyramid_pool(previous_conv= X, bin_sizes=self.pyramid_pool_bins) 

        # Fully Connected Layers
        X = self.Linear1(X)
        X = self.tanh(X)
        
        X = self.Linear2(X)
        X = self.sigmoid(X) #This is the estimated RUL
        return X
    
    def get_total_bins(self):

        tuple_to_array = np.array(self.pyramid_pool_bins)
        return np.sum(tuple_to_array, axis = 0)