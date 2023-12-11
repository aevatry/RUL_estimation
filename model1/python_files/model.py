import torch
from torch import nn
import numpy as np

from spatial_pyramidal_pooling import spatial_pyramid_pool
from variable_maxpool_layer import var_maxpool
from RUL_loss_function import RUL_loss

# We want to use the GPU if it is available
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

class CNN1D_RUL(nn.Module):

    def __innit__(self):
        super().__innit__()
        # Now, need to add layers, see : https://pytorch.org/docs/stable/nn.html
        self.conv1_1D_depthwise = nn.Conv1d(in_channels= 4, out_channels= 4*16, stride=1, kernel_size= 200, groups=4) #depthwise
        self.conv1_1D_pairwise = nn.Conv1d(in_channels= 4*16, out_channels= 32, kernel_size=10, stride =1) #pairwise
        #Pooling layers are inside of the variable maxpool functions

    def forward(self, data): #This function is the one called to do a forward pass
        
        return data