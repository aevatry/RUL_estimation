from torch import nn
import numpy as np
import torch

class CNN1D_VNL(nn.Module):

    def __init__(self, **kwargs):
        """
        Arguments:
            kwargs (unpacked dict, optional): dict to set the options, where the keys and explanation of options are:
                _pyramid_bins : list (default = [200]) -> bins to be used for pyramid pooling operation
                _pooling_mode : str (default = 'max') -> Choose between implemented pooling modes 
        """

        super().__init__()
        try:
            self._pyramid_bins = kwargs['_pyramid_bins']
            print(f"pyramid pooling bins set to custom value: {self._pyramid_bins}")
        except:
            self._pyramid_bins = [200]
            print(f"pyramid pooling bins set to default: {self._pyramid_bins}")

        try:
            self._pooling_mode = kwargs['_pooling_mode']
            print(f"pooling mode set to custom value: {self._pooling_mode}")
        except:
            self._pooling_mode = 'max'
            print(f"pooling mode set to default: {self._pooling_mode}")

 

        # Now, need to add layers, see : https://pytorch.org/docs/stable/nn.html
        self.conv1_1D_depthwise = nn.Conv1d(in_channels= 4, out_channels= 4*16, kernel_size= 30, stride= 1, groups= 4) #depthwise
        self.conv1_1D_pairwise = nn.Conv1d(in_channels= 4*16, out_channels= 32, kernel_size= 1, stride= 1) #pairwise
        #Pooling layers are inside of the variable maxpool functions
        self.conv2_1D = nn.Conv1d(in_channels = 32, out_channels= 64, kernel_size= 10, stride = 1)


        # Number of input features: number of channels outputs of previous layer * sum of all bin sizes
        self.Linear1 = nn.Linear(in_features=64*self.get_total_bins(), out_features=250)
        self.Linear2 = nn.Linear(in_features=250, out_features=1)

        # Instantiation of pooling classes
        self.pyramid_pool = Spatial_pyramid_pooling(self._pyramid_bins, self._pooling_mode)

        self.LeakyReLu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh() 

    def forward (self, X): #This function is the one called to do a forward pass
        
        # First Separable 1D Convolution
        X = self.conv1_1D_depthwise(X)
        X = self.conv1_1D_pairwise(X)
        X = self.LeakyReLu(X)

        # Variational Pooling
        X = var_pool_func(previous_conv=X, _mode = self._pooling_mode)

        # Second 1D Convolution
        X = self.conv2_1D(X)
        X = self.LeakyReLu(X)

        # Spatial Pyramidal Pooling
        X = self.pyramid_pool(X) 
        X = torch.flatten(X, start_dim=1)
        # Fully Connected Layers
        X = self.Linear1(X)
        X = self.tanh(X)
        
        X = self.Linear2(X)
        X = self.sigmoid(X) #This is the estimated RUL
        return X
    
    def get_total_bins(self):
        
        list_to_array = np.array(self._pyramid_bins)
        assert len(np.array(list_to_array).shape) == 1, 'Bins should be 1 dimensional'
        return np.sum(list_to_array, axis = 0)
    


class Spatial_pyramid_pooling (nn.Module):

    """
    Spatial pyramidal pool class implemented with mps compatibility (with reduced functionality like no dynamic kernel size and stride). 
    Structure used : https://github.com/addisonklinke/pytorch-architectures/blob/master/torcharch/conv.py

    Args:
        pyramid_bins (list): list of all output size for eacg convolution
        _mode (str): convolution mode. implemented are 'max' for maxpool and 'avg' for average pool

    When forward method is called, accepts a tensor X of shape (batch, features, input_lenght)
    Returns a tensor of shape (batch, features, sum(bins)) where each bin pooling operation is concatenated in the 2nd dimension
    """

    # not need to pass the previous convolution in the __init__ because it is pass in the forward pass
    def __init__(self, pyramid_bins:list , _mode:str):
        super().__init__()

        self.name = 'Spatial_pyramid_pooling' 

        self.pyramid_bins = pyramid_bins
        self._mode = _mode

        # How to get a module list in Pytorch
        #nn.ModuleList allows PyTorch to find the convolution layers
        #self.pools = nn.ModuleList([])

        #for size in pyramid_bins:
        #    self.pools.append(pool_func(int(size)))

    
    def forward(self, X):
        
        # X is the result of the previous convolution
        assert X.dim() == 3, 'Expect a 3D input: (Batch, Channels, Lenght)'


        pooled = []
        for bin_size in self.pyramid_bins:
            pooled.append(self.get_adaptive_pool(X, bin_size))

        return torch.cat(pooled, dim=2)
    
    def get_adaptive_pool (self, previous_conv:torch.Tensor, output_size:int)-> torch.Tensor:

        if self._mode == 'max':
            pool_func = torch.nn.functional.max_pool1d
        elif self._mode == 'avg':
            pool_func = torch.nn.functional.avg_pool1d
        else:
            raise NotImplementedError(f"Unknown pooling mode '{self._mode}', expected 'max' or 'avg'")

        stride = (previous_conv.shape[2]//output_size) # floor division
        kernel_size = previous_conv.shape[2] - (output_size-1)*stride

        return pool_func(previous_conv, kernel_size, stride)
    


def var_pool_func(previous_conv: torch.Tensor, kernel_size: int=20, stride: int=5, _mode = 'max') -> torch.Tensor:

    '''
    Performs a pooling, whose mode is decided by the _mode variable, using the functional API of PyTorch

    Args:
    
        previous_conv (torch.Tensor): Tensor of the output of the previous convolution layer
        kernel_size (int): desired kernel size
        stride (int): desired stride
        _mode (str): accepts 'max' for max pooling or 'avg' for average pooling
    
    Returns: 
        Output of torch.nn.functional.max_pool1d or torch.nn.functional.avg_pool1d depending on mode with ceil=true for the specified parameters of the inputs

    '''  

    # Get the right pooling function
    if _mode == 'max':
        pool_func = torch.nn.functional.max_pool1d
    elif _mode == 'avg':
        pool_func = torch.nn.functional.avg_pool1d
    else:
        raise NotImplementedError(f"Unknown pooling mode '{_mode}', expected 'max' or 'avg'")


    return pool_func(previous_conv, kernel_size=kernel_size, stride=stride, ceil_mode=True)