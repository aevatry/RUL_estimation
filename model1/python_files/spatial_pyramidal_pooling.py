import math
import torch
import torch.nn as nn

def spatial_pyramid_pool(previous_conv: torch.Tensor, bin_sizes: tuple, padding_front: bool = True)->torch.Tensor:
    '''
    Function to perform spatial pyramidal pooling on the output of a 1D Convolutional Layer (This function only works for 1D CNN layers). It is based on the 2 following links:
    Initial repository: https://github.com/yifanjiang19/sppnet-pytorch/tree/270529337baa5211538bf553bda222b9140838b3 
    Academic paper: https://ieeexplore.ieee.org/abstract/document/7005506
    
    Changes in adaptation:
    The previous_conv is a 1D Convolution, no longer 2D. 
    Padding is applied asymmetrically: either to the front or the back
    Higher risk of breaking network training without alerting the user for now: If the bin incorrectly set-up, will introduce -inf values in the network.    

    Args:

        previous_conv: torch.Tensor ; Vector of previous convolution layer
        bin_sizes: tuple ; vector the number of bins wanted. 
    
    Returns: 
        A tensor vector with shape [1 x n] is the concentration of multi-level pooling

    Future improvements: 
    Amelioration in checking the bin_sizes
    Checking that max_bin <= floor(sqrt(PreviousConv_lenght)) and hence that there cannot be -infinity points in the network 
    
    '''  

    # 2 next following if statements are checking for right types and making emergency fixes to try to keep the code from breaking
    if not isinstance(bin_sizes, tuple):
        bin_sizes = tuple(bin_sizes)
        print(f"{bin_sizes} was of type '{type(bin_sizes)}' and not 'tuple'. ")
    
    if isinstance(bin_sizes, tuple) & (not isinstance (bin_sizes.__getitem__(0) , int)):
        print(f"First element of {bin_sizes} was of type '{type(bin_sizes.__getitem__(0))}' and not 'int'. ")

        # Highly likely that in that case it is floats or lists. 
        # If floats: code don't break but training and forward pass could be wrong
        # If lists: code will break but we want it to break (would be weird to have a tuple of 1 element lists) 
        new_tuple =[]
        for i in range(0, len(bin_sizes)):
            new_tuple += int(bin_sizes.__getitem__(i))
        bin_sizes = tuple(new_tuple)


    # Get some important constants
    conv_batches = previous_conv.shape[0]
    conv_channels = previous_conv.shape[1]
    conv_lenght = previous_conv.shape[2]

    # Actual Spatial Pyramidal Pooling operation
    for i in range(len(bin_sizes)):
        
        # Maxpooling Class initialization
        kernel_size = math.ceil(conv_lenght/bin_sizes.__getitem__(i))
        maxpool = nn.MaxPool1d(kernel_size=kernel_size, stride=kernel_size)

        # Find necessary amount of padding
        padding_zeros = bin_sizes.__getitem__(i)*kernel_size - conv_lenght

        # Performing the adjusted maxpool
        if padding_zeros != 0: # Need for padding

            padding_shape = (conv_batches, conv_channels, padding_zeros)

            # -inf padding of sequence
            padding = torch.full(size=padding_shape, fill_value=float("-inf"), dtype=previous_conv.dtype, device = previous_conv.device, requires_grad= previous_conv.requires_grad)

            if padding_front:
                adjusted_conv = torch.concat((previous_conv, padding), dim = 2)

            if not padding_front:
                adjusted_conv = torch.concat((padding, previous_conv), dim = 2)
            
            x = maxpool(adjusted_conv)

        if padding_zeros == 0: # No need for padding
        
            x = maxpool(previous_conv)

        # Concatenating tensors to have a 1 channel, 1D tensor as the output
        if(i == 0):
            spp = x.view(conv_batches,-1)
        else:
            spp = torch.cat((spp,x.view(conv_batches,-1)), 1)

    return spp
    


