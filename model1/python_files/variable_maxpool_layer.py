import torch

def var_maxpool(previous_conv: torch.Tensor, kernel_size: int, padding_front: bool = True) -> torch.Tensor:

    '''
    Perform a variable Maxpool by padding the output of a 1D CNN layer by the right numbers of -infinity points so that the required kernel_size (and stride, with stride = kernel_size) can be applied to the current 1D series. The padding is performed asymmetrically (front or back). The number of padding points <= kernel_size - 1 to avoid any -infinity points in the network

    Args:
    
        previous_conv: a torch.Tensor vector of the output of the previous convolution layer
        kernel_size: int, desired kernel size
        padding_front: Default : bool True. If True, padds the series by the front. If False, return series padded by the back
    
    Returns: 
        A torch.tensor vector with shape (batch_size, channels, lenght/kernel_size + kernel_size - lenght % kernel_size) and is the concentration of multi-level pooling for 1D CNNS. 
        [n % k] is the modulo operator, returning the rest of n/k

    FOr 2D CNNs: 
    zeros_added = previous_conv.shape[3] % kernel_size
    update padding size
    new_input = torch.concat((previous_conv, padding), dim = 3)

    Math:
    If width of previous convolution divisible by the kernel size: just retrun the maxpool of previous convolution
    In all other cases: number of 0s to be added: the integer needed to ceil current convolutional shape to the nearest multiple of kernel size
    '''  

    maxpool = torch.nn.MaxPool1d(kernel_size=kernel_size, stride = kernel_size)
    # Will use this maxpool layer anyway

    if previous_conv.shape[2] % kernel_size != 0:

        padding_shape = (previous_conv.shape[0], previous_conv.shape[1], kernel_size - previous_conv.shape[2] % kernel_size)

        padding = torch.full(size=padding_shape, fill_value=float("-inf"), dtype=previous_conv.dtype, device = previous_conv.device, requires_grad= previous_conv.requires_grad)

        if padding_front:

            new_input = torch.concat((previous_conv, padding), dim = 2)

        if not padding_front:
            new_input = torch.concat((padding, previous_conv), dim = 2)

        return maxpool(new_input)
    
    else:
        return maxpool(previous_conv)
