import torch.nn as nn
from typing import Union
from ..meta_types import FinetuningType,ConvWeightHWType

def freeze_module(module: nn.Module, weights_type: FinetuningType=FinetuningType.FINE_TUNE,bias_type: FinetuningType=FinetuningType.FREEZE):
    """
    Freezes or unfreezes the parameters of a given module based on the specified fine_tuning types.

    Args:
        module (nn.Module): The module whose parameters need to be frozen or unfrozen.
        weights_type (FinetuningType, optional): The fine_tuning type for weight parameters. Defaults to FinetuningType.FINE_TUNE.
        bias_type (FinetuningType, optional): The fine_tuning type for bias parameters. Defaults to FinetuningType.FREEZE.
    """
    for name,param in module.named_parameters():
        if "weight" in name and weights_type == FinetuningType.TRAIN:
            param.requires_grad = True
        elif "bias" in name and bias_type == FinetuningType.TRAIN:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
def trainable_parameters(module:nn.Module,recurse:bool=True):
    for name, param in module.named_parameters(recurse=recurse):
        if param.requires_grad:
            yield param

def num_trainable_parameters(module:nn.Module):
    return sum(p.numel() for p in trainable_parameters(module))

def num_parameters(module:nn.Module):
    return sum(p.numel() for p in module.parameters())

def conv_weight_hw(module:Union[nn.Conv1d,nn.Conv2d,nn.Conv3d],
                   conv_weight_hw_type:ConvWeightHWType=ConvWeightHWType.BALANCED)->tuple[int,int]:
    """
    Transfer the shape of the weight tensor of convolutional layer, 
    $(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},\text{kernel\_size[0]}, \text{kernel\_size[1] (Conv2d and Conv3d)}, \text{kernel\_size[2] (Conv3d)})$, 
    to a simple $(H \times W)$.

    Args:
        module (Union[nn.Conv1d, nn.Conv2d, nn.Conv3d]): The convolutional module.
        conv_weight_hw_type (ConvWeightHWType, optional): The type of weight tensor. Defaults to ConvWeightHWType.BALANCED.

    Returns:
        tuple[int, int]: A tuple containing the height and width of the convolutional weight tensor.
    """
    if conv_weight_hw_type==ConvWeightHWType.BALANCED:
        h_weight=module.weight.shape[0] # module.out_channels
        w_weight=module.weight.shape[1] # module.in_channels//self.module.groups
        if isinstance(module, nn.Conv1d):
            w_weight *=module.kernel_size[0]
        elif isinstance(module, nn.Conv2d):
            h_weight *= module.kernel_size[0]
            w_weight *= module.kernel_size[1]
        elif isinstance(module, nn.Conv3d):
            h_weight *= module.kernel_size[0]
            w_weight *= module.kernel_size[1]* module.kernel_size[2]
    elif conv_weight_hw_type==ConvWeightHWType.OUTASH:
        h_weight=module.weight.shape[0] # module.out_channels
        w_weight=1 # module.in_channels//self.module.groups
        for kernel_shape_i in list(module.weight.shape)[1:]:
            w_weight*=kernel_shape_i       
    return h_weight,w_weight

'''
The weight of Convolution layer is 
.. math::
    
we need to separate the weight shape of kernels into B and A according to different convolution types.
'''