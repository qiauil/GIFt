import torch.nn as nn
from ..meta_types import FinetuningType

def default(object,default_value):
    """
    Returns the default value if the object is None.

    Args:
        object (Any): The object to check.
        default_value (Any): The default value to return if the object is None.

    Returns:
        Any: The object if it is not None, otherwise the default value.
    """
    return object if object is not None else default_value

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

def get_class_name(obj):
    return obj.__class__.__name__