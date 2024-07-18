from enum import Enum,StrEnum
import torch.nn as nn

class FinetuningType(Enum):
    FINE_TUNE = 0
    FREEZE = 1
    TRAIN = 2

class ConvWeightHWType(StrEnum):
    """
    Enumeration class for convolutional weight height-width types.
    
    Attributes:
        BALANCED (str): The balanced weight height-width type.
            Try to make $H$ and $W$ balanced.
        OUTASH (str): The out_as_h weight height-width type.
            Set $H$ as the number of output channels. 
            Used in `FSGAN: Subject Agnostic Face Swapping and Reenactment, ICCV 2019` and
            `SVDiff : Compact Parameter Space for Diffusion Fine- Tuning, CVPR 2023`
    """

    BALANCED = "balanced" 
    OUTASH = "out_as_h"
    

class FinetuableModule(nn.Module):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)