from enum import Enum
import torch.nn as nn

class FinetuningType(Enum):
    FINE_TUNE = 0
    FREEZE = 1
    TRAIN = 2

class FinetuableModule(nn.Module):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)