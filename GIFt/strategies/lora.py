from typing import Callable, Dict,Optional, Sequence, Tuple
from . import FineTuningStrategy,merger_strategy
from ..modules.lora import LoRALinear,LoRAConv1d,LoRAConv2d,LoRAConv3d
from ..utils import factories as fts
import torch.nn as nn

class LoRAConfigMixin():
    
    def __init__(self,rank:int=3,
                 lora_alpha:Optional[float]=None,
                 lora_dropout:float=0.0,
                 train_bias:bool=False) -> None:
        self._rank=rank
        self._lora_alpha=lora_alpha
        self._lora_dropout=lora_dropout
        self._train_bias=train_bias
        self._lora_configs={
            "rank":self._rank,
            "lora_alpha":self._lora_alpha, 
            "lora_dropout":self._lora_dropout, 
            "train_bias":self._train_bias
        }
    
    def lora_configs(self) -> Dict:
        return self._lora_configs   

class LoRALinearFineTuningStrategy(LoRAConfigMixin,FineTuningStrategy):
    
    def __init__(self, rank: int = 3, 
                 lora_alpha: float | None = None, 
                 lora_dropout: float = 0, 
                 train_bias: bool = False) -> None:
        LoRAConfigMixin.__init__(self,rank,lora_alpha,lora_dropout,train_bias)
        FineTuningStrategy.__init__(self,
                                    [
                                        (fts.mc_cname_equal2("Linear"),fts.ma_replace(LoRALinear),self.lora_configs()),
                                    ]
                                    )
        
class LoRAConvFineTuningStrategy(LoRAConfigMixin,FineTuningStrategy):
    
    def __init__(self, rank: int = 3, 
                 lora_alpha: float | None = None, 
                 lora_dropout: float = 0, 
                 train_bias: bool = False) -> None:
        LoRAConfigMixin.__init__(self,rank,lora_alpha,lora_dropout,train_bias)
        FineTuningStrategy.__init__(self,
                                        [
                                            (fts.mc_cname_equal2("Conv1d"),fts.ma_replace(LoRAConv1d),self.lora_configs()),
                                            (fts.mc_cname_equal2("Conv2d"),fts.ma_replace(LoRAConv2d),self.lora_configs()),
                                            (fts.mc_cname_equal2("Conv3d"),fts.ma_replace(LoRAConv3d),self.lora_configs()),
                                        ]
                                    )

def LoRAAllFineTuningStrategy(
    rank: int = 3, 
    lora_alpha: float | None = None, 
    lora_dropout: float = 0, 
    train_bias: bool = False):
    return merger_strategy([LoRALinearFineTuningStrategy(rank,lora_alpha,lora_dropout,train_bias),
                           LoRAConvFineTuningStrategy(rank,lora_alpha,lora_dropout,train_bias)])