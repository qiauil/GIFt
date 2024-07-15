from typing import Callable, Dict,Optional, Sequence, Tuple
from . import FineTuningStrategy
from ..layers.lora import LoRALinear,LoRAConv1d,LoRAConv2d,LoRAConv3d
from ..utils import factories as fts
import torch.nn as nn

class LoraConfigMixin():
    
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

class LoRALinearFineTuningStrategy(LoraConfigMixin,FineTuningStrategy):
    
    def __init__(self, rank: int = 3, 
                 lora_alpha: float | None = None, 
                 lora_dropout: float = 0, 
                 train_bias: bool = False) -> None:
        LoraConfigMixin.__init__(self,rank,lora_alpha,lora_dropout,train_bias)
        FineTuningStrategy.__init__(self,
                                    [
                                        (fts.c_cname_equal("Linear"),fts.a_replace(LoRALinear),self.lora_configs()),
                                    ]
                                    )
        
class LoRAConvFineTuningStrategy(LoraConfigMixin,FineTuningStrategy):
    
    def __init__(self, rank: int = 3, 
                 lora_alpha: float | None = None, 
                 lora_dropout: float = 0, 
                 train_bias: bool = False) -> None:
        LoraConfigMixin.__init__(self,rank,lora_alpha,lora_dropout,train_bias)
        FineTuningStrategy.__init__(self,
                                        [
                                            (fts.c_cname_equal("Conv1d"),fts.a_replace(LoRAConv1d),self.lora_configs()),
                                            (fts.c_cname_equal("Conv2d"),fts.a_replace(LoRAConv2d),self.lora_configs()),
                                            (fts.c_cname_equal("Conv3d"),fts.a_replace(LoRAConv3d),self.lora_configs()),
                                        ]
                                    )

