from typing import Callable, Dict,Optional, Sequence, Tuple
from . import FineTuningStrategy
from ..layers.lora import LoRALinear,LoRAConv1d,LoRAConv2d,LoRAConv3d
from ..utils import factories as fts

class LoRAFineTuningStrategyBasis(FineTuningStrategy):
    
    def __init__(self,
                 checks_actions_parnames:Sequence[Tuple[Callable,Callable,str]],
                 rank:int=3,
                 lora_alpha:Optional[float]=None,
                 lora_dropout:float=0.0,
                 train_bias:bool=False,
                 ) -> None:
        lora_paras={"lora":{
            "rank":rank,
            "lora_alpha":lora_alpha, 
            "lora_dropout":lora_dropout, 
            "train_bias":train_bias
        }
        }
        super().__init__(checks_actions_parnames,lora_paras)

class LoRALinearFineTuningStrategy(LoRAFineTuningStrategyBasis):
    
    def __init__(self, rank: int = 3, 
                 lora_alpha: float | None = None, 
                 lora_dropout: float = 0, 
                 train_bias: bool = False) -> None:
        checks_actions_parnames=[
            (fts.c_cname_equal("Linear"),fts.a_replace(LoRALinear),"lora"),
        ]
        super().__init__(checks_actions_parnames, rank, lora_alpha, lora_dropout, train_bias)
        
class LoRAConvFineTuningStrategy(LoRAFineTuningStrategyBasis):
    
    def __init__(self, rank: int = 3, 
                 lora_alpha: float | None = None, 
                 lora_dropout: float = 0, 
                 train_bias: bool = False) -> None:
        checks_actions_parnames=[
            (fts.c_cname_equal("Conv1d"),fts.a_replace(LoRAConv1d),"lora"),
            (fts.c_cname_equal("Conv2d"),fts.a_replace(LoRAConv2d),"lora"),
            (fts.c_cname_equal("Conv3d"),fts.a_replace(LoRAConv3d),"lora"),
        ]
        super().__init__(checks_actions_parnames, rank, lora_alpha, lora_dropout, train_bias)

