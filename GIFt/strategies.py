from typing import Sequence,Tuple,Callable,Dict,Optional
from collections.abc import Iterator
from warnings import warn
from .meta_types import FinetuningType
from .layers.lora import LoRALinear,LoRAConv1d,LoRAConv2d,LoRAConv3d
from .utils import factories as fts

class FineTuningStrategy():
    
    def __init__(self,
                 checks_actions_parnames:Sequence[Tuple[Callable,Callable,str]],
                 default_action_paras:Dict[str,Sequence],
                 customized_action_paras:Optional[Dict[str,Sequence]]=None
                 ) -> None:
        self.caps=checks_actions_parnames
        if customized_action_paras is not None:
            for key1 in customized_action_paras.keys():
                for key2 in customized_action_paras[key1].keys():
                    try:
                        default_action_paras[key1][key2]=customized_action_paras[key1][key2]
                    except KeyError:
                        warn(f"Neglect unsupported fine-tuning parameter: {key1}.{key2}.",RuntimeWarning)
        self.action_paras=default_action_paras
        self._index=0
        
    def __len__(self):
        return len(self.caps)
    
    def __getitem__(self,index):
        check_func=self.caps[index][0]
        act_func=lambda module,name, global_name, class_name, layer_obj: self.caps[index][1](module,name, global_name, class_name, layer_obj,**self.action_paras[self.caps[index][2]])
        return check_func,act_func
    '''    
    def __next__(self):
        if self._index < len(self):
            check_func=self.caps[self._index][0]
            act_func=lambda module,name, global_name, class_name, layer_obj: self.caps[self._index][1](module,name, global_name, class_name, layer_obj,**self.action_paras[self.caps[self._index][2]])
            self._index+=1
            return check_func,act_func
        else:
            raise StopIteration
    '''    
    def paras(self):
        return self.action_paras
        
class LoRAFullFineTuningStrategy(FineTuningStrategy):
    
    def __init__(self,lora_paras:Optional[Dict[str,Dict]]=None) -> None:
        default_lora_paras={
            "lora_paras":{
                "rank":3,
                "lora_alpha":None, 
                "lora_dropout":0.0, 
                "train_bias":False
            } 
        }
        checks_actions_parnames=[
            (fts.c_cname_func("Linear"),fts.a_replace_func(LoRALinear),"lora_paras"),
            (fts.c_cname_func("Conv1d"),fts.a_replace_func(LoRAConv1d),"lora_paras"),
            (fts.c_cname_func("Conv2d"),fts.a_replace_func(LoRAConv2d),"lora_paras"),
            (fts.c_cname_func("Conv3d"),fts.a_replace_func(LoRAConv3d),"lora_paras"),
        ]
        super().__init__(checks_actions_parnames,default_lora_paras,lora_paras)
        
                        
        
    
    
    
    
    
    