from typing import Any, Sequence,Tuple,Callable,Dict,Optional
from warnings import warn
from ..utils import default,get_class_name,take_intersection
import torch.nn as nn
from typing import Type,Union,Optional,Dict


class FineTuningStrategy():
    
    def __init__(self,
                 checks_actions_parnames:Sequence[Tuple[Callable,Callable,dict]]=[],
                 constrain_type:Optional[Union[Sequence[Type],Type]]=[]
                 ) -> None:
        self.caps=checks_actions_parnames  
        self.constrain_type=constrain_type
        if not isinstance(self.constrain_type,Sequence):
            self.constrain_type=[self.constrain_type]
        
    def __call__(self,
                 parent_module:nn.Module, 
                 current_name:str, 
                 global_name:str, 
                 class_name:str, 
                 current_module:nn.Module) -> Any:
        if len(self.caps)==0:
            # if no cap is provided, then the strategy is not applicable.
            return True
        for cap in self.caps:
            check_func,act_func,act_para=self._extract_cap(cap)
            if check_func(parent_module, current_name, global_name, class_name, current_module):
                if isinstance(act_func,FineTuningStrategy):
                    if act_para is not {}:
                        warn(f"Unexpected parameter {act_para} for strategy {get_class_name(act_para)} as an action function.")
                    if act_func(parent_module, current_name, global_name, class_name, current_module):
                        return True
                else:
                    act_func(parent_module, current_name, global_name, class_name, current_module,**act_para)
                    return True
        return False

    def checks(self):
        return [cap[0] for cap in self.caps]

    def actions(self):
        return [cap[1] for cap in self.caps]
    
    def paras(self):
        return [self._extract_cap(cap)[2] for cap in self.caps]
    
    def _extract_cap(self,cap:Sequence):
        if len(cap)==2:
            return cap[0],cap[1],{}
        elif len(cap)==3:
            return cap[0],cap[1],cap[2]
        else:
            raise ValueError("The cap pair must be (check_func,act_func) or (check_func,act_func,act_para).")

    def register_cap(self,check_func:Callable,act_func:Callable,act_para:Optional[dict]=None):
        self.caps.append((check_func,act_func,default(act_para,{})))
    
    def register_caps(self,caps:Sequence[Tuple[Callable,Callable,Optional[dict]]]):
        for cap in caps:
            self.register_cap(*cap)
    
    def regisier_constarin_type(self,constrain_type:Type):
        self.constrain_type.append(constrain_type)
    
    def regisier_constarin_types(self,constrain_types:Sequence[Type]):
        self.constrain_type.extend(constrain_types)

def merger_strategy(strategies: Sequence[FineTuningStrategy]) -> FineTuningStrategy:
    new_caps = []
    constrains=[]
    for strategy in strategies:
        new_caps.extend(strategy.caps)
        constrains.append(strategy.constrain_type)
    new_constrains=[]
    for constrain in constrains:
        if len(constrain)>0:
            new_constrains.append(constrain)
    if len(new_constrains)>0:
        new_constrains=take_intersection(new_constrains)
            
    return FineTuningStrategy(new_caps, new_constrains)