from typing import Any, Sequence,Tuple,Callable,Dict,Optional
from warnings import warn
from ..utils import default
import torch.nn as nn
from typing import Type,Union,Optional,Dict


class FineTuningStrategy():
    
    def __init__(self,
                 checks_actions_parnames:Sequence[Tuple[Callable,Callable,str]],
                 action_paras:Dict[str,Sequence],
                 constrain_type:Optional[Union[Sequence[Type],Type]]=None
                 ) -> None:
        self.caps=checks_actions_parnames
        self.action_paras=action_paras    
        self.constrain_type=default(constrain_type,[nn.Module])
        if not isinstance(self.constrain_type,Sequence):
            self.constrain_type=[self.constrain_type]
        
    def __call__(self,
                 parent_module:nn.Module, 
                 current_name:str, 
                 global_name:str, 
                 class_name:str, 
                 current_module:nn.Module,
                 high_priority_paras:Optional[dict]=None) -> Any:
        paras=default(high_priority_paras,self.action_paras)
        for check_func,act_func,act_para in self.caps:
            if check_func(parent_module, current_name, global_name, class_name, current_module):
                if isinstance(act_para,FineTuningStrategy):
                   if act_func(parent_module, current_name, global_name, class_name, current_module,paras[act_para]):
                       return True
                else:
                    act_func(parent_module, current_name, global_name, class_name, current_module,**paras[act_para])
                    return True
        return False

    def actions(self):
        return [action for _,action,_ in self.caps]
    
    def checks(self):
        return [check for check,_,_ in self.caps]
    
    def para_cites(self):
        return [para for _,_,para in self.caps]

    def paras(self):
        return self.action_paras

def merger_strategy(strategies: Sequence[FineTuningStrategy], new_action_paras: Optional[Dict[str, Dict]] = None) -> FineTuningStrategy:
    """
    Merge multiple fine-tuning strategies into a single strategy.
    You can also provide new action parameters for the merged strategy.
    If you don't provide new action parameters, the action parameters of the sub-strategies will be merged.
    If there are any conflict of keys during the merge of the action parameters, the conflict key will be renamed.

    Example:
        .. code-block:: python
            from GIFt.stragegies.lora import LoRALinearFineTuningStrategy,LoRAConvFineTuningStrategy
            from GIFt.stragegies import merger_strategy

            strategy_1=LoRALinearFineTuningStrategy()
            strategy_2=LoRAConvFineTuningStrategy()
            merged=merger_strategy([strategy_1,strategy_2],strategy_1.paras())
            merged.paras()
            merged.caps

    Args:
        strategies (Sequence[FineTuningStrategy]): A sequence of FineTuningStrategy objects to be merged.
        new_action_paras (Optional[Dict[str, Dict]]): Optional new action parameters for the merged strategy.

    Returns:
        FineTuningStrategy: The merged FineTuningStrategy object.

    Raises:
        ValueError: If a key in the sub-strategies is not present in the new action parameters.

    """
    checks_actions_parnames = []
    if new_action_paras is not None:
        key_new_action_paras = new_action_paras.keys()
        for i, strategy in enumerate(strategies):
            checks_actions_parnames.extend(strategy.caps)
            for key in strategy.action_paras.keys():
                if key not in key_new_action_paras:
                    raise ValueError(f"Key {key} of the sub-strategies is not in the new action parameters.")
        return FineTuningStrategy(checks_actions_parnames, new_action_paras)
    else:
        action_paras = {}
        for i, strategy in enumerate(strategies):
            key_pairs = {}
            for key in strategy.action_paras.keys():
                if key in action_paras:
                    key_pairs[key] = key + "_{}".format(i)
                    warn(f"Key {key} is renamed to {key_pairs[key]} because of conflict.")
                else:
                    key_pairs[key] = key
            checks_actions_parnames.extend(
                [(check, action, key_pairs[para]) for check, action, para in strategy.caps])
            for key, value in strategy.action_paras.items():
                action_paras[key_pairs[key]] = value
        return FineTuningStrategy(checks_actions_parnames, action_paras)