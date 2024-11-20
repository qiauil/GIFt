from typing import Any, Sequence,Tuple,Callable,Dict,Optional
from warnings import warn
from ..utils import default,get_class_name,take_intersection
from ..utils import default
from ..utils.configs import InitParaRecorder,load_obj_from_config
import torch.nn as nn
from typing import Type,Union,Optional,Dict


class _CAPBase():
    
    def __init__(self,
                 checks_actions_paras: Sequence[Tuple[Callable, Callable, dict]] = [],) -> None:
        self.caps = checks_actions_paras  

    def checks(self):
        """
        Returns a list of check functions.

        Returns:
            list: A list of check functions.

        """
        return [cap[0] for cap in self.caps]

    def actions(self):
        """
        Returns a list of action functions.

        Returns:
            list: A list of action functions.

        """
        return [cap[1] for cap in self.caps]
    
    def paras(self):
        """
        Returns a list of action parameters.

        Returns:
            list: A list of action parameters.

        """
        return [self._extract_cap(cap)[2] for cap in self.caps]
    
    def _extract_cap(self, cap: Sequence):
        """
        Extracts the check function, action function, and action parameters from a cap tuple.

        Args:
            cap (Sequence): A tuple containing the check function, action function, and action parameters.

        Returns:
            Tuple: A tuple containing the check function, action function, and action parameters.

        Raises:
            ValueError: If the cap tuple does not have the correct length.

        """
        if len(cap) == 2:
            return cap[0], cap[1], {}
        elif len(cap) == 3:
            return cap[0], cap[1], cap[2]
        else:
            raise ValueError("The cap pair must be (check_func, act_func) or (check_func, act_func, act_para).")

    def register_cap(self, check_func: Callable, act_func: Callable, act_para: Optional[dict] = None):
        """
        Registers a new cap tuple.

        Args:
            check_func (Callable): The check function.
            act_func (Callable): The action function.
            act_para (Optional[dict], optional): The action parameters. Defaults to None.

        """
        self.caps.append((check_func, act_func, default(act_para, {})))
    
    def register_caps(self, caps: Sequence[Tuple[Callable, Callable, Optional[dict]]]):
        """
        Registers multiple cap tuples.

        Args:
            caps (Sequence[Tuple[Callable, Callable, Optional[dict]]]): The cap tuples to register.

        """
        for cap in caps:
            self.register_cap(*cap)

    def __len__(self):
        return len(self.caps)
    
    def __getitem__(self, index):
        return self.caps[index]

class FineTuningStrategy(InitParaRecorder):
    """
    A class representing a fine-tuning strategy.

    Args:
        module_caps (Optional[Sequence[Tuple[Callable[[nn.Module,str,str,str,nn.Module],bool], Callable, dict]], optional):
            A list of tuples containing the check function, action function, and action parameters for modules.
            Defaults to an empty list.
        para_caps (Optional[Sequence[Tuple[Callable[[nn.Module,str,str,str,nn.Module,str,nn.Parameter],bool], Callable, dict]], optional):
            A list of tuples containing the check function, action function, and action parameters for parameters.
            Defaults to an empty list.
        constraint_type (Optional[Union[Sequence[Type], Type]], optional): 
            A list of types or a single type. In `enable_fine_tuning`, the module type will be checked against this list.
            The strategy will only be applied to modules of the specified type(s).

    Attributes:
        caps (list): A list of tuples containing the check function, action function, and action parameters.
        constraint_type (list): A list of types that the strategy is constrainted to.

    """

    def __init__(self,
                 module_caps: Sequence[Tuple[Callable[[nn.Module,str,str,str,nn.Module],bool], Callable, dict]] = [],
                 para_caps: Sequence[Tuple[Callable[[nn.Module,str,str,str,nn.Module,str,nn.Parameter],bool], Callable, dict]] = [],
                 constraint_type: Optional[Union[Sequence[Type], Type]] = []) -> None:
        super().__init__()
        self.constraint_type = constraint_type
        if not isinstance(self.constraint_type, Sequence):
            self.constraint_type = [self.constraint_type]
        self.moule_caps = _CAPBase(module_caps)
        self.para_caps = _CAPBase(para_caps)
        
    def __call__(self,
                 parent_module: Optional[nn.Module], 
                 current_name: str, 
                 global_name: str, 
                 class_name: str, 
                 current_module: nn.Module) -> Any:
        """
        Executes the fine-tuning strategy.

        Args:
            parent_module (nn.Module): The parent module.
            current_name (str): The current name.
            global_name (str): The global name.
            class_name (str): The class name.
            current_module (nn.Module): The current module.

        Returns:
            Any: True if the strategy is applicable, False otherwise.

        """
        module_modified=self.check_module(parent_module, current_name, global_name, class_name, current_module)
        if not module_modified:
            self.check_para(parent_module, current_name, global_name, class_name, current_module)
        return module_modified
        
    def check_module(self,
                     parent_module: Optional[nn.Module], 
                 current_name: str, 
                 global_name: str, 
                 class_name: str, 
                 current_module: nn.Module):
        module_modified=False
        for cap in self.moule_caps:
            check_func, act_func, act_para = self.moule_caps._extract_cap(cap)
            if check_func(parent_module, current_name, global_name, class_name, current_module):
                if isinstance(act_func, FineTuningStrategy):
                    assert act_para == {}, f"Unexpected parameter {act_para} for strategy {get_class_name(act_para)} as an action function."
                    act_func.check_module(parent_module, current_name, global_name, class_name, current_module)
                else:
                    act_func(parent_module, current_name, global_name, class_name, current_module,**act_para)    
                module_modified=True
                break
        return module_modified
    
    def check_para(self,
                   parent_module: Optional[nn.Module], 
                current_name: str, 
                 global_name: str, 
                 class_name: str, 
                 current_module: nn.Module):

        for name, para in current_module.named_parameters(recurse=False):
            find=False
            for cap in self.para_caps:
                check_func, act_func, act_para = self.para_caps._extract_cap(cap)
                if check_func(parent_module, current_name, global_name, class_name, current_module, name, para):
                    if isinstance(act_func, FineTuningStrategy):
                        assert act_para == {}, f"Unexpected parameter {act_para} for strategy {get_class_name(act_para)} as an action function."
                        act_func.check_para(parent_module, current_name, global_name, class_name, current_module)
                    else:
                        act_func(parent_module, current_name, global_name, class_name, current_module, name, para, **act_para)
                    find=True
                    break
            if not find:
                para.requires_grad=False

    def regisier_constraint_type(self, constraint_type: Type):
        """
        Registers a new constraint type.

        Args:
            constraint_type (Type): The constraint type to register.

        """
        self.constraint_type.append(constraint_type)
    
    def regisier_constraint_types(self, constraint_types: Sequence[Type]):
        """
        Registers multiple constraint types.

        Args:
            constraint_types (Sequence[Type]): The constraint types to register.

        """
        self.constraint_type.extend(constraint_types)


def DeBugStrategy(strategy:FineTuningStrategy) -> FineTuningStrategy:
    """
    A strategy for debugging purposes. 
    This function will return a new strategy that prints the information of the current modules/paras when the check functions return true.
    
    Args:
        strategy (FineTuningStrategy): The strategy to debug.

    Returns:
        FineTuningStrategy: The debugged strategy.

    """
    module_checks=strategy.moule_caps.checks()
    para_checks=strategy.para_caps.checks()
    return FineTuningStrategy(
        module_caps=[(
                    module_checks[i],
                    lambda parent_module, current_name, global_name, class_name, current_module: 
                        print(f"module check_func {i} is true. the target info are {current_name}, {global_name}, {class_name}"),
                    {}
                    ) for i in range(len(module_checks))],
        para_caps=[(
                    para_checks[i],
                    lambda parent_module, current_name, global_name, class_name, current_module, name, para: 
                        print(f"para check_func {i} is true. the target info are {current_name}, {global_name}, {class_name}, {name}"),
                    {}
                    ) for i in range(len(para_checks))],
        constraint_type=strategy.constraint_type
    )
            
class FullFineTuningStrategy(FineTuningStrategy):
    
    def __init__(self, ) -> None:
        super().__init__([],[(lambda *args,**kwargs: True,
                              lambda *args,**kwargs: True,
                              {})],[])

def merger_strategy(strategies: Sequence[FineTuningStrategy],
                    additional_module_caps: Sequence[Tuple[Callable[[nn.Module,str,str,str,nn.Module],bool], Callable, dict]] = [],
                    additional_para_caps: Sequence[Tuple[Callable[[nn.Module,str,str,str,nn.Module,str,nn.Parameter],bool], Callable, dict]] = [],) -> FineTuningStrategy:
    """
    Merges multiple FineTuningStrategy objects into a single FineTuningStrategy object.

    Args:
        strategies (Sequence[FineTuningStrategy]): A sequence of FineTuningStrategy objects.
        additional_module_caps (Sequence[Tuple[Callable[[nn.Module,str,str,str,nn.Module],bool], Callable, dict]], optional):
            Additional module caps to add to the merged strategy. Defaults to an empty list.
        additional_para_caps (Sequence[Tuple[Callable[[nn.Module,str,str,str,nn.Module,str,nn.Parameter],bool], Callable, dict]], optional):
            Additional parameter caps to add to the merged strategy. Defaults to an empty list.

    Returns:
        FineTuningStrategy: The merged FineTuningStrategy object.

    """
    new_module_caps=[]
    new_para_caps=[]
    constraints=[]
    for strategy in strategies:
        new_module_caps.extend(strategy.moule_caps.caps)
        new_para_caps.extend(strategy.para_caps.caps)
        constraints.append(strategy.constraint_type)
    new_module_caps.extend(additional_module_caps)
    new_para_caps.extend(additional_para_caps)
    new_constraints=[]
    for constraint in constraints:
        if len(constraint)>0:
            new_constraints.append(constraint)
    if len(new_constraints)>0:
        new_constraints=take_intersection(new_constraints)
            
    return FineTuningStrategy(new_module_caps,new_para_caps, new_constraints)

def load_strategy_from_config(path: Optional[str] = None, config_dict: Optional[Dict] = None) -> FineTuningStrategy:
    """
    Load a fine-tuning strategy from a configuration file or dictionary.

    Args:
        path (Optional[str]): The path to the yaml configuration file. If provided, the configuration will be loaded from the file.
        config_dict (Optional[Dict]): The configuration dictionary. If provided, the configuration will be loaded from the dictionary.

    Returns:
        FineTuningStrategy: The loaded fine-tuning strategy.

    """
    return load_obj_from_config(path, config_dict)