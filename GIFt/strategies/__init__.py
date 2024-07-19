from typing import Any, Sequence,Tuple,Callable,Dict,Optional
from warnings import warn
from ..utils import default,get_class_name,take_intersection
from ..utils import factories as fct
import torch.nn as nn
from typing import Type,Union,Optional,Dict

class FineTuningStrategy():
    """
    A class representing a fine-tuning strategy.

    Args:
        checks_actions_parnames (Sequence[Tuple[Callable, Callable, dict]], optional): 
            A list of tuples containing the check function, action function, and action parameters.
            Defaults to an empty list.
        constrain_type (Optional[Union[Sequence[Type], Type]], optional): 
            A list of types or a single type that the strategy is constrained to.
            Defaults to an empty list.

    Attributes:
        caps (list): A list of tuples containing the check function, action function, and action parameters.
        constrain_type (list): A list of types that the strategy is constrained to.

    """

    def __init__(self,
                 checks_actions_parnames: Sequence[Tuple[Callable, Callable, dict]] = [],
                 constrain_type: Optional[Union[Sequence[Type], Type]] = []) -> None:
        self.caps = checks_actions_parnames  
        self.constrain_type = constrain_type
        if not isinstance(self.constrain_type, Sequence):
            self.constrain_type = [self.constrain_type]
        
    def __call__(self,
                 parent_module: nn.Module, 
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
        if len(self.caps) == 0:
            # if no cap is provided, then the strategy is not applicable.
            # i.e. full fine-tuning
            return True
        for cap in self.caps:
            check_func, act_func, act_para = self._extract_cap(cap)
            if check_func(parent_module, current_name, global_name, class_name, current_module):
                if isinstance(act_func, FineTuningStrategy):
                    if act_para is not {}:
                        warn(f"Unexpected parameter {act_para} for strategy {get_class_name(act_para)} as an action function.")
                    if act_func(parent_module, current_name, global_name, class_name, current_module):
                        return True
                else:
                    act_func(parent_module, current_name, global_name, class_name, current_module, **act_para)
                    return True
        return False

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
    
    def regisier_constarin_type(self, constrain_type: Type):
        """
        Registers a new constrain type.

        Args:
            constrain_type (Type): The constrain type to register.

        """
        self.constrain_type.append(constrain_type)
    
    def regisier_constarin_types(self, constrain_types: Sequence[Type]):
        """
        Registers multiple constrain types.

        Args:
            constrain_types (Sequence[Type]): The constrain types to register.

        """
        self.constrain_type.extend(constrain_types)

class DeBugStrategy(FineTuningStrategy):
    """
    A strategy for debugging purposes.
    
    Args:
        check_funcs: Sequence[Callable]: 
            A list of check functions. If the check_func returns true, it will print the information of current layers.
    """

    def __init__(self, check_funcs: Sequence[Callable]) -> None:
        super().__init__()
        for i, check_func in enumerate(check_funcs):
            self.register_cap(
                check_func,
                lambda parent_module, current_name, global_name, class_name, current_module: print(
                    f"check_func {i} is true. the target info are {current_name}, {global_name}, {class_name}"
                ),
            )
            
class FullFineTuningStrategy(FineTuningStrategy):
    
    def __init__(self, ) -> None:
        super().__init__([], [])

class UnitStrategy(FineTuningStrategy):
    
    def __init__(self) -> None:
        super().__init__()

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