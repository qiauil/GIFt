'''
This module contains factory functions that return check functions and action functions for the fine-tuning process.

for modules:

check_func:
    A function that takes in the following parameters and returns True if the module meets the condition, False otherwise.
    
    - parent_module (nn.Module): The parent module.
    - current_name (str): The name of current module.
    - global_name (str): The global name of current modul.
    - class_name (str): The class name of current module.
    - current_module (nn.Module): The current module object.

    Returns:
        bool: True if the module meets the condition, False otherwise.

Functions here starts with mc_ is a check function for modules.
    
action_func: 
    A function that takes in the following parameters and performs the action.
    
    - parent_module (nn.Module): The parent module.
    - current_name (str): The name of current module.
    - global_name (str): The global name of current modul.
    - class_name (str): The class name of current module.
    - current_module (nn.Module): The current module object.
    - **fine_tuning_paras: The additional parameters for the action function.
    
    
Functions here starts with ma_ is an action function for modules.

for parameters:

check_func: 
    - A function that takes in the following parameters and returns True if the parameter meets the condition, False otherwise.
    
    - parent_module (nn.Module): The parent module.
    - current_name (str): The name of current module.
    - global_name (str): The global name of current modul.
    - class_name (str): The class name of current module.
    - current_module (nn.Module): The current module object.
    - parameter_name (str): The name of current parameter.
    - parameter (nn.Parameter): The current parameter object. 
    
    Returns:
        bool: True if the parameter meets the condition, False otherwise.
    
Functions here starts with pc_ is a check function for parameters.
    
action_func: 

    A function that takes in the following parameters and performs the action.
    
    - parent_module (nn.Module): The parent module.
    - current_name (str): The name of current module.
    - global_name (str): The global name of current modul.
    - class_name (str): The class name of current module.
    - current_module (nn.Module): The current module object.
    - parameter_name (str): The name of current parameter.
    - parameter (nn.Parameter): The current parameter object. 
    - **fine_tuning_paras: The additional parameters for the action function.  
    
Functions here starts with pa_ is an action function for parameters.

'''

import torch.nn as nn
from typing import Callable,Sequence

def mc_name_equal2(target_name:str):
    """
    Returns a check function that checks if the module name is equal to the target_name.

    Args:
        target_name (str): The name to compare against.

    Returns:
        check_func (function)
    """
    def check_func(parent_module:nn.Module, current_name:str, global_name:str, class_name:str, current_module:nn.Module):
        if current_name==target_name:
            return True
        else:
            return False
    return check_func

def mc_name_equal2_sequence(target_names:Sequence[str]):
    """
    Returns a check function that checks if the module name is equal to one of the target_names.

    Args:
        target_name (str): The name to compare against.

    Returns:
        check_func (function)
    """
    def check_func(parent_module:nn.Module, 
                   current_name:str, 
                   global_name:str, 
                   class_name:str, 
                   current_module:nn.Module):
        for target_name in target_names: 
            if current_name==target_name:
                return True
        return False
    return check_func

def mc_name_in(target_name:str):
    """
    Returns a check function that checks if the module name is in the target name.

    Parameters:
        target_name (str): The name to compare against.

    Returns:
        check_func (function)
    """
    def check_func(parent_module:nn.Module,current_name:str, global_name:str, class_name:str, current_module:nn.Module):
        if current_name in target_name:
            return True
        else:
            return False
    return check_func

def mc_name_in_sequence(target_names:Sequence[str]):
    """
    Returns a check function that checks if the the module name is in the one of the target name.

    Parameters:
        target_name (str): The name to compare against.

    Returns:
        check_func (function)
    """
    def check_func(parent_module:nn.Module,current_name:str, global_name:str, class_name:str, current_module:nn.Module):
        for target_name in target_names:
            if current_name in target_name:
                return True
        return False
    return check_func

def mc_name_contains(target_name:str):
    """
    Returns a check function that checks if the module name contains the target name.

    Parameters:
        target_name (str): The name to compare against.

    Returns:
        check_func (function)
    """
    def check_func(parent_module:nn.Module,current_name:str, global_name:str, class_name:str, current_module:nn.Module):
        if target_name in current_name:
            return True
        else:
            return False
    return check_func

def mc_name_contains_sequence(target_names:Sequence[str]):
    """
    Returns a check function that checks if the the module name contains one of the target name.

    Parameters:
        target_name (str): The name to compare against.

    Returns:
        check_func (function)
    """
    def check_func(parent_module:nn.Module,current_name:str, global_name:str, class_name:str, current_module:nn.Module):
        for target_name in target_names:
            if target_name in current_name:
                return True
        return False
    return check_func

def mc_cname_equal2(target_classname:str):
    """
    Returns a check function that checks if the module class name is equal to the target_name.

    Args:
        target_name (str): The name to compare against.

    Returns:
        check_func (function)
    """
    def check_func(parent_module:nn.Module,current_name:str, global_name:str, class_name:str, current_module:nn.Module):
            if class_name==target_classname:
                return True
            else:
                return False
    return check_func

def mc_cname_equal2_sequence(target_classnames:Sequence[str]):
    """
    Returns a check function that checks if the module class name is equal to one of the target_names.

    Args:
        target_name (str): The name to compare against.

    Returns:
        check_func (function)
    """
    def check_func(parent_module:nn.Module,current_name:str, global_name:str, class_name:str, current_module:nn.Module):
        for target_classname in target_classnames:
            if class_name==target_classname:
                return True
        return False
    return check_func

def mc_cname_in(target_classname:str):
    """
    Returns a check function that checks if the module class name is in the target name.

    Parameters:
        target_name (str): The name to compare against.

    Returns:
        check_func (function)
    """
    def check_func(parent_module:nn.Module,current_name:str, global_name:str, class_name:str, current_module:nn.Module):
            if class_name in target_classname:
                return True
            else:
                return False
    return check_func

def mc_cname_in_sequence(target_classnames:Sequence[str]):
    """
    Returns a check function that checks if the module class name is in one of the target_names.

    Parameters:
        target_name (str): The name to compare against.

    Returns:
        check_func (function)
    """
    def check_func(parent_module:nn.Module,current_name:str, global_name:str, class_name:str, current_module:nn.Module):
        for target_classname in target_classnames:
            if class_name in target_classname:
                return True
        return False
    return check_func

def mc_cname_contains(target_classname:str):
    """
    Returns a check function that checks if the module class name contains the target name.

    Parameters:
        target_name (str): The name to compare against.

    Returns:
        check_func (function)
    """
    def check_func(parent_module:nn.Module,current_name:str, global_name:str, class_name:str, current_module:nn.Module):
            if target_classname in class_name:
                return True
            else:
                return False
    return check_func

def mc_cname_contains_sequence(target_classnames:Sequence[str]):
    """
    Returns a check function that checks if the module class name contains one of the target_names.

    Parameters:
        target_name (str): The name to compare against.

    Returns:
        check_func (function)
    """
    def check_func(parent_module:nn.Module,current_name:str, global_name:str, class_name:str, current_module:nn.Module):
        for target_classname in target_classnames:
            if target_classname in class_name:
                return True
        return False
    return check_func

def ma_replace(replace_func:Callable):
    """
    return an action function that replaces the current module with a new module.

    Args:
        replace_func: The function that should return a new module to replace the current module.

    Returns:
        action_func (function)

    """
    def action_func(parent_module:nn.Module,current_name:str, global_name:str, class_name:str, current_module:nn.Module,**fine_tuning_paras):
        setattr(
            parent_module,
            current_name,
            replace_func(current_module,**fine_tuning_paras)
            )
    return action_func

def pc_independent():
    """
    return a check function that checks if the parameter is independent. 
    I.e., the current module does not have any submodules but only parameters.
    """
    
    def check_func(parent_module:nn.Module, 
                   current_name:str, 
                   global_name:str, 
                   class_name:str, 
                   current_module:nn.Module,
                   name:str,
                   para:nn.parameter):
        if len(list(current_module._modules.items())) != 0 and len(list(current_module.named_parameters(recurse=False))) != 0:
            return True
        return False
    
    return check_func