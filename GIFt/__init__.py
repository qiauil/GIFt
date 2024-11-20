from collections.abc import Iterator
import torch.nn as nn
from .strategies import FineTuningStrategy
from .utils import get_class_name
from .utils.hooks import fine_tuning_sd_hook,fine_tuning_loadsd_posthook
from .utils.network_tool import freeze_module,trainable_parameters
from .meta_types import FinetuableModule

class ModuleIterator(Iterator):
    """
    An iterator that iterates over the layers of a given PyTorch module.
    The iterator returns the layer name, global_name, layer class name, layer object, and a boolean indicating if the layer has child layers.

    Args:
        module (nn.Module): The PyTorch module to iterate over.

    Attributes:
        module (nn.Module): The PyTorch module being iterated over.
        iterations (list): A list of tuples containing the names and layers of the module.
        _index (int): The current index of the iterator.

    Methods:
        __len__(): Returns the number of layers in the module.
        __next__(): Returns the next layer in the iteration.

    """

    def __init__(self, module: nn.Module,parent_name:str) -> None:
        self.module = module
        self.iterations = list(self.module._modules.items())
        self._index = 0
        self.parent_name=parent_name
    
    def __len__(self):
        return len(self.iterations)
    
    def __next__(self):
        if self._index < len(self):
            layer_name, layer = self.iterations[self._index]
            layer_class_name = get_class_name(layer)
            has_child = True if layer._modules else False
            global_name=self.parent_name+'.'+layer_name if self.parent_name !="" else layer_name
            self._index += 1
            return layer_name, global_name, layer_class_name, layer, has_child
        else:
            raise StopIteration

def modify_modules(module:nn.Module,
                   fine_tuning_strategy:FineTuningStrategy,
                   parent_name:str="",
                   recurrence_level:int=0):
    """
    Recursively modifies the modules in a given module based on a fine-tuning strategy.

    Args:
        module (nn.Module): The module to modify.
        fine_tuning_strategy (FineTuningStrategy): The fine-tuning strategy to apply.
        parent_name (str, optional): DO NOT CHANGE ITS VALUE! This parameter is designed for recursivece function.
        recurrence_level (int, optional): DO NOT CHANGE ITS VALUE! This parameter is designed for recursivece function.

    Raises:
        ValueError: If the module type is not supported by the fine-tuning strategy.

    Returns:
        None
    """
    if recurrence_level==0 and len(fine_tuning_strategy.constraint_type)>0:
        type_check=[not isinstance(module,constraint_type) for constraint_type in fine_tuning_strategy.constraint_type]
        if all(type_check):
            e_msg=f"Unsupport module type {get_class_name(module)} for strategy {get_class_name(fine_tuning_strategy)};"
            e_msg+=f"Supported module types are {fine_tuning_strategy.constraint_type}."
            raise ValueError(e_msg)
    for current_name, global_name, class_name, current_module, has_child in ModuleIterator(module,parent_name):
        if isinstance(current_module,FinetuableModule):
            raise ValueError(f"Layer {global_name} is already finetuable")
        module_modified=fine_tuning_strategy(module,current_name,global_name,class_name,current_module)
        if has_child and not module_modified:
            modify_modules(current_module,fine_tuning_strategy,global_name,recurrence_level+1)
    if recurrence_level==0:
        # check parameters for the top module
        fine_tuning_strategy.check_para(parent_module=None,
                                        current_name=parent_name,
                                        global_name=parent_name,
                                        class_name=module.__class__.__name__,
                                        current_module=module)

def enable_fine_tuning(module:nn.Module,
                      fine_tuning_strategy:FineTuningStrategy,
                      replace_parameter_function:bool=True):
    """
    Enable fine-tuning for a given module.

    Args:
        module (nn.Module): The module to enable fine-tuning for.
        fine_tuning_strategy (FineTuningStrategy): The strategy to use for fine-tuning.
        replace_parameter_function (bool): Whether to replace the `parameters` function of the module.
            If True, the `parameters` function will only return trainable parameters. This helps you 
            avoiding you modifying your optimizer initialization code. If you set it as False, you 
            can use the `trainable_parameters` function from `GIFt.utils.network_tool` to get trainable parameters of 
            your network for an optimizer.

    Returns:
        None
    """
    # replace modules
    modify_modules(module,fine_tuning_strategy)
    # add hook to the module to remove untrainable parameters from the state_dict
    module._register_state_dict_hook(fine_tuning_sd_hook)
    # add hook to the module to enable load_state_dict to load the finetuned model
    module.register_load_state_dict_post_hook(fine_tuning_loadsd_posthook)
    # add trainable_parameters function to the module
    if replace_parameter_function:
        setattr(module,"parameters",lambda recurse=True: trainable_parameters(module,recurse))
    
