import torch.nn as nn

def c_name_func(target_name):
    def check_func(parent_module:nn.Module,name:str, global_name:str, class_name:str, layer_obj:nn.Module):
            if name==target_name:
                return True
            else:
                return False
    return check_func

def c_cname_func(target_classname):
    def check_func(parent_module:nn.Module,name:str, global_name:str, class_name:str, layer_obj:nn.Module):
            if class_name==target_classname:
                return True
            else:
                return False
    return check_func

def a_replace_func(replace_func):
    def action_func(parent_module:nn.Module,name:str, global_name:str, class_name:str, layer_obj:nn.Module,**fine_tuning_paras):
        setattr(
            parent_module,
            name,
            replace_func(layer_obj,**fine_tuning_paras)
            )
    return action_func