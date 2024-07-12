import torch.nn as nn

def c_name_equal(target_name):
    def check_func(parent_module:nn.Module,name:str, global_name:str, class_name:str, current_module:nn.Module):
            if name==target_name:
                return True
            else:
                return False
    return check_func

def c_name_in(target_name):
    def check_func(parent_module:nn.Module,name:str, global_name:str, class_name:str, current_module:nn.Module):
            if name in target_name:
                return True
            else:
                return False
    return check_func

def c_cname_equal(target_classname):
    def check_func(parent_module:nn.Module,name:str, global_name:str, class_name:str, current_module:nn.Module):
            if class_name==target_classname:
                return True
            else:
                return False
    return check_func

def c_cname_in(target_classname):
    def check_func(parent_module:nn.Module,name:str, global_name:str, class_name:str, current_module:nn.Module):
            if class_name in target_classname:
                return True
            else:
                return False
    return check_func

def a_replace(replace_func):
    def action_func(parent_module:nn.Module,name:str, global_name:str, class_name:str, current_module:nn.Module,**fine_tuning_paras):
        setattr(
            parent_module,
            name,
            replace_func(current_module,**fine_tuning_paras)
            )
    return action_func