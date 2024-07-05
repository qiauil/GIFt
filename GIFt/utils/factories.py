def c_name_func(target_name):
    def check_func(name:str, global_name:str, class_name:str, layer_obj):
            if name==target_name:
                return True
            else:
                return False
    return check_func

def c_cname_func(target_classname):
    def check_func(name:str, global_name:str, class_name:str, layer_obj):
            if class_name==target_classname:
                return True
            else:
                return False
    return check_func

def a_replace_func(replace_func):
    def action_func(module, name, global_name, class_name, layer_obj,**lora_paras):
        setattr(
            module,
            name,
            replace_func(layer_obj,**lora_paras)
            )
    return action_func