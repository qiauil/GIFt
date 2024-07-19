import torch.nn as nn
def fine_tuning_sd_hook(module:nn.Module, state_dict, *args, **kwargs):
    '''
    Clean the state_dict of the module, removing all the parameters that are not trainable.
    It is better to remove all the parameters that are not trainable from the state_dict rather than create a new state_dict
    rather than create a new state_dict with trainable parameters only. This is because sometimes the state_dict also contains 
    untrainable buffers, which should be kept in the state_dict.
    '''
    new_state_dict = {}
    not_requires_grad_paras=[name for name,param in module.named_parameters() if not param.requires_grad]
    # Maybe also include buffer?
    for key, value in state_dict.items():
        if key not in not_requires_grad_paras:
            new_state_dict[key] = value
    return new_state_dict

def fine_tuning_loadsd_posthook(module:nn.Module, incompatible_keys):
    '''
    Enable load_state_dict to load the finetuned model.
    The default load_state_dict will raise an error since it also tries to load the unfinetuned parameters.
    If you don't want to load this hook, you can also set `strick=False` in `load_state_dict` function.
    '''
    finetuned_sd_keys=module.state_dict().keys()
    key_copys=incompatible_keys.missing_keys.copy()
    for key in key_copys:
        if key not in finetuned_sd_keys:
            incompatible_keys.missing_keys.remove(key)