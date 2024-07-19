import inspect
from omegaconf import OmegaConf,DictConfig
from typing import Union,Dict,Optional
import importlib

def _get_arg_value(frame, target_class):
    """
    Recursively searches for the frame corresponding to the call of `target_class` and returns the corresponding arg_values.

    Args:
        frame (frame): The current frame to inspect.
        target_class (class): The target class to search for.

    Returns:
        inspect.ArgInfo or None: The argument information of the frame where the target_class is found,
        or None if the target_class is not found in any frame.
    """
    if frame is not None:
        arg_info = inspect.getargvalues(frame)
        local = arg_info.locals
        if "__class__" in local:
            if local["__class__"] == type(target_class):
                return arg_info
        if frame.f_back is not None:
            return _get_arg_value(frame.f_back, target_class)
    return None

def load_obj_from_config(path: Optional[str] = None, config_dict: Optional[Union[DictConfig, Dict]] = None):
    """
    Load an object from a YAML configuration file or dictionary.

    Args:
        path (str, optional): The path to the YAML configuration file. Only one of `path` and `config_dict` should be provided.
        config_dict (DictConfig or dict, optional): The configuration dictionary. Only one of `path` and `config_dict` should be provided.

    Returns:
        object: The instantiated object based on the configuration.

    Raises:
        ValueError: If both `path` and `config_dict` are provided or if neither `path` nor `config_dict` are provided.
        ValueError: If the `config_dict` does not contain the 'target' key.

    """
    if path is not None and config_dict is not None:
        raise ValueError("Only one of path and config_dict should be provided.")
    if path is None and config_dict is None:
        raise ValueError("One of path and config_dict should be provided.")
    if path is not None:
        config_dict = OmegaConf.load(path)
    if isinstance(config_dict, DictConfig):
        config_dict = OmegaConf.to_container(config_dict)
    if "target" not in config_dict.keys():
        raise ValueError("The config_dict should at least contain 'target' key.")
    module, cls = config_dict["target"].rsplit(".", 1)
    if "params" not in config_dict.keys():
        return getattr(importlib.import_module(module, package=None), cls)()
    else:
        paras = config_dict["params"]
        return getattr(importlib.import_module(module, package=None), cls)(**paras)

class InitParaRecorder():
    """
    A class for recording the init parameters of a class. Inspired by the `HyperparametersMixin` of `Lightning`.
    Any subclass of this class will have a `init_paras` attribute that contains the full class name and `__init__` parameters of the class.
    If you use this class as a mixin, you should call the `__init__` method of `InitParaRecorder` or explicitly call `collect_init_paras` function in the `__init__`.
    
    Example:
    ```python
    # as subclass:
    class A(InitParaRecorder):
    
    def __init__(self,a,b,c=42,*args,**kwargs):
        super().__init__()
        print(self.init_paras)
    a=A(5,"b",3,4,5,addition=42)
    # >>> {'target': '__main__.A', 'parameters': {'a': 5, 'b': 'b', 'c': 3, 'addition': 42, 'args': (4, 5)}}
    
    #as mixin:
    import torch

    class B(torch.nn.Module,InitParaRecorder):
        
        def __init__(self,a,b,c=42,*args,**kwargs):
            super().__init__()
            self.collect_init_paras()
            print(self.init_paras)
    b=B(5,"b",3,4,5,addition=42)
    # >>> {'target': '__main__.B', 'parameters': {'a': 5, 'b': 'b', 'c': 3, 'addition': 42, 'args': (4, 5)}}
    ```
    
    """

    def __init__(self) -> None:
        self.init_paras={}
        self.collect_init_paras()

    def collect_init_paras(self):
        """
        Collects the hyperparameters from the current frame.

        Returns:
        - init_paras: A dictionary containing the target and parameters.
        """
        if self.init_paras != {}:
            return self.init_paras
        arg_info=_get_arg_value(inspect.currentframe(),self)
        if arg_info is not None:
            keys=arg_info.args
            try:
                keys.remove("self")
            except:
                pass
            para_dict={key:arg_info.locals[key] for key in keys}
            if arg_info.keywords is not None:
                keys_of_kwargs=arg_info.locals[arg_info.keywords].keys()
                if len(keys_of_kwargs)>0:
                    for key in keys_of_kwargs:
                        para_dict[key]=arg_info.locals[arg_info.keywords][key]
            if arg_info.varargs is not None:
                if len(arg_info.locals[arg_info.varargs])>0:
                    para_dict["args"]=arg_info.locals[arg_info.varargs]
            self.init_paras={
                "target":".".join([self.__class__.__module__,self.__class__.__name__]),
                "params":para_dict
            }
        else:
            print("Failed to find frame for current call of __init__")
        return self.init_paras
    
    def save_init_paras(self,path):
        """
        Saves the hyperparameters to a yaml file.

        Args:
        - path (str): The path to save the hyperparameters.
        """
        if self.init_paras == {}:
            raise ValueError("Init parameters not collected. Please refer to the usage of `InitParaRecorder`.")
        if "args" in self.init_paras["params"].keys():
            raise ValueError("The `args` parameter is currently not supported in saving to yaml file as it is hard to reconstruct the original call.")
        OmegaConf.save(OmegaConf.create(self.init_paras),path)

