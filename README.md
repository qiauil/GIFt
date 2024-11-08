# GIFt: Generic and Intuitive Fine-tuning Library

Fine-tuning is a common technique in deep learning to adapt a pre-trained model to a new task. It is widely used in computer vision, natural language processing, and other domains. However, fine-tuning is not always straightforward. It requires a good understanding of the model, the dataset, and the task. In this notebook, we introduce GIFt, a generic and intuitive fine-tuning library that simplifies the process of fine-tuning pre-trained models. GIFt is designed to be easy to use, flexible, and extensible. 

## Quick Start

### Structure of a neural network and Caps
A modern neural network usually consists of many layers. In each layer, there are two main components: a submodule which links to the next layer (or not) and some independent parameters (at the last layer, there will only be parameters).

```
Network
- module1
    - submodule1
        - parameters
        - subsubmodule1
            - ***
                ***
                -lastsubmodule
                    - parameter
    - submodule2
        - parameters
- parameters
```

The core of fine-tuning a network is to modify part of the submodule and parameters and freeze the rest with the pre-trained result. To enable this, we introduce the concept of `Caps`, i.e., "check-action-parameters". A `Caps` is a sequence of tuple where each tuple contains three elements: a check function, an action function, and a parameter. The check function is used to determine whether the submodule or parameter should be modified. The action function is used to modify the submodule or parameter. The parameter is the value that will be used in the action function. With a designed `Caps`, we can easily fine-tune a network.

The allowed check function in `GIFt` should have the following structure:

for submodule:
```
check_func (function):
    A function that takes in the following parameters and returns True if the module meets the condition, False otherwise.
    
    - parent_module (nn.Module): The parent module.
    - current_name (str): The name of current module.
    - global_name (str): The global name of current modul.
    - class_name (str): The class name of current module.
    - current_module (nn.Module): The current module object.

    Returns:
        bool: True if the module meets the condition, False otherwise.
```

for parameter:
```
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
```

The corresponding action function receives the same parameters as the check function and modifies the submodule or parameter. You can also set additional parameters to the action function with the `parameter` in caps tuple.

All the above parameters can be get through the `ModuleIterator` class in `GIFt`. Here is an simple example
:


```python
import torch.nn as nn
import torch
from GIFt import ModuleIterator

class TestNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.nn_seq = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

net=TestNet()
iterator=ModuleIterator(net,"TestNet")
for current_name, global_name, class_name, current_module, has_children in iterator:
    print(current_name, global_name, class_name, has_children)
    for para_name, param in current_module.named_parameters(recurse=False):
        print(">", global_name, para_name)
```

    conv1 TestNet.conv1 Conv2d False
    > TestNet.conv1 weight
    > TestNet.conv1 bias
    pool TestNet.pool MaxPool2d False
    fc1 TestNet.fc1 Linear False
    > TestNet.fc1 weight
    > TestNet.fc1 bias
    nn_seq TestNet.nn_seq Sequential True


In `GIFt.factories` module, we provide some common check functions and action functions. You can also define your own check functions and action functions. 

### Finetuning strategy

In `GIFt`, we use a `FineTuningStrategy` class to orgainze the `Caps`:


```python
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
    """
```

`FineTuningStrategy` collects all the `Caps` and can be applied to a network. You may notice that there is an additional initialization parameter `constraint_type` in the class. You can set a specific type to this `constraint_type` to make sure this strategy only works on the specific type of module. Note that this type check is not working inside of `FineTuningStrategy` but in the `enable_fine_tuning` function.


```python
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
```

`enable_fine_tuning` function is the main function to apply the `FineTuningStrategy` to a network. It will iterate through all the modules and parameters in the network and apply the `Caps` to the network. It will also replace the state_dict of the network to make sure that everytime you save or load the model, you will only save or load the fine-tuned part.

In `enable_fine_tuning` function, It will first run the check function for modules, if check function returns True, it will run the action function (If you just don't want to make any changes to the module but just keep it training, your action function can just do nothing and return None). Then it will run the check function for parameters, if check function returns True, it will run the action function, otherwise, it will freeze the parameter. Finally, If it detects that current module has a submodule, it will recursively run the `enable_fine_tuning` function on the submodule.

Currently, we provide some bilit-in `FineTuningStrategy` in `GIFt.strategies` module. You can also define your own `FineTuningStrategy` by inheriting the `FineTuningStrategy` class. Here is a very simple example to enable fine-tuning on all the `nn.Linear` modules with LoRA:


```python
from torchvision.models.resnet import resnet18
from GIFt import enable_fine_tuning
from GIFt.strategies.lora import LoRAAllFineTuningStrategy
from GIFt.utils.info import collect_trainable_parameters,table_info

net=resnet18()
paras_info,num_paras=collect_trainable_parameters(net)
print("Before fine-tuning, the number of trainable parameters is:",num_paras)
enable_fine_tuning(net,LoRAAllFineTuningStrategy())
paras_info,num_paras=collect_trainable_parameters(net)
print("After fine-tuning, the number of trainable parameters is:",num_paras)
print(table_info(paras_info,header=["index","Name","Type","Shape"]))
```

    Before fine-tuning, the number of trainable parameters is: 11689512
    After fine-tuning, the number of trainable parameters is: 75063
    --------------------------------------------------
    index | Name                         | Type      | Shape
    --------------------------------------------------
    0     | conv1.lora_B                 | [448, 3]  | 1344 
    1     | conv1.lora_A                 | [3, 21]   | 63   
    2     | layer1.0.conv1.lora_B        | [192, 3]  | 576  
    3     | layer1.0.conv1.lora_A        | [3, 192]  | 576  
    4     | layer1.0.conv2.lora_B        | [192, 3]  | 576  
    5     | layer1.0.conv2.lora_A        | [3, 192]  | 576  
    6     | layer1.1.conv1.lora_B        | [192, 3]  | 576  
    7     | layer1.1.conv1.lora_A        | [3, 192]  | 576  
    8     | layer1.1.conv2.lora_B        | [192, 3]  | 576  
    9     | layer1.1.conv2.lora_A        | [3, 192]  | 576  
    10    | layer2.0.conv1.lora_B        | [384, 3]  | 1152 
    11    | layer2.0.conv1.lora_A        | [3, 192]  | 576  
    12    | layer2.0.conv2.lora_B        | [384, 3]  | 1152 
    13    | layer2.0.conv2.lora_A        | [3, 384]  | 1152 
    14    | layer2.0.downsample.0.lora_B | [128, 3]  | 384  
    15    | layer2.0.downsample.0.lora_A | [3, 64]   | 192  
    16    | layer2.1.conv1.lora_B        | [384, 3]  | 1152 
    17    | layer2.1.conv1.lora_A        | [3, 384]  | 1152 
    18    | layer2.1.conv2.lora_B        | [384, 3]  | 1152 
    19    | layer2.1.conv2.lora_A        | [3, 384]  | 1152 
    20    | layer3.0.conv1.lora_B        | [768, 3]  | 2304 
    21    | layer3.0.conv1.lora_A        | [3, 384]  | 1152 
    22    | layer3.0.conv2.lora_B        | [768, 3]  | 2304 
    23    | layer3.0.conv2.lora_A        | [3, 768]  | 2304 
    24    | layer3.0.downsample.0.lora_B | [256, 3]  | 768  
    25    | layer3.0.downsample.0.lora_A | [3, 128]  | 384  
    26    | layer3.1.conv1.lora_B        | [768, 3]  | 2304 
    27    | layer3.1.conv1.lora_A        | [3, 768]  | 2304 
    28    | layer3.1.conv2.lora_B        | [768, 3]  | 2304 
    29    | layer3.1.conv2.lora_A        | [3, 768]  | 2304 
    30    | layer4.0.conv1.lora_B        | [1536, 3] | 4608 
    31    | layer4.0.conv1.lora_A        | [3, 768]  | 2304 
    32    | layer4.0.conv2.lora_B        | [1536, 3] | 4608 
    33    | layer4.0.conv2.lora_A        | [3, 1536] | 4608 
    34    | layer4.0.downsample.0.lora_B | [512, 3]  | 1536 
    35    | layer4.0.downsample.0.lora_A | [3, 256]  | 768  
    36    | layer4.1.conv1.lora_B        | [1536, 3] | 4608 
    37    | layer4.1.conv1.lora_A        | [3, 1536] | 4608 
    38    | layer4.1.conv2.lora_B        | [1536, 3] | 4608 
    39    | layer4.1.conv2.lora_A        | [3, 1536] | 4608 
    40    | fc.lora_B                    | [1000, 3] | 3000 
    41    | fc.lora_A                    | [3, 512]  | 1536 
    --------------------------------------------------

