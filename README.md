# GIFt: Generic and Intuitive Fine-tuning Library

## Examples of using GIFt for fine-tuning:

First, Let's build a neural network


```python
import torch
import torch.nn as nn
from GIFt import enable_fine_tuning
from GIFt.strategies import LoRAFullFineTuningStrategy
from GIFt.utils import num_trainable_parameters

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,num_layers):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for i in range(num_layers-1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.relu = nn.ReLU()
    
    def forward(self, x):
        for i in range(self.num_layers):
            x = self.layers[i](x)
            x = self.relu(x)
        x = self.layers[-1](x)
        return x

mlp=MLP(1, 100, 1, 5)
print(mlp)
print("Network before enable fine-tuning:",mlp)
print("Number of trainable parameters:",num_trainable_parameters(mlp))
```

    MLP(
      (layers): ModuleList(
        (0): Linear(in_features=1, out_features=100, bias=True)
        (1-4): 4 x Linear(in_features=100, out_features=100, bias=True)
        (5): Linear(in_features=100, out_features=1, bias=True)
      )
      (relu): ReLU()
    )
    Network before enable fine-tuning: MLP(
      (layers): ModuleList(
        (0): Linear(in_features=1, out_features=100, bias=True)
        (1-4): 4 x Linear(in_features=100, out_features=100, bias=True)
        (5): Linear(in_features=100, out_features=1, bias=True)
      )
      (relu): ReLU()
    )
    Number of trainable parameters: 40701


We can enable fine-tuning for this neural network with a single line of command:


```python
enable_fine_tuning(mlp, LoRAFullFineTuningStrategy())
print("Network after enable fine-tuning:",mlp)
print("Number of trainable parameters after fine-tuning:",num_trainable_parameters(mlp))
```

    Network after enable fine-tuning: MLP(
      (layers): ModuleList(
        (0): LoRALinear(
          (parent_module): Linear(in_features=1, out_features=100, bias=True)
        )
        (1-4): 4 x LoRALinear(
          (parent_module): Linear(in_features=100, out_features=100, bias=True)
        )
        (5): LoRALinear(
          (parent_module): Linear(in_features=100, out_features=1, bias=True)
        )
      )
      (relu): ReLU()
    )
    Number of trainable parameters after fine-tuning: 3006


Here, `LoRAFullFineTuningStrategy` is a subclass of `FineTuningStrategy` where it can replace all Linear layers with LoRA Linear layers and all Conv1/2/3D layers with LoRAConv1/2/3D layers. We will discuss how to build up a new fine-tuning strategy later.

After fine-tuning with the `enable_fine_tuning` function, the `parameters()` of the network instance will be replaced with a new function where only trainable parameters are returned:


```python
optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)
```

You can also use `GIFt.utils.trainable_parameters(mlp)` to get the trainable parameters.

Besides, after enabling fine-tuning for the neural network, the statedict of the model will be updated to only include the trainable parameters (fine-tuned parameters). Thus, we can directly save and load the weights of the fine-tuned network as the conventional way we usually do in PyTorch.


```python
state_dict = mlp.state_dict()
print(state_dict.keys())
torch.save(state_dict, "mlp.pt")
mlp.load_state_dict(torch.load("mlp.pt"))
```

    dict_keys(['layers.0.lora_B', 'layers.0.lora_A', 'layers.1.lora_B', 'layers.1.lora_A', 'layers.2.lora_B', 'layers.2.lora_A', 'layers.3.lora_B', 'layers.3.lora_A', 'layers.4.lora_B', 'layers.4.lora_A', 'layers.5.lora_B', 'layers.5.lora_A'])





    <All keys matched successfully>



## Training strategy

`enable_fine_tuning` function requires an instance of `FineTuningStrategy` class. Actually, any iterable object returns a `check` function, and an `action` function works for the `enable_fine_tuning` function. Here, the `check` function checks whether the layer satisfies some specific condition, and the `action` function will be activated if the `check` function returns true.

The parameters of the `check` and `action` functions are `parent_module, name, global_name, class_name, layer_obj` respectively. Let's use a simple example to show how you a fine-tuning strategy and the meaning of these parameters:


```python
class ExampleMLP(nn.Module):
    def __init__(self):
        super(ExampleMLP, self).__init__()
        self.in_model=nn.Linear(1, 10)
        self.mid_model=MLP(10, 10, 10, 2)
        self.out_model=nn.Linear(10, 1)
    
    def forward(self, x):
        return self.mlp(x)

example_mlp=ExampleMLP()
print("Example MLP:",example_mlp)
```

    Example MLP: ExampleMLP(
      (in_model): Linear(in_features=1, out_features=10, bias=True)
      (mid_model): MLP(
        (layers): ModuleList(
          (0-2): 3 x Linear(in_features=10, out_features=10, bias=True)
        )
        (relu): ReLU()
      )
      (out_model): Linear(in_features=10, out_features=1, bias=True)
    )


The following strategy will replace all the linear layer with convolution layer:


```python
class ExampleStrategy():
    def __init__(self):
        
        def check_function(parent_module,name, global_name, class_name, layer_obj):
            if class_name == "Linear":
                return True
            return False
        
        def action_function(parent_module,name, global_name, class_name, layer_obj):
            print("Parent Module",parent_module)
            print("Layer name:",name)
            print("Global name:",global_name)
            print("Class name:",class_name)
            print("Layer object:",layer_obj)
            print("_"*50)
            setattr(parent_module, name, nn.Conv2d(
                layer_obj.in_features, layer_obj.out_features,
                kernel_size=3
            ))
        self.check_actions=[(check_function, action_function)]
        
    def __len__(self):
        return len(self.check_actions)
    
    def __getitem__(self, index):
        return self.check_actions[index]
    
enable_fine_tuning(example_mlp, ExampleStrategy())
```

    Parent Module ExampleMLP(
      (in_model): Linear(in_features=1, out_features=10, bias=True)
      (mid_model): MLP(
        (layers): ModuleList(
          (0-2): 3 x Linear(in_features=10, out_features=10, bias=True)
        )
        (relu): ReLU()
      )
      (out_model): Linear(in_features=10, out_features=1, bias=True)
    )
    Layer name: in_model
    Global name: in_model
    Class name: Linear
    Layer object: Linear(in_features=1, out_features=10, bias=True)
    __________________________________________________
    Parent Module ModuleList(
      (0-2): 3 x Linear(in_features=10, out_features=10, bias=True)
    )
    Layer name: 0
    Global name: mid_model.layers.0
    Class name: Linear
    Layer object: Linear(in_features=10, out_features=10, bias=True)
    __________________________________________________
    Parent Module ModuleList(
      (0): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1))
      (1-2): 2 x Linear(in_features=10, out_features=10, bias=True)
    )
    Layer name: 1
    Global name: mid_model.layers.1
    Class name: Linear
    Layer object: Linear(in_features=10, out_features=10, bias=True)
    __________________________________________________
    Parent Module ModuleList(
      (0-1): 2 x Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1))
      (2): Linear(in_features=10, out_features=10, bias=True)
    )
    Layer name: 2
    Global name: mid_model.layers.2
    Class name: Linear
    Layer object: Linear(in_features=10, out_features=10, bias=True)
    __________________________________________________
    Parent Module ExampleMLP(
      (in_model): Conv2d(1, 10, kernel_size=(3, 3), stride=(1, 1))
      (mid_model): MLP(
        (layers): ModuleList(
          (0-2): 3 x Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1))
        )
        (relu): ReLU()
      )
      (out_model): Linear(in_features=10, out_features=1, bias=True)
    )
    Layer name: out_model
    Global name: out_model
    Class name: Linear
    Layer object: Linear(in_features=10, out_features=1, bias=True)
    __________________________________________________


Form the previous example, we can know that:
* `enable_fine_tuning` function iterates over all layers from top to bottom, from outside to inside.
* `parent_module` parameter is a `nn.Module` representing the parent module of current layer.
* `layer_name` parameter is a `str` representing the name of current layer.
* `global_name` parameter is a `str` representing the global name of current layer, i.e., it contains all the name of parent layers.
* `layer_obj` is a `nn.Module` representing the current layer.

`FineTuningStrategy` class is a helper class which makes your procedure of designing the training strategy more simpler. It also support additional parameters for the action `function`. You can refer to the source code of `LoRAFullFineTuningStrategy()` to see how it works. Here we give an example of using `FineTuningStrategy` class to build up the previous strategy: 


```python
from typing import Callable, Dict, Sequence, Tuple
from GIFt.strategies import FineTuningStrategy
import GIFt.utils.factories as fts

class ExampleStrategy2(FineTuningStrategy):
    
    def __init__(self, kernel_size=3) -> None:
        default_action_paras = {"conv_para":{"kernel_size": 3}}
        customized_action_paras = {"conv_para":{"kernel_size": kernel_size}}
        checks_actions_parnames = [
            (fts.c_cname_func("Linear"),
             fts.a_replace_func(lambda layer_obj,kernel_size: nn.Conv2d(layer_obj.in_features, layer_obj.out_features, kernel_size=kernel_size)),
             "conv_para")
        ]
        super().__init__(checks_actions_parnames, default_action_paras, customized_action_paras)

example_mlp=ExampleMLP()
enable_fine_tuning(example_mlp, ExampleStrategy2())
print(example_mlp)
```

    ExampleMLP(
      (in_model): Conv2d(1, 10, kernel_size=(3, 3), stride=(1, 1))
      (mid_model): MLP(
        (layers): ModuleList(
          (0-2): 3 x Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1))
        )
        (relu): ReLU()
      )
      (out_model): Conv2d(10, 1, kernel_size=(3, 3), stride=(1, 1))
    )


A good thing of using `FineTuningStrategy` is that we can extract the parameters of fine-tuning and save them separately:


```python
print(ExampleStrategy2().paras())
print(LoRAFullFineTuningStrategy().paras())
```

    {'conv_para': {'kernel_size': 3}}
    {'lora_paras': {'rank': 3, 'lora_alpha': None, 'lora_dropout': 0.0, 'train_bias': False}}


The last thing we need to mention is that we recommend making all the new layers in the fine-tuning model an instance of `GIFt.meta_types.FinetuableModule` as the `enable_fine_tuning` function will check whether a layer is already an instance of the `FinetuableModule` to avoid incorrectly duplicate setting networks.

## An example of applying LoRA fine-tuning to attention layers

The `Q`, `K`, and `V` matrix in attention layer can be calculated through `Linear` layer if the input is a sequence or `Conv` layer is the input is a field. Thus, we can simply use our previous code to enable LoRA fine-tuning for attention layers:

Build up attention layer:


```python
import math
from einops import rearrange

# more examples of attention implementation can be found in my another repository:
# https://github.com/qiauil/Foxutils/blob/main/foxutils/network/attentions.py
class MultiHeadAttentionBase(nn.Module):
    
    def __init__(self, num_heads:int,linear_attention=False, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, queries, keys, values):
        queries,keys,values =map(self.apart_input,(queries,keys,values))
        d_k = keys.shape[-1]
        weights = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d_k)
        weights = nn.functional.softmax(weights, dim=-1)
        return self.concat_output(torch.bmm(self.dropout(weights), values))

    def apart_input(self,x):
        #(batch_size, num_elements, num_heads$\times$dim_deads)  >>> (batch_size, num_elements, num_heads, dim_deads)
        x = x.reshape(x.shape[0], x.shape[1], self.num_heads, -1)
        #(batch_size, num_elements, num_heads, dim_deads) >>> (batch_size, num_heads, num_elements, dim_deads) 
        x = x.permute(0, 2, 1, 3)
        #(batch_size, num_heads, num_elements, dim_deads)  >>> (batch_size$\times$num_heads, num_elements, dim_deads) 
        return x.reshape(-1, x.shape[2], x.shape[3])


    def concat_output(self, x):
        #(batch_size$\times$num_heads, num_elements, dim_deads) >>> (batch_size, num_heads, num_elements, dim_deads)
        x = x.reshape(-1, self.num_heads, x.shape[1], x.shape[2])
        #(batch_size, num_heads, num_elements, dim_deads) >>> (batch_size, num_elements, num_heads, dim_deads)
        x = x.permute(0, 2, 1, 3)
        #(batch_size, num_elements, num_heads, dim_deads) >>> (batch_size, num_elements, num_heads$\times$dim_deads)
        return x.reshape(x.shape[0], x.shape[1], -1)

class SequenceMultiHeadAttention(nn.Module):
    def __init__(self,dim_q:int, dim_k:int, dim_v:int, num_heads:int, dim_heads:int,dim_out:int, linear_attention=False, dropout=0.0,bias=False):
        super().__init__()
        dim_hiddens=num_heads*dim_heads
        self.w_q = nn.Linear(dim_q, dim_hiddens,bias=bias)
        self.w_k = nn.Linear(dim_k, dim_hiddens,bias=bias)
        self.w_v = nn.Linear(dim_v, dim_hiddens,bias=bias)
        self.mha=MultiHeadAttentionBase(num_heads=num_heads,linear_attention=linear_attention,dropout=dropout)
        self.w_o = nn.Linear(dim_hiddens, dim_out,bias=bias)
    
    def forward(self, queries, keys, values):
        q=self.w_q(queries)
        k=self.w_k(keys)
        v=self.w_v(values)
        att=self.mha(q,k,v)
        return self.w_o(att)

class TwoDFieldMultiHeadAttention(nn.Module):

    def __init__(self,dim_q, dim_k, dim_v, num_heads, dim_heads,dim_out, linear_attention=False, dropout=0.0,bias=False):
        super().__init__()
        dim_hiddens=num_heads*dim_heads
        self.w_q = nn.Conv2d(dim_q, dim_hiddens, 1, bias=bias)
        self.w_k = nn.Conv2d(dim_k, dim_hiddens, 1, bias=bias)
        self.w_v = nn.Conv2d(dim_v, dim_hiddens, 1, bias=bias)
        self.mha=MultiHeadAttentionBase(num_heads=num_heads,linear_attention=linear_attention,dropout=dropout)
        self.w_o = nn.Conv2d(dim_hiddens, dim_out,1,bias=bias)
    
    def forward(self, queries, keys, values):
        width=queries.shape[-1]
        q=self.w_q(queries)
        k=self.w_k(keys)
        v=self.w_v(values)
        q, k, v = map(lambda t: rearrange(t, "b c h w -> b (h w) c"), (q,k,v))
        att=self.mha(q,k,v)
        att_2D=rearrange(att,"b (h w) c -> b c h w",w=width)
        return self.w_o(att_2D)
    
```


```python
sequence_attention=SequenceMultiHeadAttention(10,10,10,2,5,10)
print("sequence_attention before enable fine_tuning:")
print(sequence_attention)
enable_fine_tuning(sequence_attention, LoRAFullFineTuningStrategy())
print("sequence_attention after enable fine_tuning:")
print(sequence_attention)
print("")
print("field attention before enable fine_tuning:")
field_attention=TwoDFieldMultiHeadAttention(10,10,10,2,5,10)
print("field attention after enable fine_tuning:")
print(field_attention)
enable_fine_tuning(field_attention, LoRAFullFineTuningStrategy())
print(field_attention)
```

    sequence_attention before enable fine_tuning:
    SequenceMultiHeadAttention(
      (w_q): Linear(in_features=10, out_features=10, bias=False)
      (w_k): Linear(in_features=10, out_features=10, bias=False)
      (w_v): Linear(in_features=10, out_features=10, bias=False)
      (mha): MultiHeadAttentionBase()
      (w_o): Linear(in_features=10, out_features=10, bias=False)
    )
    sequence_attention after enable fine_tuning:
    SequenceMultiHeadAttention(
      (w_q): LoRALinear(
        (parent_module): Linear(in_features=10, out_features=10, bias=False)
      )
      (w_k): LoRALinear(
        (parent_module): Linear(in_features=10, out_features=10, bias=False)
      )
      (w_v): LoRALinear(
        (parent_module): Linear(in_features=10, out_features=10, bias=False)
      )
      (mha): MultiHeadAttentionBase()
      (w_o): LoRALinear(
        (parent_module): Linear(in_features=10, out_features=10, bias=False)
      )
    )
    
    field attention before enable fine_tuning:
    field attention after enable fine_tuning:
    TwoDFieldMultiHeadAttention(
      (w_q): Conv2d(10, 10, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (w_k): Conv2d(10, 10, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (w_v): Conv2d(10, 10, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (mha): MultiHeadAttentionBase()
      (w_o): Conv2d(10, 10, kernel_size=(1, 1), stride=(1, 1), bias=False)
    )
    TwoDFieldMultiHeadAttention(
      (w_q): LoRAConv2d(
        (parent_module): Conv2d(10, 10, kernel_size=(1, 1), stride=(1, 1), bias=False)
      )
      (w_k): LoRAConv2d(
        (parent_module): Conv2d(10, 10, kernel_size=(1, 1), stride=(1, 1), bias=False)
      )
      (w_v): LoRAConv2d(
        (parent_module): Conv2d(10, 10, kernel_size=(1, 1), stride=(1, 1), bias=False)
      )
      (mha): MultiHeadAttentionBase()
      (w_o): LoRAConv2d(
        (parent_module): Conv2d(10, 10, kernel_size=(1, 1), stride=(1, 1), bias=False)
      )
    )

