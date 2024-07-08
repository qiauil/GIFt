import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Union,Optional
from ..utils import freeze_module,default
from ..meta_types import FinetuningType,FinetuableModule

class LoRALayer(FinetuableModule):
    
    def __init__(
        self, 
        rank: int, 
        lora_alpha: Optional[int]=None, 
        lora_dropout: float=0.0
    ):
        super().__init__()
        if rank < 0:
            raise ValueError('Rank must be greater than 0')
        self.rank = rank
        self.lora_alpha = default(lora_alpha, rank)
        
        self.run_dropout = True if lora_dropout > 0. else False
        if self.run_dropout:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        self.merged = False
        self.scaling = self.lora_alpha / self.rank

    def _lora_B_initialization(self,B_para):
        nn.init.zeros_(B_para)
        
    def _lora_A_initilization(self,A_para):
        nn.init.kaiming_uniform_(A_para, a=math.sqrt(5))

class LoRALinearLike(LoRALayer):
    
    def __init__(self,
                 parent_module:nn.Module, 
                 h_weight: int,
                 w_weight: int,
                 rank: int, 
                 lora_alpha: Optional[int]=None, 
                 lora_dropout: float = 0, 
                 train_bias=False):
        super().__init__(rank, lora_alpha, lora_dropout)
        self.parent_module = parent_module
        # Actual trainable parameters
        self.lora_B = nn.Parameter(self.parent_module.weight.new_zeros((h_weight,rank)))
        self.lora_A = nn.Parameter(self.parent_module.weight.new_zeros((rank,w_weight)))
        self._lora_B_initialization(self.lora_B)
        self._lora_A_initilization(self.lora_A)
        
        weight_shape=self.parent_module.weight.shape
        if len(weight_shape) != 2 or (weight_shape[0]!=h_weight or weight_shape[1]!=w_weight):
            if torch.numel(self.parent_module.weight) != w_weight*h_weight:
                raise ValueError('The shape of the parent module weight is not compatible with the LoRA input and output size')
            self.shape_transfer = lambda ori_tensor: ori_tensor.view(h_weight,rank,w_weight)
        else:
            self.shape_transfer = lambda ori_tensor: ori_tensor
        bias_type=FinetuningType.FREEZE if not train_bias else FinetuningType.TRAIN
        freeze_module(self.parent_module,
                      weights_type=FinetuningType.FINE_TUNE,
                      bias_type=bias_type)

    def train(self, mode: bool = True):
        # Note: eval() is actually tran(False), thus we do not need to override it
        self.parent_module.train(mode)
        if mode:
            if self.merged:
                self.parent_module.weight.data -= self.shape_transfer(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if not self.merged:
                self.parent_module.weight.data += self.shape_transfer(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True  
    
    def lora_weight(self):
        return self.shape_transfer(self.lora_B @ self.lora_A)*self.scaling
        
    
    def forward(self, x: torch.Tensor):
        raise NotImplementedError('This is an abstract class, please use LoRALinear')

class LoRAConvLike(LoRALinearLike):
    
    def __init__(self, parent_module: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d],
                 rank: int, 
                 lora_alpha: Optional[int]=None, 
                 lora_dropout: float = 0, 
                 train_bias=False):
        h_weight=parent_module.weight.shape[0] # parent_module.out_channels//self.parent_module.groups
        w_weight=parent_module.weight.shape[1] # parent_module.in_channels
        if isinstance(parent_module, nn.Conv1d):
            h_weight *=parent_module.kernel_size[0]
        elif isinstance(parent_module, nn.Conv2d):
            h_weight *= parent_module.kernel_size[0]
            w_weight *= parent_module.kernel_size[1]
        elif isinstance(parent_module, nn.Conv3d):
            h_weight *= parent_module.kernel_size[0]*parent_module.kernel_size[1]
            w_weight *= parent_module.kernel_size[2]
        super().__init__(parent_module, h_weight, w_weight, rank, lora_alpha, lora_dropout, train_bias)

    def forward(self, x):
        if not self.merged:
            if self.run_dropout:
                self.conv(x)+self.conv._conv_forward(
                self.lora_dropout(x), 
                self.lora_weight(),
                bias=None)
            else:    
                return self.conv._conv_forward(
                    x, 
                    self.conv.weight + self.lora_weight(),
                    self.conv.bias
                )
        return self.conv(x)

class LoRALinear(LoRALinearLike):
    
    def __init__(self, 
                 parent_module: nn.Linear, 
                 rank: int, 
                 lora_alpha: Optional[int]=None, 
                 lora_dropout: float = 0.0, 
                 train_bias=False):
        super().__init__(parent_module, 
                         parent_module.out_features, 
                         parent_module.in_features, 
                         rank, lora_alpha, 
                         lora_dropout, 
                         train_bias)
        
    def forward(self, x: torch.Tensor):
        if not self.merged:
            if self.run_dropout:
                '''
                There are three method to add the lora_weight to the parent_module:
                * `self.parent_module(x)+(self.lora_weight() @ x.T).T` 
                * `self.parent_module(x)+torch.functional.F.linear(x,self.lora_weight())`
                * `torch.functional.F.linear(x,self.parent_module.weight+self.lora_weight(),self.parent_module.bias)`
                The first method is the most slow one and the last one is the fastest one.
                However, the last one does not support dropout.
                Thus, if dropout is used, the second method is the best choice, otherwise the last one is the best choice.
                '''
                return self.parent_module(x)+F.linear(self.lora_dropout(x),
                                                      self.lora_weight(),
                                                      bias=None)
            else:
                return F.linear(x,
                                self.parent_module.weight+self.lora_weight(),
                                self.parent_module.bias)
        else:
            return self.parent_module(x)  

class LoRAConv1d(LoRAConvLike):
    
    def __init__(self, 
                 parent_module: nn.Conv1d, 
                 rank: int, 
                 lora_alpha: Optional[int]=None, 
                 lora_dropout: float = 0.0, 
                 train_bias=False):
        super().__init__(parent_module, rank, lora_alpha, lora_dropout, train_bias)

class LoRAConv2d(LoRAConvLike):
        
    def __init__(self, 
                parent_module: nn.Conv2d, 
                rank: int, 
                lora_alpha: Optional[int]=None, 
                lora_dropout: float = 0.0, 
                train_bias=False):
        super().__init__(parent_module, rank, lora_alpha, lora_dropout, train_bias)

class LoRAConv3d(LoRAConvLike):
    
    def __init__(self, 
                 parent_module: nn.Conv3d, 
                 rank: int, 
                 lora_alpha: Optional[int]=None, 
                 lora_dropout: float = 0.0, 
                 train_bias=False):
        super().__init__(parent_module, rank, lora_alpha, lora_dropout, train_bias)