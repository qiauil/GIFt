import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..utils import freeze_module
from ..meta_types import FinetuningType,FinetuableModule

class LoRALayer(FinetuableModule):
    
    def __init__(
        self, 
        rank: int, 
        lora_alpha: int, 
        lora_dropout: float=0.0,
        weights_type: FinetuningType=FinetuningType.FINE_TUNE,
        bias_type: FinetuningType=FinetuningType.FREEZE,
    ):
        super().__init__()
        if rank < 0:
            raise ValueError('Rank must be greater than 0')
        self.rank = rank
        self.lora_alpha = lora_alpha
        
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
            
        if weights_type != FinetuningType.FINE_TUNE and bias_type != FinetuningType.FINE_TUNE:
            raise ValueError('At least one of the weights or bias must be fine-tuned')
        self.weights_type = weights_type
        self.bias_type = bias_type
        # Mark the weight as unmerged
        
        self.merged = False
        self.scaling = self.lora_alpha / self.rank


class LoRALinear(LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        parent_module: nn.Linear,
        rank:int,
        lora_alpha: int,
        lora_dropout: float=0.0,
        weights_type: FinetuningType=FinetuningType.FINE_TUNE,
        bias_type: FinetuningType=FinetuningType.FREEZE,
    ):
        super().__init__(rank, lora_alpha, lora_dropout,weights_type,bias_type)
        self.parent_module = parent_module
        # Actual trainable parameters
        if self.weights_type == FinetuningType.FINE_TUNE:
            self.lora_weight_A = nn.Parameter(self.parent_module.weight.new_zeros((self.parent_module.in_features,rank)))
            self.lora_weight_B = nn.Parameter(self.parent_module.weight.new_zeros((rank,self.parent_module.out_features)))
            self._weight_lora_initilization(self.lora_weight_A)
            self._bias_lora_initilization(self.lora_weight_B)
        if self.bias_type == FinetuningType.FINE_TUNE:
            self.lora_bias_A = nn.Parameter(self.parent_module.weight.new_zeros((self.parent_module.in_features,rank)))
            self.lora_bias_B = nn.Parameter(self.parent_module.weight.new_zeros((rank,self.parent_module.out_features)))
            self._weight_lora_initilization(self.lora_bias_A)
            self._bias_lora_initilization(self.lora_bias_B)
            nn.init.zeros_(self.lora_bias_B)
        freeze_module(self.parent_module,
                      weights_type=self.weights_type,
                      bias_type=self.bias_type)

    def _weight_lora_initilization(self,weight_para):
        nn.init.kaiming_uniform_(weight_para, a=math.sqrt(5))
        
    def _bias_lora_initilization(self,bias_para):
        nn.init.zeros_(bias_para)

    def train(self, mode: bool = True):
        # Note: eval() is actually tran(False), thus we do not need to override it
        self.parent_module.train(mode)
        if mode:
            if self.merged:
                # Make sure that the weights are not merged
                if self.weights_type == FinetuningType.FINE_TUNE:
                    self.parent_module.weight.data -= self.lora_weight_A @ self.lora_weight_B * self.scaling
                if self.bias_type == FinetuningType.FINE_TUNE:
                    self.parent_module.bias.data -= self.lora_bias_A @ self.lora_bias_B * self.scaling
                self.merged = False
        else:
            if not self.merged:
                # Merge the weights and mark it
                if self.weights_type == FinetuningType.FINE_TUNE:
                    self.parent_module.weight.data += self.lora_weight_A @ self.lora_weight_B * self.scaling
                if self.bias_type == FinetuningType.FINE_TUNE:
                    self.parent_module.bias.data += self.lora_bias_A @ self.lora_bias_B * self.scaling
                self.merged = True       
        
    def forward(self, x: torch.Tensor):
        result = self.parent_module(x)   
        if not self.merged:
            if self.weights_type == FinetuningType.FINE_TUNE:
                result += (self.lora_weight_A @ self.lora_weight_B @ self.lora_dropout(x)) * self.scaling
            if self.bias_type == FinetuningType.FINE_TUNE:
                result += (self.lora_bias_A @ self.lora_bias_B) * self.scaling
            return result
        else:
            return result
