# -*- coding: utf-8 -*-
'''
LoRA: Low-Rank Adaptation of Large Language Models: https://arxiv.org/abs/2106.09685

.. math::
    W = W_0+BA
where
.. math::
    \text{Size of }W_0:(h,w)
    \text{Size of }B:(h,r)
    \text{Size of }A:(r,w)

where r is the rank.

modified from: https://github.com/microsoft/LoRA
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Union,Optional
from ..utils import freeze_module,default
from ..meta_types import FinetuningType,FinetuableModule
torch.bmm()

class LoRALayer(FinetuableModule):
    """
    LoRALayer is a class that represents a layer in the LoRA (Learnable Rank Aggregation) model.

    Args:
        rank (int): The rank of the LoRALayer. Must be greater than 0.
        lora_alpha (Optional[int]): The alpha value for LoRA. Defaults to None.
            In the LoRA paper, the alpha is used to rescale the Lora weight :math:`BA` as math:`\frac{\alpha}{r}BA`.
            If None, the alpha value is set to the rank. So the final scaling factor is always 1, which is the same as the LoRA paper.
        lora_dropout (float): The optional dropout rate for LoRA. Defaults to 0.0.
    """
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
        """
        Initializes the B parameter with zeros.
        In the LoRA paper, they mentioned that 'We use a random Gaussian initialization for A and zero for B'.

        Args:
            B_para (torch.Tensor): The B parameter tensor.
        """
        nn.init.zeros_(B_para)
        
    def _lora_A_initilization(self,A_para):
        """
        Initializes the A parameter using Kaiming uniform initialization.
        In the LoRA paper, they mentioned that 'We use a random Gaussian initialization for A and zero for B'.
        The kaiming_uniform initialization is the default initialization for the weight in PyTorch.
        This is also the same in the official implementation of LoRA.

        Args:
            A_para (torch.Tensor): The A parameter tensor.
        """
        nn.init.kaiming_uniform_(A_para, a=math.sqrt(5))

class LoRALinearLike(LoRALayer):
    """
    A class representing a LoRA linear-like layer.
    A linear-like layer is a layer that has a weight matrix and a bias matrix, such as nn.Linear and nn.Conv1/2/3d.
    Usually we will only fine-tune the weight matrix, and freeze the bias matrix. Considering the input feature is
    :math:`w` and the output feature is :math:`h`, then the shape of weight matrix and bias matrix is :math:`h \times w` 
    and :math:`h`, respectively. The fined-tuned weight matrix can be smaller than the original weight matrix, if 
    :math:`r \times (h+w) < h \times w`. However, if we also fine-tune the bias matrix, the size of the new bias matrix
    is always larger than the original bias matrix as :math:`h \times r +r > h`. In the official implementation of LoRA,
    the bias matrix is not freeze and not fine-tuned. However, in this implementation, we can choose to freeze the bias matrix
    by default.

    Args:
        parent_module (nn.Module): The parent module.
        h_weight (int): The number of input channels.
        w_weight (int): The number of output channels.
        rank (int): The rank of the LoRA decomposition.
        lora_alpha (Optional[int], optional): The alpha parameter for LoRA. Defaults to None.
        lora_dropout (float, optional): The dropout rate for LoRA. Defaults to 0.
        train_bias (bool, optional): Whether to train the bias. Defaults to False.
    """

    def __init__(self,
                 parent_module: nn.Module,
                 h_weight: int,
                 w_weight: int,
                 rank: int,
                 lora_alpha: Optional[int] = None,
                 lora_dropout: float = 0,
                 train_bias: bool = False):
        super().__init__(rank, lora_alpha, lora_dropout)
        self.parent_module = parent_module
        # Actual trainable parameters
        self.lora_B = nn.Parameter(self.parent_module.weight.new_zeros((h_weight, rank)))
        self.lora_A = nn.Parameter(self.parent_module.weight.new_zeros((rank, w_weight)))
        self._lora_B_initialization(self.lora_B)
        self._lora_A_initilization(self.lora_A)

        weight_shape = self.parent_module.weight.shape
        if len(weight_shape) != 2 or (weight_shape[0] != h_weight or weight_shape[1] != w_weight):
            if torch.numel(self.parent_module.weight) != w_weight * h_weight:
                raise ValueError('The shape of the parent module weight is not compatible with the LoRA input and output size')
            self.shape_transfer = lambda ori_tensor: ori_tensor.view(h_weight, rank, w_weight)
        else:
            self.shape_transfer = lambda ori_tensor: ori_tensor
        bias_type = FinetuningType.FREEZE if not train_bias else FinetuningType.TRAIN
        freeze_module(self.parent_module,
                      weights_type=FinetuningType.FINE_TUNE,
                      bias_type=bias_type)

    def train(self, mode: bool = True):
        """
        Sets the module in training mode and updates the weights accordingly.

        Args:
            mode (bool, optional): Whether to set the module in training mode. Defaults to True.
        """
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
        """
        Computes the LoRA weight.

        Returns:
            torch.Tensor: The LoRA weight.
        """
        return self.shape_transfer(self.lora_B @ self.lora_A) * self.scaling

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the module.

        Args:
            x (torch.Tensor): The input tensor.

        Raises:
            NotImplementedError: This is an abstract class, please use LoRALinear.
        """
        raise NotImplementedError('This is an abstract class, please use LoRALinear')

class LoRAConvLike(LoRALinearLike):
    """
    Convolutional layer with LoRA. It is recommended to use `LoraConv1d`, `LoraConv2d`, and `LoraConv3d` instead of this class.

    Args:
        parent_module (Union[nn.Conv1d, nn.Conv2d, nn.Conv3d]): The parent convolutional module.
        rank (int): The rank of the LoRA regularization.
        lora_alpha (Optional[int], optional): The alpha parameter for LoRA regularization. Defaults to None.
        lora_dropout (float, optional): The dropout rate for LoRA regularization. Defaults to 0.
        train_bias (bool, optional): Whether to train the bias term. Defaults to False.
    """

    def __init__(self, parent_module: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d],
                 rank: int, 
                 lora_alpha: Optional[int]=None, 
                 lora_dropout: float = 0, 
                 train_bias=False):
        '''
        The weight of Convolution layer is 
        .. math::
            (\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},\text{kernel\_size[0]}, \text{kernel\_size[1] (Conv2d and Conv3d)}, \text{kernel\_size[2] (Conv3d)})
        we need to separate the weight shape of kernels into B and A according to different convolution types.
        '''
        h_weight=parent_module.weight.shape[0] # parent_module.out_channels
        w_weight=parent_module.weight.shape[1] # parent_module.in_channels//self.parent_module.groups
        if isinstance(parent_module, nn.Conv1d):
            w_weight *=parent_module.kernel_size[0]
        elif isinstance(parent_module, nn.Conv2d):
            h_weight *= parent_module.kernel_size[0]
            w_weight *= parent_module.kernel_size[1]
        elif isinstance(parent_module, nn.Conv3d):
            h_weight *= parent_module.kernel_size[0]
            w_weight *= parent_module.kernel_size[1]* parent_module.kernel_size[2]
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
    """
    Linear layer with LoRA.

    Args:
        parent_module nn.Linear: The parent linear module.
        rank (int): The rank of the LoRA regularization.
        lora_alpha (Optional[int], optional): The alpha parameter for LoRA regularization. Defaults to None.
        lora_dropout (float, optional): The dropout rate for LoRA regularization. Defaults to 0.
        train_bias (bool, optional): Whether to train the bias term. Defaults to False.
    """
    
    def __init__(self, 
                 parent_module: nn.Linear, 
                 rank: int, 
                 lora_alpha: Optional[int]=None, 
                 lora_dropout: float = 0.0, 
                 train_bias=False):        super().__init__(parent_module, 
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
    """
    LoRAConv1d is a class that represents a 1D LoRA convolutional layer.

    Args:
        parent_module (nn.Conv1d): The parent module of the LoRAConv1d layer.
        rank (int): The rank of the LoRAConv1d layer.
        lora_alpha (Optional[int], optional): The alpha parameter for LoRAConv1d. Defaults to None.
        lora_dropout (float, optional): The dropout rate for LoRAConv1d. Defaults to 0.0.
        train_bias (bool, optional): Whether to train the bias of LoRAConv1d. Defaults to False.
    """
    
    def __init__(self, 
                 parent_module: nn.Conv1d, 
                 rank: int, 
                 lora_alpha: Optional[int]=None, 
                 lora_dropout: float = 0.0, 
                 train_bias=False):
        super().__init__(parent_module, rank, lora_alpha, lora_dropout, train_bias)

class LoRAConv2d(LoRAConvLike):
    """
    LoRAConv1d is a class that represents a 2D LoRA convolutional layer.

    Args:
        parent_module (nn.Conv2d): The parent module of the LoRAConv1d layer.
        rank (int): The rank of the LoRAConv1d layer.
        lora_alpha (Optional[int], optional): The alpha parameter for LoRAConv1d. Defaults to None.
        lora_dropout (float, optional): The dropout rate for LoRAConv1d. Defaults to 0.0.
        train_bias (bool, optional): Whether to train the bias of LoRAConv1d. Defaults to False.
    """
        
    def __init__(self, 
                parent_module: nn.Conv2d, 
                rank: int, 
                lora_alpha: Optional[int]=None, 
                lora_dropout: float = 0.0, 
                train_bias=False):
        super().__init__(parent_module, rank, lora_alpha, lora_dropout, train_bias)

class LoRAConv3d(LoRAConvLike):
    """
    LoRAConv3d is a class that represents a 2D LoRA convolutional layer.

    Args:
        parent_module (nn.Conv3d): The parent module of the LoRAConv1d layer.
        rank (int): The rank of the LoRAConv1d layer.
        lora_alpha (Optional[int], optional): The alpha parameter for LoRAConv1d. Defaults to None.
        lora_dropout (float, optional): The dropout rate for LoRAConv1d. Defaults to 0.0.
        train_bias (bool, optional): Whether to train the bias of LoRAConv1d. Defaults to False.
    """
    
    def __init__(self, 
                 parent_module: nn.Conv3d, 
                 rank: int, 
                 lora_alpha: Optional[int]=None, 
                 lora_dropout: float = 0.0, 
                 train_bias=False):
        super().__init__(parent_module, rank, lora_alpha, lora_dropout, train_bias)