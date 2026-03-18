# -*- coding: utf-8 -*-
r'''
LoRA: Low-Rank Adaptation of Large Language Models: https://arxiv.org/abs/2106.09685

$$
    W = W_0+BA
$$
where
$$
\begin{matrix}
    \text{Size of }W_0:(h,w)\\
    \text{Size of }B:(h,r)\\
    \text{Size of }A:(r,w)
\end{matrix}
$$

Here, $r$ is the rank.

modified from: https://github.com/microsoft/LoRA
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from typing import Union,Optional
from warnings import warn
from ..utils import default
from ..utils.network_tool import conv_weight_hw,freeze_module
from ..meta_types import FinetuableModule,FinetuningType


class LoRALayer(FinetuableModule):
    """
    Base class of LoRA fine-tuning layers.

    Args:
        rank (int): The rank of the LoRALayer. Must be greater than 0.
        lora_alpha (Optional[int]): The alpha value for LoRA. Defaults to None.
            In the LoRA paper, the alpha is used to rescale the LoRA weight :math:`BA` as math:`\frac{\alpha}{r}BA`.
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
        
    def _lora_A_initialization(self,A_para):
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
        train_bias (bool, optional): Whether to train the bias. Defaults to False. If False, the training of the bias is the same as the origional module.
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
        if h_weight*w_weight < rank*(h_weight+w_weight):
            msg=f"Your rank is so large that the new LoRA weight matrix {h_weight}x{rank}+{rank}x{w_weight} is larger than the original weight matrix {h_weight}x{w_weight}."
            warn(msg)
        self.parent_module = parent_module
        # Actual trainable parameters
        self.lora_B = nn.Parameter(self.parent_module.weight.new_zeros((h_weight, rank)))
        self.lora_A = nn.Parameter(self.parent_module.weight.new_zeros((rank, w_weight)))
        self._lora_B_initialization(self.lora_B)
        self._lora_A_initialization(self.lora_A)

        weight_shape = self.parent_module.weight.shape
        if len(weight_shape) != 2 or (weight_shape[0] != h_weight or weight_shape[1] != w_weight):
            if torch.numel(self.parent_module.weight) != w_weight * h_weight:
                raise ValueError('The shape of the parent module weight is not compatible with the LoRA input and output size')
            self.shape_transfer = lambda ori_tensor: ori_tensor.view(self.parent_module.weight.shape)
        else:
            self.shape_transfer = lambda ori_tensor: ori_tensor
        freeze_module(self.parent_module, 
                      weights_type=FinetuningType.FINE_TUNE,
                      bias_type=FinetuningType.TRAIN if train_bias else FinetuningType.AUTO
                      )

    '''
    We notice that in PyTorch Lighning, the train loop will actually not call the train() function of the module to enable 
    training. Thus, for safety, we will not override the train() function of the module. Instead, we will judge whether we
    are in the training model in forward() function.
    
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
    '''
    def lora_weight(self):
        """
        Computes the LoRA weight.

        Returns:
            torch.Tensor: The LoRA weight.
        """
        return self.shape_transfer(self.lora_B @ self.lora_A) * self.scaling

    def merge_weight(self):
        if self.merged:
            raise ValueError('The weight is already merged')
        self.parent_module.weight.data += self.lora_weight()
        self.merged = True
    
    def unmerge_weight(self):
        if not self.merged:
            raise ValueError('The weight is already unmerged')
        self.parent_module.weight.data -= self.lora_weight()
        self.merged = False

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the module.

        Args:
            x (torch.Tensor): The input tensor.

        Raises:
            NotImplementedError: This is an abstract class, please use LoRALinear.
        """
        raise NotImplementedError('This is an abstract class, please use LoRALinear')
    
    @property
    def weight(self):
        """
        The weight of the parent module.

        Returns:
            torch.Tensor: The weight of the parent module.
        """
        if self.merged:
            return self.parent_module.weight
        else:
            return self.parent_module.weight + self.lora_weight()
    
    @property
    def bias(self):
        """
        The bias of the parent module.

        Returns:
            torch.Tensor: The bias of the parent module.
        """
        return self.parent_module.bias

    def state_dict(self, *args, **kwargs):
        if self.merged:
            self.unmerge_weight()
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        if self.merged:
            self.unmerge_weight()
        super().load_state_dict(state_dict, strict)

class LoRAConv(LoRALinearLike):
    """
    LoRA based Convolutional layer. 
    You can use `LoRAConv1d`, `LoRAConv2d`, and `LoRAConv3d` instead of this class.

    Args:
        parent_module (Union[nn.Conv1d, nn.Conv2d, nn.Conv3d]): The parent convolutional module.
        rank (int): The rank of the LoRA regularization.
        lora_alpha (Optional[int], optional): The alpha parameter for LoRA regularization. Defaults to None.
        lora_dropout (float, optional): The dropout rate for LoRA regularization. Defaults to 0.
        train_bias (bool, optional): Whether to train the bias term. Defaults to False. If False, the training of the bias is the same as the origional module.
    
    Although a more general implementation of LoRAConv is as follows, where we avoid construct the whole LoRA weight matrix 
    (check https://qiauil.github.io/blog/2026/lora/ to see whether you need avoid construct the whole matrix),
    
    class LoRAConv3d(Module):
    def __init__(self, parent_module, rank):
        super().__init__()
        self.parent_module=parent_module
        self.rank=rank
        weight_shapes=tuple(self.parent_module.weight.data.shape)
        self.lora_b=nn.Parameter(self.parent_module.weight.new_zeros((weight_shapes[0],rank)+(1,)*len(weight_shapes[2:])))
        self.lora_a=nn.Parameter(self.parent_module.weight.new_zeros((rank,weight_shapes[1])+weight_shapes[2:]))
        nn.init.kaiming_uniform_(self.lora_a,a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_b,a=math.sqrt(5))
        self.parent_module.weight.requires_grad=False
        self.parent_module.bias.requires_grad=False
        
    def forward(self,x):
        out=self.parent_module(x)
        out+=F.conv3d(self.parent_module._conv_forward(x,self.lora_a,bias=None),weight=self.lora_b,bias=None)
        return out
    
    def merged_forward(self,x):
        dw=torch.einsum("or,ri...->oi...",self.lora_b.view(self.lora_b.shape[0],self.lora_b.shape[1]),self.lora_a)
        return self.parent_module._conv_forward(x,self.parent_module.weight+dw,self.parent_module.bias)
    
    
    the bottleneck of the convolution operation in PyTorch is actually the convolution itself, rather than the addition of the LoRA weight. Thus, we will construct the whole LoRA weight matrix for simplicity.
    Thus we should avoid introducing new convolution operations.
    
    """

    def __init__(self, parent_module: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d],
                 rank: int, 
                 lora_alpha: Optional[int]=None, 
                 lora_dropout: float = 0, 
                 train_bias=False):
        h_weight, w_weight = conv_weight_hw(parent_module)
        super().__init__(parent_module, h_weight, w_weight, rank, lora_alpha, lora_dropout, train_bias)
        

    def forward(self, x):
        if self.training:
            if self.merged:
                self.unmerge_weight()
            if self.run_dropout:
                '''
                see comments in LoRALinearLike
                '''
                return self.parent_module(x)+self.parent_module._conv_forward(
                self.lora_dropout(x), 
                self.lora_weight(),
                bias=None)
            else:    
                return self.parent_module._conv_forward(
                    x, 
                    self.parent_module.weight + self.lora_weight(),
                    self.parent_module.bias
                )
        else:
            if not self.merged:
                self.merge_weight()
            return self.parent_module(x)

class LoRALinear(LoRALinearLike):
    """
    LoRA based Linear layer.

    Args:
        parent_module nn.Linear: The parent linear module.
        rank (int): The rank of the LoRA regularization.
        lora_alpha (Optional[int], optional): The alpha parameter for LoRA regularization. Defaults to None.
        lora_dropout (float, optional): The dropout rate for LoRA regularization. Defaults to 0.
        train_bias (bool, optional): Whether to train the bias term. Defaults to False. If False, the training of the bias is the same as the origional module.
    """
    
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
        
    def forward(self, x):
        if self.training:
            if self.merged:
                self.unmerge_weight()
            out=self.parent_module(x)
            if self.run_dropout:
                x=self.lora_dropout(x)
            out+=(x @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return out
        else:
            if not self.merged:
                self.merge_weight()
            out=self.parent_module(x)
            return out

class LoRAConv1d(LoRAConv):
    """
    A LoRA based Conv1d layer.

    Args:
        parent_module (nn.Conv1d): The parent module of the LoRAConv1d layer.
        rank (int): The rank of the LoRAConv1d layer.
        lora_alpha (Optional[int], optional): The alpha parameter for LoRAConv1d. Defaults to None.
        lora_dropout (float, optional): The dropout rate for LoRAConv1d. Defaults to 0.0.
        train_bias (bool, optional): Whether to train the bias of LoRAConv1d. Defaults to False. If False, the training of the bias is the same as the origional module.
    """
    
    def __init__(self, 
                 parent_module: nn.Conv1d, 
                 rank: int, 
                 lora_alpha: Optional[int]=None, 
                 lora_dropout: float = 0.0, 
                 train_bias=False):
        super().__init__(parent_module, rank, lora_alpha, lora_dropout, train_bias)

class LoRAConv2d(LoRAConv):
    """
    A LoRA based Conv2d layer.

    Args:
        parent_module (nn.Conv2d): The parent module of the LoRAConv1d layer.
        rank (int): The rank of the LoRAConv1d layer.
        lora_alpha (Optional[int], optional): The alpha parameter for LoRAConv1d. Defaults to None.
        lora_dropout (float, optional): The dropout rate for LoRAConv1d. Defaults to 0.0.
        train_bias (bool, optional): Whether to train the bias of LoRAConv1d. Defaults to False. If False, the training of the bias is the same as the origional module.
    """
        
    def __init__(self, 
                parent_module: nn.Conv2d, 
                rank: int, 
                lora_alpha: Optional[int]=None, 
                lora_dropout: float = 0.0, 
                train_bias=False):
        super().__init__(parent_module, rank, lora_alpha, lora_dropout, train_bias)

class LoRAConv3d(LoRAConv):
    """
    A LoRA based Conv3d layer.

    Args:
        parent_module (nn.Conv3d): The parent module of the LoRAConv1d layer.
        rank (int): The rank of the LoRAConv1d layer.
        lora_alpha (Optional[int], optional): The alpha parameter for LoRAConv1d. Defaults to None.
        lora_dropout (float, optional): The dropout rate for LoRAConv1d. Defaults to 0.0.
        train_bias (bool, optional): Whether to train the bias of LoRAConv1d. Defaults to False. If False, the training of the bias is the same as the origional module.
    """
    
    def __init__(self, 
                 parent_module: nn.Conv3d, 
                 rank: int, 
                 lora_alpha: Optional[int]=None, 
                 lora_dropout: float = 0.0, 
                 train_bias=False):
        super().__init__(parent_module, rank, lora_alpha, lora_dropout, train_bias)