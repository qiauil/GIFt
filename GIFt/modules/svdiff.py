r'''
SVDiff: Compact Parameter Space for Diffusion Fine-Tuning: https://arxiv.org/abs/2303.11305

$$
    W = U \mathrm{diag}(\mathrm{ReLU}(\sigma+\delta))V^T 
$$
where
$$
\begin{matrix}
    U \Sigma V^T = \mathrm{SVD}(W_0)\\
    \Sigma=\mathrm{diag}(\sigma)
\end{matrix}
$$
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from ..utils.network_tool import conv_weight_hw,freeze_module
from ..meta_types import FinetuableModule,FinetuningType

class SVDiffLayer(FinetuableModule):
    """
    Base class of SVDiff fine-tuning layer.
    
    Args:
        weight (torch.Tensor): The weight tensor to be decomposed using SVD.
    
    Attributes:
        weight_u (torch.Tensor): The left singular vectors of the weight tensor.
        weight_s (torch.Tensor): The singular values of the weight tensor.
        weight_vh (torch.Tensor): The right singular vectors of the weight tensor.
        spectral_shift (nn.Parameter): The learnable parameter used to shift the singular values.
        spectral_activation (nn.ReLU): The activation function applied to the shifted singular values.
    """
    
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        with torch.no_grad():
            weight_u, weight_s, weight_vh = torch.linalg.svd(weight, full_matrices=False)
        self.spectral_shift = nn.Parameter(weight_s.new_zeros(weight_s.shape))
        self.register_buffer("weight_u",weight_u)
        self.register_buffer("weight_s",weight_s)
        self.register_buffer("weight_vh",weight_vh)
        self.spectral_activation = nn.ReLU()
    
    def svdiff_weight(self):
        """
        Computes the weight transformation using the singular value decomposition (SVD) and the spectral shift.
        
        Returns:
            torch.Tensor: The transformed weight tensor.
        """
        return self.weight_u @ torch.diag(
            self.spectral_activation(self.weight_s + self.spectral_shift)
        ) @ self.weight_vh
            
    def forward(self, x):
        raise NotImplementedError("SVDiffLayer is not a layer, it is a weight transformation method.")
    
class SVDiffLinear(SVDiffLayer):
    """
    SVDiff based Linear layer.

    Args:
        parent_module (nn.Linear): The parent linear module.
        train_bias (bool): Whether to train the bias parameter. Default is False.

    Attributes:
        bias (nn.Parameter): The bias parameter of the linear module.

    """

    def __init__(self, parent_module: nn.Linear, train_bias=False):
        super().__init__(parent_module.weight)
        if hasattr(parent_module, 'bias'):
            if parent_module.bias is not None:
                self.bias = nn.Parameter(parent_module.bias, requires_grad=train_bias)
            else:
                self.bias = None

    def forward(self, x):
        """
        Forward pass of the SVDiffLinear layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the linear transformation.

        """
        return F.linear(x, self.svdiff_weight(), self.bias)
    
class SVDiffConv(SVDiffLayer):
    """
    SVDiff based Convolutional layer.
    You can use `SVDiffConv1d`, `SVDiffConvd`, and `SVDiffConv3d` instead of this class.

    Args:
        parent_module (Union[nn.Conv1d, nn.Conv2d, nn.Conv3d]): The parent convolutional module.
        train_bias (bool): Whether to train the bias or not.

    Attributes:
        parent_module (Union[nn.Conv1d, nn.Conv2d, nn.Conv3d]): The parent convolutional module.
    """

    def __init__(self, parent_module: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d], train_bias=False):
        super().__init__(parent_module.weight.view(conv_weight_hw(parent_module)))
        self.parent_module = parent_module
        freeze_module(self.parent_module, 
                      weights_type=FinetuningType.FINE_TUNE,
                      bias_type=FinetuningType.TRAIN if train_bias else FinetuningType.FREEZE
                      )
    
    def forward(self, x):
        """
        Forward pass of the SVDiffConv layer.

        Args:
            x: The input tensor.

        Returns:
            The output tensor after applying the SVDiffConv operation.
        """
        return self.parent_module._conv_forward(x, self.svdiff_weight().view(self.parent_module.weight.shape), self.parent_module.bias)

class SVDiffConv1d(SVDiffConv):
    """
    A SVDiff based Conv1d layer.
    
    Args:
        parent_module (nn.Conv2d): The parent module to apply singular value differencing to.
        train_bias (bool, optional): Whether to train the bias term of the parent module. Defaults to False.
    """
    
    def __init__(self,
                 parent_module:nn.Conv1d,
                 train_bias=False):
        super().__init__(parent_module,train_bias)

class SVDiffConv2d(SVDiffConv):
    """
    A SVDiff based Conv2d layer.
    
    Args:
        parent_module (nn.Conv2d): The parent module to apply singular value differencing to.
        train_bias (bool, optional): Whether to train the bias term of the parent module. Defaults to False.
    """
        
    def __init__(self,
                parent_module:nn.Conv2d,
                train_bias=False):
        super().__init__(parent_module,train_bias)

class SVDiffConv3d(SVDiffConv):
    """
    A SVDiff based Conv1d layer.
    
    Args:
        parent_module (nn.Conv2d): The parent module to apply singular value differencing to.
        train_bias (bool, optional): Whether to train the bias term of the parent module. Defaults to False.
    """
    
    def __init__(self,
                 parent_module:nn.Conv3d,
                 train_bias=False):
        super().__init__(parent_module,train_bias)